import gradio as gr
import os
import sys
import torch
import tempfile
import shutil
import math
import re
import imageio
import ffmpeg
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from huggingface_hub import snapshot_download

# Import FlashVSR components
from src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from src.models.TCDecoder import build_tcdecoder
from src.models.utils import get_device_list, clean_vram, Buffer_LQ4x_Proj

root = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# Utility functions from run.py
# ============================================================================

def log(message: str, message_type: str = "normal"):
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    print(f"{message}")

def model_downlod(model_name="JunhaoZhuang/FlashVSR"):
    model_dir = os.path.join(root, "models", "FlashVSR")
    if not os.path.exists(model_dir):
        log(f"Downloading model '{model_name}' from huggingface...", message_type='info')
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)

def is_ffmpeg_available():
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        log("[FlashVSR] FFmpeg not found!", message_type="warning")
        return False
    return True

def tensor2video(frames: torch.Tensor):
    video_squeezed = frames.squeeze(0)
    video_permuted = rearrange(video_squeezed, "C F H W -> F H W C")
    video_final = (video_permuted.float() + 1.0) / 2.0
    return video_final

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', os.path.basename(name))]

def list_images_natural(folder: str):
    exts = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    fs = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(exts)]
    fs.sort(key=natural_key)
    return fs

def largest_8n1_leq(n):
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def is_video(path):
    return os.path.isfile(path) and path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))

def save_video(frames, save_path, fps=30, quality=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    frames_np = (frames.cpu().float() * 255.0).clip(0, 255).numpy().astype(np.uint8)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for frame_np in tqdm(frames_np, desc=f"[FlashVSR] Saving video"):
        w.append_data(frame_np)
    w.close()

def merge_video_with_audio(video_path, audio_source_path):
    temp = video_path + "temp.mp4"
    
    if os.path.isdir(audio_source_path):
        log(f"[FlashVSR] Output video saved to '{video_path}'", message_type='info')
        return
    
    if not is_ffmpeg_available():
        log(f"[FlashVSR] Output video saved to '{video_path}'", message_type='info')
        return
    
    try:
        probe = ffmpeg.probe(audio_source_path)
        audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
        if not audio_streams:
            log(f"[FlashVSR] Output video saved to '{video_path}'", message_type='info')
            return
        log("[FlashVSR] Copying audio tracks...")
        os.rename(video_path, temp)
        input_video = ffmpeg.input(temp)['v']
        input_audio = ffmpeg.input(audio_source_path)['a']
        ffmpeg.output(
            input_video, input_audio, video_path,
            vcodec='copy', acodec='copy'
        ).run(overwrite_output=True, quiet=True)
        log(f"[FlashVSR] Output video saved to '{video_path}'", message_type='info')
    except ffmpeg.Error as e:
        print("[ERROR] FFmpeg error during merge:", e.stderr.decode() if e.stderr else "Unknown error")
        log(f"[FlashVSR] Audio merge failed. A silent video has been saved to '{video_path}'.", message_type='warning')
    finally:
        if os.path.exists(temp):
            try:
                os.remove(temp)
            except OSError as e:
                log(f"[FlashVSR] Could not remove temporary file '{temp}': {e}", message_type='error')

def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")
    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH

def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int) -> torch.Tensor:
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0)
    sW, sH = w0 * scale, h0 * scale
    upscaled_tensor = torch.nn.functional.interpolate(tensor_bchw, size=(sH, sW), mode='bicubic', align_corners=False)
    l = max(0, (sW - tW) // 2)
    t = max(0, (sH - tH) // 2)
    cropped_tensor = upscaled_tensor[:, :, t:t + tH, l:l + tW]
    return cropped_tensor.squeeze(0)

def prepare_tensors(path: str, dtype=torch.bfloat16):
    if os.path.isdir(path):
        paths0 = list_images_natural(path)
        if not paths0:
            raise FileNotFoundError(f"No images in {path}")
        with Image.open(paths0[0]) as _img0:
            w0, h0 = _img0.size
        N0 = len(paths0)
        frames = []
        for p in paths0:
            with Image.open(p).convert('RGB') as img:
                img_np = np.array(img).astype(np.float32) / 255.0
                frames.append(torch.from_numpy(img_np).to(dtype))
        vid = torch.stack(frames, 0)
        fps = 30
        return vid, fps

    if is_video(path):
        rdr = imageio.get_reader(path)
        meta = {}
        try:
            meta = rdr.get_meta_data()
            first_frame = rdr.get_data(0)
            h0, w0, _ = first_frame.shape
        except Exception:
            first_frame = rdr.get_data(0)
            h0, w0, _ = first_frame.shape
        fps_val = meta.get('fps', 30)
        fps = int(round(fps_val)) if isinstance(fps_val, (int, float)) else 30
        total = meta.get('nframes', rdr.count_frames())
        if total is None or total <= 0:
            total = len([_ for _ in rdr])
            rdr = imageio.get_reader(path)
        if total <= 0:
            rdr.close()
            raise RuntimeError(f"Cannot read frames from {path}")
        frames = []
        try:
            for frame_data in rdr:
                frame_np = frame_data.astype(np.float32) / 255.0
                frames.append(torch.from_numpy(frame_np).to(dtype))
        finally:
            try:
                rdr.close()
            except Exception:
                pass
        vid = torch.stack(frames, 0)
        return vid, fps
    raise ValueError(f"Unsupported input: {path}")

def prepare_input_tensor(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    N0, h0, w0, _ = image_tensor.shape
    multiple = 128
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    F = largest_8n1_leq(num_frames_with_padding)
    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")
    frames = []
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale=scale, tW=tW, tH=tH)
        tensor_out = tensor_chw * 2.0 - 1.0
        tensor_out = tensor_out.to('cpu').to(dtype)
        frames.append(tensor_out)
    vid_stacked = torch.stack(frames, 0)
    vid_final = vid_stacked.permute(1, 0, 2, 3).unsqueeze(0)
    del vid_stacked
    clean_vram()
    return vid_final, tH, tW, F

def calculate_tile_coords(height, width, tile_size, overlap):
    coords = []
    stride = tile_size - overlap
    num_rows = math.ceil((height - overlap) / stride)
    num_cols = math.ceil((width - overlap) / stride)
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * stride
            x1 = c * stride
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            if y2 - y1 < tile_size:
                y1 = max(0, y2 - tile_size)
            if x2 - x1 < tile_size:
                x1 = max(0, x2 - tile_size)
            coords.append((x1, y1, x2, y2))
    return coords

def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    return mask

def init_pipeline(mode, device, dtype):
    model_downlod()
    model_path = os.path.join(root, "models", "FlashVSR")
    if not os.path.exists(model_path):
        raise RuntimeError(f'Model directory does not exist! Please save all weights to "{model_path}"')
    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f'"diffusion_pytorch_model_streaming_dmd.safetensors" does not exist!')
    vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
    if not os.path.exists(vae_path):
        raise RuntimeError(f'"Wan2.1_VAE.pth" does not exist!')
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    if not os.path.exists(lq_path):
        raise RuntimeError(f'"LQ_proj_in.ckpt" does not exist!')
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    if not os.path.exists(tcd_path):
        raise RuntimeError(f'"TCDecoder.ckpt" does not exist!')
    prompt_path = os.path.join(root, "models", "posi_prompt.pth")
    
    mm = ModelManager(torch_dtype=dtype, device="cpu")
    if mode == "full":
        mm.load_models([ckpt_path, vae_path])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        mm.load_models([ckpt_path])
        if mode == "tiny":
            pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        else:
            pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)
        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
        mis = pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location=device), strict=False)
        pipe.TCDecoder.clean_mem()
    
    pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    if os.path.exists(lq_path):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit", "vae"])
    return pipe

def process_video_main(input_path, mode, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, 
                       unload_dit, dtype, sparse_ratio=2, kv_ratio=3, local_range=11, seed=0, device="auto"):
    _device = device
    if device == "auto":
        _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    devices = get_device_list()
    if _device == "auto" or _device not in devices:
        raise RuntimeError("No devices found to run FlashVSR!")
    if _device.startswith("cuda"):
        torch.cuda.set_device(_device)
    
    if tiled_dit and (tile_overlap > tile_size / 2):
        raise ValueError('The "tile_overlap" must be less than half of "tile_size"!')
    
    frames, fps = prepare_tensors(input_path, dtype=dtype)
    
    if frames.shape[0] < 21:
        raise ValueError(f"Number of frames must be at least 21, got {frames.shape[0]}")
    
    if tiled_dit:
        log("[FlashVSR] Preparing frames...")
        N, H, W, C = frames.shape
        num_aligned_frames = largest_8n1_leq(N + 4) - 4
        final_output_canvas = torch.zeros((num_aligned_frames, H * scale, W * scale, C), dtype=torch.float32, device="cpu")
        weight_sum_canvas = torch.zeros_like(final_output_canvas)
        tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
        
        pipe = init_pipeline(mode, _device, dtype)
        
        for i, (x1, y1, x2, y2) in enumerate(tile_coords):
            log(f"[FlashVSR] Processing tile {i+1}/{len(tile_coords)}: coords ({x1},{y1}) to ({x2},{y2})", message_type='info')
            input_tile = frames[:, y1:y2, x1:x2, :]
            LQ_tile, th, tw, F = prepare_input_tensor(input_tile, _device, scale=scale, dtype=dtype)
            if "long" not in mode:
                LQ_tile = LQ_tile.to(_device)
            
            output_tile_gpu = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                LQ_video=LQ_tile, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                color_fix=color_fix, unload_dit=unload_dit
            )
            
            processed_tile_cpu = tensor2video(output_tile_gpu).to("cpu")
            mask_nchw = create_feather_mask((processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]), tile_overlap * scale).to("cpu")
            mask_nhwc = mask_nchw.permute(0, 2, 3, 1)
            out_x1, out_y1 = x1 * scale, y1 * scale
            tile_H_scaled = processed_tile_cpu.shape[1]
            tile_W_scaled = processed_tile_cpu.shape[2]
            out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
            final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
            weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
            del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
            clean_vram()
        
        weight_sum_canvas[weight_sum_canvas == 0] = 1.0
        final_output = final_output_canvas / weight_sum_canvas
    else:
        log("[FlashVSR] Preparing frames...")
        LQ, th, tw, F = prepare_input_tensor(frames, _device, scale=scale, dtype=dtype)
        if "long" not in mode:
            LQ = LQ.to(_device)
        
        pipe = init_pipeline(mode, _device, dtype)
        log(f"[FlashVSR] Processing {frames.shape[0]} frames...", message_type='info')
        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
            LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
            color_fix=color_fix, unload_dit=unload_dit
        )
        log("[FlashVSR] Preparing output...")
        final_output = tensor2video(video).to("cpu")
        del pipe, video, LQ
        clean_vram()
    
    return final_output, fps

# ============================================================================
# Gradio Interface
# ============================================================================

def process_batch_gradio(
    input_files,
    scale,
    mode,
    tiled_vae,
    tiled_dit,
    tile_size,
    overlap,
    unload_dit,
    color_fix,
    seed,
    dtype,
    device,
    quality
):
    """Process multiple videos with FlashVSR+ and return results"""
    
    if not input_files or len(input_files) == 0:
        return [], "❌ Please upload at least one video file!"
    
    results = []
    success_count = 0
    failed_files = []
    
    try:
        # Create outputs directory
        output_dir = os.path.join(root, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Map dtype string to torch dtype
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        
        total_files = len(input_files)
        
        for idx, input_video in enumerate(input_files):
            try:
                log(f"🎬 Processing {idx+1}/{total_files}: {Path(input_video).name}", message_type='info')
                
                # Validate tile settings
                if tiled_dit and overlap >= tile_size / 2:
                    failed_files.append(f"{Path(input_video).name} (tile overlap error)")
                    continue
                
                # Process the video
                result, fps = process_video_main(
                    input_path=input_video,
                    mode=mode,
                    scale=scale,
                    color_fix=color_fix,
                    tiled_vae=tiled_vae,
                    tiled_dit=tiled_dit,
                    tile_size=tile_size,
                    tile_overlap=overlap,
                    unload_dit=unload_dit,
                    dtype=torch_dtype,
                    seed=seed,
                    device=device
                )
                
                # Save the output video
                input_name = Path(input_video).stem
                output_path = os.path.join(output_dir, f"FlashVSR_{mode}_{input_name}_{seed}.mp4")
                save_video(result, output_path, fps=fps, quality=quality)
                
                # Merge audio if available
                merge_video_with_audio(output_path, input_video)
                
                results.append(output_path)
                success_count += 1
                
                # Clean up memory
                del result
                clean_vram()
                
            except Exception as e:
                print(f"Error processing {Path(input_video).name}: {e}")
                failed_files.append(f"{Path(input_video).name} ({str(e)[:50]}...)")
                continue
        
        log("✅ Batch processing complete!", message_type='finish')
        
        # Create status message
        status_msg = f"""
✅ **Batch Processing Complete!**

📊 **Summary:**
- Total files: {total_files}
- Successfully processed: {success_count}
- Failed: {len(failed_files)}

⚙️ **Settings:**
- Mode: {mode}
- Scale: {scale}x
- Seed: {seed}

💾 **Output folder:** `outputs/`
"""
        
        if failed_files:
            status_msg += "\n\n❌ **Failed files:**\n"
            for failed in failed_files:
                status_msg += f"- {failed}\n"
        
        return results, status_msg
        
    except Exception as e:
        error_msg = f"❌ **Error during batch processing:**\n\n{str(e)}"
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return [], error_msg

# ============================================================================
# Gradio Interface
# ============================================================================


def get_available_devices():
    """Get list of available devices"""
    devices = ["auto"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices

def process_video_gradio(
    input_video,
    scale,
    mode,
    tiled_vae,
    tiled_dit,
    tile_size,
    overlap,
    unload_dit,
    color_fix,
    seed,
    dtype,
    device,
    quality
):
    """Process video with FlashVSR+ and return the result"""
    
    if input_video is None:
        return None, "❌ Please upload a video file first!"
    
    try:
        # Create outputs directory in FlashVSR_plus folder
        output_dir = os.path.join(root, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        log("🔄 Initializing models...", message_type='info')
        
        # Validate tile settings
        if tiled_dit and overlap >= tile_size / 2:
            return None, "❌ Error: Overlap must be less than half of tile size!"
        
        # Map dtype string to torch dtype
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        
        log("📹 Loading video...", message_type='info')
        
        # Process the video
        result, fps = process_video_main(
            input_path=input_video,
            mode=mode,
            scale=scale,
            color_fix=color_fix,
            tiled_vae=tiled_vae,
            tiled_dit=tiled_dit,
            tile_size=tile_size,
            tile_overlap=overlap,
            unload_dit=unload_dit,
            dtype=torch_dtype,
            seed=seed,
            device=device
        )
        
        log("💾 Saving output video...", message_type='info')
        
        # Save the output video
        input_name = Path(input_video).stem
        output_path = os.path.join(output_dir, f"FlashVSR_{mode}_{input_name}_{seed}.mp4")
        save_video(result, output_path, fps=fps, quality=quality)
        
        # Merge audio if available
        merge_video_with_audio(output_path, input_video)
        
        log("✅ Complete!", message_type='finish')
        
        status_msg = f"""
✅ **Processing Complete!**

📊 **Details:**
- Mode: {mode}
- Scale: {scale}x
- Frames: {result.shape[0]}
- Resolution: {result.shape[2]}x{result.shape[1]}
- FPS: {fps}
- Seed: {seed}

💾 **Saved to:** `outputs/{os.path.basename(output_path)}`
        """
        
        return output_path, status_msg
        
    except Exception as e:
        error_msg = f"❌ **Error during processing:**\n\n{str(e)}"
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, error_msg

# Create Gradio interface
with gr.Blocks(title="FlashVSR+ - Video Super Resolution", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # ⚡ FlashVSR+ : Video Super Resolution
    
    **Real-Time Diffusion-Based Streaming Video Super-Resolution**
    
    Enhance your videos with AI-powered super-resolution. Upscale by 4x while maintaining high quality.
    
    ⭐ [GitHub Repository](https://github.com/lihaoyun6/FlashVSR_plus) | 📄 [Paper](https://arxiv.org/abs/2510.12747)
    """)
    
    with gr.Tabs():
        # ===================================================================
        # SINGLE VIDEO TAB
        # ===================================================================
        with gr.Tab("🎬 Single Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Input")
                    
                    input_video = gr.Video(
                        label="Upload Video",
                        sources=["upload"],
                        height=300
                    )
                    
                    with gr.Accordion("⚙️ Basic Settings", open=True):
                        scale = gr.Slider(
                            minimum=2,
                            maximum=4,
                            step=1,
                            value=4,
                            label="Upscale Factor",
                            info="Recommended: 4x for best results"
                        )
                        
                        mode = gr.Radio(
                            choices=["tiny", "tiny-long", "full"],
                            value="tiny",
                            label="Pipeline Mode",
                            info="tiny: Fast & efficient | tiny-long: Better for longer videos | full: Highest quality"
                        )
                        
                        seed = gr.Number(
                            value=0,
                            label="Random Seed",
                            info="Set to 0 for random, or use specific value for reproducibility",
                            precision=0
                        )
                    
                    with gr.Accordion("🎨 Enhancement Options", open=True):
                        color_fix = gr.Checkbox(
                            label="Color Correction",
                            value=False,
                            info="Apply color correction to output"
                        )
                        
                        quality = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=6,
                            label="Output Video Quality",
                            info="Higher = better quality but larger file size"
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Output")
                    
                    output_video = gr.Video(
                        label="Enhanced Video",
                        height=300
                    )
                    
                    status_text = gr.Markdown("Ready to process. Upload a video and click 'Process Video'.")
                    
                    with gr.Accordion("🔧 Advanced Settings", open=False):
                        tiled_vae = gr.Checkbox(
                            label="Enable Tiled VAE",
                            value=False,
                            info="Reduces VRAM usage during decoding"
                        )
                        
                        tiled_dit = gr.Checkbox(
                            label="Enable Tiled DiT Inference",
                            value=False,
                            info="Process video in tiles (8GB VRAM can handle 1080p)"
                        )
                        
                        with gr.Row():
                            tile_size = gr.Slider(
                                minimum=128,
                                maximum=512,
                                step=64,
                                value=256,
                                label="Tile Size",
                                info="Larger = faster but more VRAM"
                            )
                            
                            overlap = gr.Slider(
                                minimum=8,
                                maximum=64,
                                step=8,
                                value=24,
                                label="Tile Overlap",
                                info="Overlap between tiles (must be < tile_size/2)"
                            )
                        
                        unload_dit = gr.Checkbox(
                            label="Unload DiT Before Decoding",
                            value=False,
                            info="Saves VRAM at cost of speed"
                        )
                        
                        dtype = gr.Radio(
                            choices=["bf16", "fp16"],
                            value="bf16",
                            label="Data Type",
                            info="bf16 recommended for most GPUs"
                        )
                        
                        device = gr.Dropdown(
                            choices=get_available_devices(),
                            value="auto",
                            label="Device",
                            info="Select computation device"
                        )
                    
                    process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
                    
                    gr.Markdown("""
                    ### 💡 Tips
                    
                    - **First run**: Models will be downloaded automatically (~3GB)
                    - **Best quality**: Use `scale=4` and `mode=tiny` or `mode=full`
                    - **Low VRAM**: Enable `Tiled VAE` and `Tiled DiT`, then adjust tile size
                    - **8GB VRAM**: Can process 1080p with tiled inference
                    - **Minimum frames**: Video must have at least 21 frames
                    
                    ### 📊 Performance Guide
                    
                    | GPU VRAM | Recommended Settings |
                    |----------|---------------------|
                    | 8GB      | Tiled DiT ON, tile_size=256 |
                    | 12GB     | Tiled VAE ON (optional) |
                    | 16GB+    | All tiles OFF for best speed |
                    """)
            
            # Connect the button to the processing function
            process_btn.click(
                fn=process_video_gradio,
                inputs=[
                    input_video,
                    scale,
                    mode,
                    tiled_vae,
                    tiled_dit,
                    tile_size,
                    overlap,
                    unload_dit,
                    color_fix,
                    seed,
                    dtype,
                    device,
                    quality
                ],
                outputs=[output_video, status_text]
            )
        
        # ===================================================================
        # BATCH PROCESSING TAB
        # ===================================================================
        with gr.Tab("📦 Batch Processing"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Batch Input")
                    
                    batch_input_files = gr.Files(
                        label="Upload Multiple Videos",
                        file_types=["video"],
                        file_count="multiple",
                        height=300
                    )
                    
                    gr.Markdown("""
                    **Upload multiple video files to process them all with the same settings.**
                    
                    - Select multiple files at once
                    - All videos will be processed sequentially
                    - Progress will be shown for each file
                    - Failed files will be reported at the end
                    """)
                    
                    with gr.Accordion("⚙️ Batch Settings", open=True):
                        batch_scale = gr.Slider(
                            minimum=2,
                            maximum=4,
                            step=1,
                            value=4,
                            label="Upscale Factor",
                            info="Applied to all videos"
                        )
                        
                        batch_mode = gr.Radio(
                            choices=["tiny", "tiny-long", "full"],
                            value="tiny",
                            label="Pipeline Mode",
                            info="tiny recommended for batch processing"
                        )
                        
                        batch_seed = gr.Number(
                            value=0,
                            label="Random Seed",
                            info="Same seed for all videos",
                            precision=0
                        )
                        
                        batch_color_fix = gr.Checkbox(
                            label="Color Correction",
                            value=False,
                            info="Apply to all videos"
                        )
                        
                        batch_quality = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=6,
                            label="Output Video Quality"
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Batch Output")
                    
                    batch_output_gallery = gr.Gallery(
                        label="Processed Videos",
                        show_label=True,
                        columns=2,
                        rows=2,
                        height=300,
                        object_fit="contain"
                    )
                    
                    batch_status_text = gr.Markdown("Ready for batch processing. Upload multiple videos and click 'Process Batch'.")
                    
                    with gr.Accordion("🔧 Advanced Settings", open=False):
                        batch_tiled_vae = gr.Checkbox(
                            label="Enable Tiled VAE",
                            value=False
                        )
                        
                        batch_tiled_dit = gr.Checkbox(
                            label="Enable Tiled DiT Inference",
                            value=False
                        )
                        
                        with gr.Row():
                            batch_tile_size = gr.Slider(
                                minimum=128,
                                maximum=512,
                                step=64,
                                value=256,
                                label="Tile Size"
                            )
                            
                            batch_overlap = gr.Slider(
                                minimum=8,
                                maximum=64,
                                step=8,
                                value=24,
                                label="Tile Overlap"
                            )
                        
                        batch_unload_dit = gr.Checkbox(
                            label="Unload DiT Before Decoding",
                            value=False
                        )
                        
                        batch_dtype = gr.Radio(
                            choices=["bf16", "fp16"],
                            value="bf16",
                            label="Data Type"
                        )
                        
                        batch_device = gr.Dropdown(
                            choices=get_available_devices(),
                            value="auto",
                            label="Device"
                        )
                    
                    batch_process_btn = gr.Button("🚀 Process Batch", variant="primary", size="lg")
                    
                    gr.Markdown("""
                    ### 💡 Batch Processing Tips
                    
                    - **Memory Management**: Memory is cleared between each video
                    - **Failed Files**: Will be reported but won't stop the batch
                    - **Output Location**: All files saved to `FlashVSR_plus/outputs/`
                    - **Recommended**: Use `tiny` mode for faster batch processing
                    - **Progress**: Watch the progress bar for real-time status
                    
                    ### ⚠️ Important
                    
                    - Processing time = (single video time) × (number of videos)
                    - Keep all videos with similar specs for consistent results
                    - Enable tiling if processing high-resolution videos
                    """)
            
            # Connect the batch button to the batch processing function
            batch_process_btn.click(
                fn=process_batch_gradio,
                inputs=[
                    batch_input_files,
                    batch_scale,
                    batch_mode,
                    batch_tiled_vae,
                    batch_tiled_dit,
                    batch_tile_size,
                    batch_overlap,
                    batch_unload_dit,
                    batch_color_fix,
                    batch_seed,
                    batch_dtype,
                    batch_device,
                    batch_quality
                ],
                outputs=[batch_output_gallery, batch_status_text]
            )
    
    gr.Markdown("""
    ---
    
    ### 📚 Citation
    
    If you use FlashVSR in your research, please cite:
    
    ```bibtex
    @misc{zhuang2025flashvsrrealtimediffusionbasedstreaming,
          title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution}, 
          author={Junhao Zhuang and Shi Guo and Xin Cai and Xiaohui Li and Yihao Liu and Chun Yuan and Tianfan Xue},
          year={2025},
          eprint={2510.12747},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }
    ```
    
    Made with ❤️ using [Gradio](https://gradio.app)
    """)

if __name__ == "__main__":
    # Download models before launching
    print("Checking for required models...")
    model_downlod()
    
    # Launch the interface
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )
