#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse

parser = argparse.ArgumentParser(description="FlashVSR+: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution.")
parser.add_argument("-i", "--input", type=str, help="Path to video file or folder of images")
parser.add_argument("-s", "--scale", type=int, default=4, help="Upscale factor, default=4")
parser.add_argument("-v", "--version", type=str, default="10", choices=["10", "11"], help="Model version, default=10")
parser.add_argument("-m", "--mode", type=str, default="tiny", choices=["tiny", "tiny-long", "full"], help="The type of pipeline to use, default=tiny")
parser.add_argument("--tiled-vae", action="store_true", help="Enable tile decoding")
parser.add_argument("--tiled-dit", action="store_true", help="Enable tile inference")
parser.add_argument("--tile-size", type=int, default=256, help="Chunk size of tile inference, default=256")
parser.add_argument("--overlap", type=int, default=24, help="Overlap size of tile inference, default=24")
parser.add_argument("--unload-dit", action="store_true", help="Unload DiT before decoding")
parser.add_argument("--color-fix", action="store_true", help="Correct output video color")
parser.add_argument("--seed", type=int, default=0, help="Random Seed, default=0")
parser.add_argument("-t", "--dtype", type=str, default="bf16", choices=["fp16", "bf16"], help="Data type for processing, default=bf16")
parser.add_argument("-d", "--device", type=str, default="auto", help="Device to run FlashVSR")
parser.add_argument("-f", "--fps", type=int, default=30, help="Output FPS (for image sequences only), default=30")
parser.add_argument("-q", "--quality", type=int, default=6, help="Output video quality, default=6")
parser.add_argument("-a", "--attention", default="sage", choices=["sage", "block"], help="Attention mode, default=sage")
parser.add_argument("output_folder", type=str, help="Path to save output video")
args = parser.parse_args()

def log(message:str, message_type:str="normal"):
    if message_type == 'error':
        message = '\033[1;41m' + message + '\033[m'
    elif message_type == 'warning':
        message = '\033[1;31m' + message + '\033[m'
    elif message_type == 'finish':
        message = '\033[1;32m' + message + '\033[m'
    elif message_type == 'info':
        message = '\033[1;33m' + message + '\033[m'
    else:
        message = message
    print(f"{message}")

from tqdm import tqdm
log("[FlashVSR] Preparing dependencies...", message_type="finish")

import os
import re
import math
import uuid
import torch
import shutil
import imageio
import ffmpeg
import numpy as np
import torch.nn.functional as F

from PIL import Image
from einops import rearrange
from huggingface_hub import snapshot_download
from src import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from src.models import wan_video_dit
from src.models.TCDecoder import build_tcdecoder
from src.models.utils import get_device_list, clean_vram, Buffer_LQ4x_Proj, Causal_LQ4x_Proj

root = os.path.dirname(os.path.abspath(__file__))
temp = os.path.join(root, "_temp")
devices = get_device_list()

def model_downlod(model_name="JunhaoZhuang/FlashVSR"):
    model_dir = os.path.join(root, "models", model_name.split("/")[-1])
    if not os.path.exists(model_dir):
        log(f"Downloading model '{model_name}' from huggingface...", message_type='info')
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)

def is_ffmpeg_available():
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path is None:
        log("[FlashVSR] FFmpeg not found!", message_type="warning")
        log("Please install FFmpeg and ensure it is in your system's PATH.")
        log("- Windows: Download from https://www.ffmpeg.org/download.html and add the 'bin' directory to PATH.")
        log("- macOS (via Homebrew): brew install ffmpeg")
        log("- Linux (Ubuntu/Debian): sudo apt-get install ffmpeg")
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

def largest_8n1_leq(n):  # 8n+1
    return 0 if n < 1 else ((n - 1)//8)*8 + 1

def next_8n5(n):  # next 8n+5
    return 21 if n < 21 else ((n - 5 + 7) // 8) * 8 + 5

def is_video(path): 
    return os.path.isfile(path) and path.lower().endswith(('.mp4','.mov','.avi','.mkv'))

def save_video(frames, save_path, fps=30, quality=5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    frames_np = (frames.cpu().float() * 255.0).clip(0, 255).numpy().astype(np.uint8)
    w = imageio.get_writer(save_path, fps=fps, quality=quality)
    for frame_np in tqdm(frames_np, desc=f"[FlashVSR] Saving video"):
        w.append_data(frame_np)
    w.close()

def merge_video_with_audio(video_path, audio_source_path):
    temp = video_path+"temp.mp4"
    
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
        output_ffmpeg = ffmpeg.output(
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
                lgo(f"[FlashVSR] Could not remove temporary file '{temp}': {e}", message_type='error')
    
def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("invalid original size")
        
    sW, sH = w0 * scale, h0 * scale
    tW = max(multiple, (sW // multiple) * multiple)
    tH = max(multiple, (sH // multiple) * multiple)
    return sW, sH, tW, tH

def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int) -> torch.Tensor:
    h0, w0, c = frame_tensor.shape
    tensor_bchw = frame_tensor.permute(2, 0, 1).unsqueeze(0) # HWC -> CHW -> BCHW
    
    sW, sH = w0 * scale, h0 * scale
    upscaled_tensor = F.interpolate(tensor_bchw, size=(sH, sW), mode='bicubic', align_corners=False)
    
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
        if total is None or total <= 0 :
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

def get_input_params(image_tensor, scale):
    N0, h0, w0, _ = image_tensor.shape
    
    multiple = 128
    sW, sH, tW, tH = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple)
    num_frames_with_padding = N0 + 4
    F = largest_8n1_leq(num_frames_with_padding)
    
    if F == 0:
        raise RuntimeError(f"Not enough frames after padding. Got {num_frames_with_padding}.")
    
    return tH, tW, F

def input_tensor_generator(image_tensor: torch.Tensor, device, scale: int = 4, dtype=torch.bfloat16):
    """
    一个生成器函数，逐帧处理并 yield 准备好的帧张量，以节省内存。
    产出的每个张量形状为 (C, H, W)。
    """
    N0, h0, w0, _ = image_tensor.shape
    tH, tW, F = get_input_params(image_tensor, scale)
        
    for i in range(F):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx].to(device)
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale=scale, tW=tW, tH=tH)
        tensor_out = tensor_chw * 2.0 - 1.0
        del tensor_chw
        yield tensor_out.to('cpu').to(dtype)

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

def create_feather_mask_numpy(size, overlap):
    H, W = size
    mask = np.ones((H, W, 1), dtype=np.float32)
    ramp = np.linspace(0, 1, overlap, dtype=np.float32)
    
    mask[:, :overlap, :] *= ramp[np.newaxis, :, np.newaxis]
    mask[:, -overlap:, :] *= np.flip(ramp)[np.newaxis, :, np.newaxis]
    
    mask[:overlap, :, :] *= ramp[:, np.newaxis, np.newaxis]
    mask[-overlap:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]
    
    return mask

def create_feather_mask(size, overlap):
    H, W = size
    mask = torch.ones(1, 1, H, W)
    ramp = torch.linspace(0, 1, overlap)
    
    mask[:, :, :, :overlap] = torch.minimum(mask[:, :, :, :overlap], ramp.view(1, 1, 1, -1))
    mask[:, :, :, -overlap:] = torch.minimum(mask[:, :, :, -overlap:], ramp.flip(0).view(1, 1, 1, -1))
    
    mask[:, :, :overlap, :] = torch.minimum(mask[:, :, :overlap, :], ramp.view(1, 1, -1, 1))
    mask[:, :, -overlap:, :] = torch.minimum(mask[:, :, -overlap:, :], ramp.flip(0).view(1, 1, -1, 1))
    
    return mask

def stitch_video_tiles(
    tile_paths, 
    tile_coords, 
    final_dims, 
    scale, 
    overlap, 
    output_path, 
    fps, 
    quality, 
    cleanup=True,
    chunk_size=40  # --- 新增参数：每次在内存中处理的帧数 ---
):
    if not tile_paths:
        log("No tile videos found to stitch.", message_type='error')
        return
    
    final_W, final_H = final_dims
    
    # 1. 一次性打开所有视频文件
    readers = [imageio.get_reader(p) for p in tile_paths]
    
    try:
        # 获取总帧数
        num_frames = readers[0].count_frames()
        if num_frames is None or num_frames <= 0:
            num_frames = len([_ for _ in readers[0]])
            for r in readers: r.close()
            readers = [imageio.get_reader(p) for p in tile_paths]
            
        # 打开最终的写入器
        with imageio.get_writer(output_path, fps=fps, quality=quality) as writer:
            
            # 2. 按 chunk_size 遍历所有帧
            # tqdm 现在描述的是处理了多少个“块”
            for start_frame in tqdm(range(0, num_frames, chunk_size), desc="[FlashVSR] Stitching Chunks"):
                end_frame = min(start_frame + chunk_size, num_frames)
                current_chunk_size = end_frame - start_frame
                
                # 3. 为整个“块”在内存中创建画布
                # 形状: (Frames, Height, Width, Channels)
                chunk_canvas = np.zeros((current_chunk_size, final_H, final_W, 3), dtype=np.float32)
                weight_canvas = np.zeros_like(chunk_canvas, dtype=np.float32)
                
                # 4. 遍历每个分块视频 (tile)
                for i, reader in enumerate(readers):
                    # 5. 一次性读取这个 tile 在当前 chunk 中的所有帧
                    # 这是利用顺序读取的关键优化
                    try:
                        # get_reader().iter_data() 是高效读取连续帧的方式
                        tile_chunk_frames = [
                            frame.astype(np.float32) / 255.0 
                            for idx, frame in enumerate(reader.iter_data()) 
                            if start_frame <= idx < end_frame
                        ]
                        # 将帧列表转换为一个 NumPy 数组
                        tile_chunk_np = np.stack(tile_chunk_frames, axis=0)
                    except Exception as e:
                        log(f"Warning: Could not read chunk from tile {i}. Error: {e}", message_type='warning')
                        continue
                    
                    if tile_chunk_np.shape[0] != current_chunk_size:
                        log(f"Warning: Tile {i} chunk has incorrect frame count. Skipping.", message_type='warning')
                        continue
                    
                    # 6. 创建羽化蒙版 (只需要创建一次)
                    tile_H, tile_W, _ = tile_chunk_np.shape[1:]
                    ramp = np.linspace(0, 1, overlap * scale, dtype=np.float32)
                    mask = np.ones((tile_H, tile_W, 1), dtype=np.float32)
                    mask[:, :overlap*scale, :] *= ramp[np.newaxis, :, np.newaxis]
                    mask[:, -overlap*scale:, :] *= np.flip(ramp)[np.newaxis, :, np.newaxis]
                    mask[:overlap*scale, :, :] *= ramp[:, np.newaxis, np.newaxis]
                    mask[-overlap*scale:, :, :] *= np.flip(ramp)[:, np.newaxis, np.newaxis]
                    # 扩展蒙版以匹配 chunk 的帧数维度
                    mask_4d = mask[np.newaxis, :, :, :] # 形状: (1, H, W, C)
                    
                    # 7. 在内存中拼接整个 chunk
                    x1_orig, y1_orig, _, _ = tile_coords[i]
                    out_y1, out_x1 = y1_orig * scale, x1_orig * scale
                    out_y2, out_x2 = out_y1 + tile_H, out_x1 + tile_W
                    
                    # 使用 NumPy 的广播机制 (broadcasting)
                    chunk_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += tile_chunk_np * mask_4d
                    weight_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_4d
                    
                # 8. 归一化整个 chunk
                weight_canvas[weight_canvas == 0] = 1.0
                stitched_chunk = chunk_canvas / weight_canvas
                
                # 9. 将这个 chunk 的所有帧一次性写入文件
                for frame_idx_in_chunk in range(current_chunk_size):
                    frame_uint8 = (np.clip(stitched_chunk[frame_idx_in_chunk], 0, 1) * 255).astype(np.uint8)
                    writer.append_data(frame_uint8)
                    
    finally:
        log("Closing all tile reader instances...")
        for reader in readers:
            reader.close()
            
    if cleanup:
        log("Cleaning up temporary tile files...")
        for path in tile_paths:
            try:
                os.remove(path)
            except OSError as e:
                log(f"Could not remove temporary file '{path}': {e}", message_type='warning')

def init_pipeline(version, mode, device, dtype):
    if version == "10":
        model = "FlashVSR"
    else:
        model = "FlashVSR-v1.1"
    model_downlod(model_name="JunhaoZhuang/" + model)
    model_path = os.path.join(root, "models", model)
    if not os.path.exists(model_path):
        raise RuntimeError(f'Model directory does not exist! Please save all weights to "{model_path}"')
    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f'"diffusion_pytorch_model_streaming_dmd.safetensors" does not exist! Please save it to "{model_path}"')
    vae_path = os.path.join(model_path, "Wan2.1_VAE.pth")
    if not os.path.exists(vae_path):
        raise RuntimeError(f'"Wan2.1_VAE.pth" does not exist! Please save it to "{model_path}"')
    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    if not os.path.exists(lq_path):
        raise RuntimeError(f'"LQ_proj_in.ckpt" does not exist! Please save it to "{model_path}"')
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    if not os.path.exists(tcd_path):
        raise RuntimeError(f'"TCDecoder.ckpt" does not exist! Please save it to "{model_path}"')
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
        
    if model == "FlashVSR":
        pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    else:
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)
    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit","vae"])
    
    return pipe

def main(input, version, mode, scale, color_fix, tiled_vae, tiled_dit, tile_size, tile_overlap, unload_dit, dtype, sparse_ratio=2, kv_ratio=3, local_range=11, seed=0, device="auto", quality=6, output=None):
    _device = device
    if device == "auto":
        _device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else device
    if _device == "auto" or _device not in devices:
        raise RuntimeError("No devices found to run FlashVSR!")
    if _device.startswith("cuda"):
        torch.cuda.set_device(_device)
        
    if tiled_dit and (tile_overlap > tile_size / 2):
        raise ValueError('The "tile_overlap" must be less than half of "tile_size"!')
    
    _frames, fps = prepare_tensors(input, dtype=dtype)
    _fps = fps if is_video(input) else args.fps
    
    add = next_8n5(_frames.shape[0]) - _frames.shape[0]
    padding_frames = _frames[-1:, :, :, :].repeat(add, 1, 1, 1)
    frames = torch.cat([_frames, padding_frames], dim=0)
    frame_count = _frames.shape[0]
    del _frames
    clean_vram()
    
    log("[FlashVSR] Preparing frames...", message_type="finish")
    
    if tiled_dit:
        N, H, W, C = frames.shape
        num_aligned_frames = largest_8n1_leq(N + 4) - 4
        
        if mode == "tiny-long":
            local_temp = os.path.join(temp, str(uuid.uuid4()))
            os.makedirs(local_temp, exist_ok=True)
        else:
            final_output_canvas = torch.zeros(
                (num_aligned_frames, H * scale, W * scale, C), 
                dtype=dtype, 
                device="cpu"
            )
            weight_sum_canvas = torch.zeros_like(final_output_canvas)
            
        tile_coords = calculate_tile_coords(H, W, tile_size, tile_overlap)
        latent_tiles_cpu = []
        temp_videos = []
        
        pipe = init_pipeline(version, mode, _device, dtype)
        
        for i, (x1, y1, x2, y2) in enumerate(tile_coords):
            input_tile = frames[:, y1:y2, x1:x2, :]
                
            if mode == "tiny-long":
                temp_name = os.path.join(local_temp, f"{i+1:05d}.mp4") 
                th, tw, F = get_input_params(input_tile, scale=scale)
                LQ_tile = input_tensor_generator(input_tile, _device, scale=scale, dtype=dtype)
            else:
                LQ_tile, th, tw, F = prepare_input_tensor(input_tile, _device, scale=scale, dtype=dtype)
                LQ_tile = LQ_tile.to(_device)
            
            if i == 0:
                log(f"[FlashVSR] Processing {frame_count} frames...", message_type='info')
            log(f"[FlashVSR] Processing tile {i+1}/{len(tile_coords)}: ({x1},{y1}) to ({x2},{y2})", message_type='info')
            
            output_tile_gpu = pipe(
                prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
                LQ_video=LQ_tile, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
                color_fix=color_fix, unload_dit=unload_dit, fps=_fps, quality=10, output_path=temp_name, tiled_dit=True
            )
            
            temp_videos.append(temp_name)
            if mode == "tiny-long":
                final_output = output_tile_gpu
                del LQ_tile, input_tile
                clean_vram()
                continue
            
            processed_tile_cpu = tensor2video(output_tile_gpu).to("cpu")
            
            mask_nchw = create_feather_mask(
                (processed_tile_cpu.shape[1], processed_tile_cpu.shape[2]),
                tile_overlap * scale
            ).to("cpu")
            mask_nhwc = mask_nchw.permute(0, 2, 3, 1)
            out_x1, out_y1 = x1 * scale, y1 * scale
            
            tile_H_scaled = processed_tile_cpu.shape[1]
            tile_W_scaled = processed_tile_cpu.shape[2]
            out_x2, out_y2 = out_x1 + tile_W_scaled, out_y1 + tile_H_scaled
            final_output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += processed_tile_cpu * mask_nhwc
            weight_sum_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask_nhwc
            
            del LQ_tile, output_tile_gpu, processed_tile_cpu, input_tile
            clean_vram()
        
        if mode == "tiny-long":
            stitch_video_tiles(tile_paths=temp_videos, tile_coords=tile_coords, final_dims=(W * scale, H * scale),
                scale=scale, overlap=tile_overlap, output_path=output, fps=_fps, quality=quality, cleanup=True
            )
            shutil.rmtree(local_temp)
        else:
            weight_sum_canvas[weight_sum_canvas == 0] = 1.0
            final_output = final_output_canvas / weight_sum_canvas
    else:
        if mode == "tiny-long":
            th, tw, F = get_input_params(frames, scale=scale)
            LQ = input_tensor_generator(frames, _device, scale=scale, dtype=dtype)
        else:
            LQ, th, tw, F = prepare_input_tensor(frames, _device, scale=scale, dtype=dtype)
            LQ = LQ.to(_device)

        pipe = init_pipeline(version, mode, _device, dtype)
        log(f"[FlashVSR] Processing {frame_count} frames...", message_type='info')
        video = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed, tiled=tiled_vae,
            LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio*768*1280/(th*tw), kv_ratio=kv_ratio, local_range=local_range,
            color_fix = color_fix, unload_dit=unload_dit, fps=_fps, output_path=output, tiled_dit=True
        )
        
        if mode == "tiny-long":
            del pipe, LQ
            clean_vram()
            return video, _fps
        
        log("[FlashVSR] Preparing frames...")
        final_output = tensor2video(video).to("cpu")
        del pipe, video, LQ
        clean_vram()
    
    return final_output[:frame_count, :, :, :], fps

if __name__ == "__main__":
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    try:
        dtype = dtype_map[args.dtype]
    except:
        dtype = torch.bfloat16
        
    if args.attention == "sage":
        wan_video_dit.USE_BLOCK_ATTN = False
    else:
        wan_video_dit.USE_BLOCK_ATTN = True
    
    if os.path.exists(temp):
        shutil.rmtree(temp)
    os.makedirs(temp, exist_ok=True)
    name = os.path.basename(args.input.rstrip('/'))
    final = os.path.join(args.output_folder, f"FlashVSR_{args.mode}_{name.split('.')[0]}_{args.seed}.mp4")
    result, fps = main(args.input, args.version, args.mode, args.scale, args.color_fix, args.tiled_vae, args.tiled_dit,args.tile_size,
        args.overlap, args.unload_dit, dtype, seed=args.seed, device=args.device, quality=args.quality, output=final)
    if args.mode != "tiny-long":
        save_video(result, final, fps=fps, quality=args.quality)
        
    merge_video_with_audio(final, args.input)
    log("[FlashVSR] Done.", message_type='finish')