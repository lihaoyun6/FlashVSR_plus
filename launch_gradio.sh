echo "🚀 Launching FlashVSR+ Gradio Interface..."
echo ""

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Activating conda environment 'flashvsr'..."
    eval "$(conda shell.bash hook)"
    conda activate flashvsr
    if [ $? -ne 0 ]; then
        echo "⚠️  Could not activate conda environment 'flashvsr'"
        echo "Make sure the environment exists or activate it manually"
        echo ""
    fi
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No conda or venv found"
    echo "Proceeding with current Python environment..."
    echo ""
fi

# Launch the Gradio app
echo "🌐 Starting Gradio interface..."
echo "   The browser will open automatically at:"
echo "   http://127.0.0.1:7860"
echo ""
echo "   Press Ctrl+C to stop"
echo ""

python gradio_app.py
