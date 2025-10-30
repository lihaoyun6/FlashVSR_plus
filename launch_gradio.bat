@echo off

echo ================================================
echo    FlashVSR+ Gradio Interface Launcher
echo ================================================
echo.

REM Check if conda environment exists
where conda >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Activating conda environment 'flashvsr'...
    call conda activate flashvsr
    if %ERRORLEVEL% NEQ 0 (
        echo [WARNING] Could not activate conda environment 'flashvsr'
        echo [INFO] Make sure the environment exists or activate it manually
        echo.
    )
) else (
    REM Try to activate venv if it exists
    if exist "venv\Scripts\activate.bat" (
        echo [INFO] Activating virtual environment...
        call venv\Scripts\activate.bat
    ) else (
        echo [WARNING] No conda or venv found
        echo [INFO] Proceeding with current Python environment...
        echo.
    )
)

REM Launch the Gradio app
echo ================================================
echo    Starting Gradio Interface...
echo ================================================
echo.
echo    The browser will open automatically at:
echo    http://127.0.0.1:7860
echo.
echo    Press Ctrl+C to stop the server
echo.

python gradio_app.py

pause
