@echo off
echo 🧠 Starting Upgraded Melvin Unified Brain System
echo ================================================

if not exist melvin_upgraded.exe (
    echo ❌ melvin_upgraded.exe not found!
    echo Please run build_upgraded_melvin.bat first.
    pause
    exit /b 1
)

echo Starting Upgraded Melvin Unified Brain...
echo.
echo 🧠 UPGRADED FEATURES ACTIVE:
echo ============================
echo ✅ Background autonomous thinking (every 30s)
echo ✅ Ollama integration (localhost:11434)
echo ✅ Force-driven responses (continuous 0.0-1.0)
echo ✅ Contradiction detection and regeneration
echo ✅ BinaryNode storage for ALL I/O
echo ✅ Instinct-driven reasoning
echo ✅ Transparent reasoning paths
echo.
echo 📋 COMMANDS:
echo - Type anything to process and store in memory
echo - Type 'background' to see autonomous thinking activity
echo - Type 'ollama' to see Ollama integration status
echo - Type 'forces' to see force-driven response system
echo - Type 'status' to see brain statistics
echo - Type 'memory' to see memory statistics
echo - Type 'instincts' to see instinct weights
echo - Type 'help' for full command list
echo - Type 'quit' to exit
echo.

melvin_upgraded.exe

echo.
echo Upgraded Melvin Unified Brain session ended.
pause
