@echo off
echo 🧠 MELVIN NEURAL NETWORK SIZE CHECK
echo ===================================

echo Checking binary memory files...
echo.

if exist "melvin_binary_memory\nodes.bin" (
    echo 📊 nodes.bin:
    dir "melvin_binary_memory\nodes.bin" | findstr "nodes.bin"
    echo.
    echo Estimated nodes (assuming 50-200 bytes per node):
    for /f "tokens=3" %%a in ('dir "melvin_binary_memory\nodes.bin" ^| findstr "nodes.bin"') do (
        set /a nodes_50=%%a/50
        set /a nodes_100=%%a/100
        set /a nodes_200=%%a/200
        echo   - 50 bytes/node: ~!nodes_50! nodes
        echo   - 100 bytes/node: ~!nodes_100! nodes  
        echo   - 200 bytes/node: ~!nodes_200! nodes
    )
) else (
    echo ❌ nodes.bin not found
)

echo.

if exist "melvin_binary_memory\connections.bin" (
    echo 🔗 connections.bin:
    dir "melvin_binary_memory\connections.bin" | findstr "connections.bin"
    echo.
    echo Estimated connections (assuming 25-100 bytes per connection):
    for /f "tokens=3" %%a in ('dir "melvin_binary_memory\connections.bin" ^| findstr "connections.bin"') do (
        set /a conn_25=%%a/25
        set /a conn_50=%%a/50
        set /a conn_100=%%a/100
        echo   - 25 bytes/connection: ~!conn_25! connections
        echo   - 50 bytes/connection: ~!conn_50! connections
        echo   - 100 bytes/connection: ~!conn_100! connections
    )
) else (
    echo ❌ connections.bin not found
)

echo.
echo 🎯 SUMMARY:
echo Melvin has a substantial neural network stored in binary format.
echo The exact count depends on the internal data structure.
echo Run the unified system to see live node creation!

pause
