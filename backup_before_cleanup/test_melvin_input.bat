@echo off
echo Testing Melvin input handling...
echo.
echo This will test if Melvin can receive input properly.
echo.
echo Creating test input...
echo what is cancer > input.txt
echo quit >> input.txt
echo.
echo Running Melvin with test input...
melvin_debug.exe < input.txt
echo.
echo Test completed!
echo.
echo Cleaning up...
del input.txt
echo.
echo If you saw Melvin's response above, the input fix worked!
pause
