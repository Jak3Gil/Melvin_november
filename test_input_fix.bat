@echo off
echo Testing Melvin input fix...
echo.
echo Creating test input file...
echo what is cancer > test_input.txt
echo quit >> test_input.txt
echo.
echo Running Melvin with input redirection...
melvin_debug.exe < test_input.txt
echo.
echo Cleaning up...
del test_input.txt
echo Test complete!
pause
