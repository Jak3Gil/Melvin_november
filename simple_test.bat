@echo off
echo Testing Melvin with simple input...
echo.
echo what is cancer > input.txt
echo quit >> input.txt
echo.
echo Running Melvin...
melvin_debug.exe < input.txt
echo.
echo Cleaning up...
del input.txt
echo Done!
