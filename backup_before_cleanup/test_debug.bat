@echo off
echo Testing basic input handling...
echo.
echo what is cancer > input.txt
echo quit >> input.txt
echo.
echo Running debug test...
debug_test.exe < input.txt
echo.
echo Cleaning up...
del input.txt
echo Done!
