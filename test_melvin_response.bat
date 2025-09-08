@echo off
echo Testing Melvin's response generation...
echo.
echo Creating test input with multiple questions...
echo what is cancer > test_input.txt
echo what is a dog >> test_input.txt
echo quit >> test_input.txt
echo.
echo Running Melvin with test input...
echo ========================================
melvin_debug.exe < test_input.txt
echo ========================================
echo.
echo Test completed!
echo.
echo Cleaning up...
del test_input.txt
echo.
echo If you saw Melvin's responses above, the system is working!
pause
