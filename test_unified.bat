@echo off
echo Testing Melvin Unified Brain System...
echo.

echo Creating comprehensive test input...
echo what is cancer > test_input.txt
echo what is a dog >> test_input.txt
echo who are you >> test_input.txt
echo what can you do >> test_input.txt
echo status >> test_input.txt
echo quit >> test_input.txt
echo.

echo Running Melvin Unified with test input...
echo ========================================
melvin_unified.exe < test_input.txt
echo ========================================
echo.

echo Test completed!
echo.

echo Cleaning up...
del test_input.txt
echo.

echo If you saw Melvin's responses above, the unified system is working!
echo The system includes:
echo - Binary node memory with global persistence
echo - Hebbian learning and connections
echo - Instinct-driven reasoning
echo - Web search integration
echo - Natural response generation
echo - Debug output for troubleshooting
echo.
pause
