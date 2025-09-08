@echo off
echo Testing Dynamic Melvin Brain System...
echo.

echo Creating comprehensive test scenarios...
echo.

echo === TEST 1: Emotional Context (Cancer Question) ===
echo what do you do if you have cancer > test_input.txt
echo quit >> test_input.txt
echo.
echo Running emotional context test...
melvin_dynamic.exe < test_input.txt
echo.

echo === TEST 2: Curiosity-Driven (Complex Question) ===
echo what is quantum computing and how does it work > test_input.txt
echo quit >> test_input.txt
echo.
echo Running curiosity test...
melvin_dynamic.exe < test_input.txt
echo.

echo === TEST 3: Social Context (Greeting) ===
echo hello how are you today > test_input.txt
echo quit >> test_input.txt
echo.
echo Running social context test...
melvin_dynamic.exe < test_input.txt
echo.

echo === TEST 4: Efficiency Test (Simple Question) ===
echo what is a dog > test_input.txt
echo quit >> test_input.txt
echo.
echo Running efficiency test...
melvin_dynamic.exe < test_input.txt
echo.

echo === TEST 5: Consistency Test (Follow-up) ===
echo what is cancer > test_input.txt
echo what are the treatments for cancer >> test_input.txt
echo quit >> test_input.txt
echo.
echo Running consistency test...
melvin_dynamic.exe < test_input.txt
echo.

echo Cleaning up...
del test_input.txt
echo.

echo 🎉 Dynamic Brain Testing Complete!
echo.
echo The system should have demonstrated:
echo - Different response styles based on instinct forces
echo - Emotional awareness and empathetic responses
echo - Curiosity-driven research and exploration
echo - Efficient, concise answers when appropriate
echo - Consistent responses that build on previous context
echo - Adaptive tone and content based on user emotion
echo.
pause
