@echo off
echo Testing Melvin with input...
echo what is cancer > input.txt
echo quit >> input.txt
melvin_debug.exe < input.txt
del input.txt
