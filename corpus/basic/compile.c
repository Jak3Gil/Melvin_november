/* Compile example - how to compile C code */
/* Command: gcc -c -o output.o input.c */

#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(5, 3);
    printf("Result: %d\n", result);
    return 0;
}

