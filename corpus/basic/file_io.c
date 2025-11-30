/* Basic file I/O - read and write files */
#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *f = fopen("test.txt", "w");
    if (f) {
        fprintf(f, "Hello from file!\n");
        fclose(f);
    }
    
    f = fopen("test.txt", "r");
    if (f) {
        char buf[256];
        while (fgets(buf, sizeof(buf), f)) {
            printf("%s", buf);
        }
        fclose(f);
    }
    return 0;
}

