/* Basic syscalls - system interaction */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

int main() {
    /* Write to stdout */
    write(1, "Output\n", 8);
    
    /* Read from stdin */
    char buf[256];
    read(0, buf, sizeof(buf));
    
    /* Execute command */
    system("echo 'Command executed'");
    
    return 0;
}

