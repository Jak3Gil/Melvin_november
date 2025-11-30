#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
int main() {
    DIR *dir = opendir("test_corpus");
    if (!dir) { printf("Cannot open\n"); return 1; }
    struct dirent *e;
    while ((e = readdir(dir))) {
        if (strcmp(e->d_name, ".") == 0 || strcmp(e->d_name, "..") == 0) continue;
        printf("Found: %s\n", e->d_name);
    }
    closedir(dir);
    return 0;
}
