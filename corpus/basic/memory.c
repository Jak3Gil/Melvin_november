/* Memory management - malloc and free */
#include <stdio.h>
#include <stdlib.h>

int main() {
    /* Allocate memory */
    int *arr = malloc(10 * sizeof(int));
    if (arr) {
        for (int i = 0; i < 10; i++) {
            arr[i] = i * 2;
        }
        
        /* Use memory */
        for (int i = 0; i < 10; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
        
        /* Free memory */
        free(arr);
    }
    return 0;
}

