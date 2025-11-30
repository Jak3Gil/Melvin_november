/*
 * test_exec_basic.c
 * 
 * Standalone test to verify RWX (Read-Write-Execute) memory works.
 * 
 * This test:
 * 1. Allocates RWX anonymous memory via mmap
 * 2. Writes architecture-correct machine code into it
 * 3. Calls it as a function pointer
 * 4. Verifies it returns 0x42
 * 
 * No Melvin graph, no events, no EXEC nodes - just pure OS + CPU.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

// Detect architecture at compile time
#if defined(__x86_64__) || defined(_M_X64)
#define MELVIN_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define MELVIN_ARCH_AARCH64 1
#else
#error "test_exec_basic.c only supports x86_64 or aarch64 for now"
#endif

// Architecture-specific stub code
#if MELVIN_ARCH_X86_64
// x86-64: mov $0x42, %rax; ret
// mov $0x42, %rax = 48 C7 C0 42 00 00 00
// ret = C3
static const uint8_t stub_code[] = {
    0x48, 0xC7, 0xC0, 0x42, 0x00, 0x00, 0x00,  // mov $0x42, %rax
    0xC3                                        // ret
};
#elif MELVIN_ARCH_AARCH64
// ARM64: mov x0, #0x42; ret
// Verified with: as test.s && objdump -d
// mov x0, #0x42 = d2800840 (little-endian: 40 08 80 d2)
// ret = d65f03c0 (little-endian: c0 03 5f d6)
static const uint8_t stub_code[] = {
    0x40, 0x08, 0x80, 0xD2,   // mov x0, #0x42
    0xC0, 0x03, 0x5F, 0xD6    // ret
};
#endif

int main(void) {
    size_t code_size = sizeof(stub_code);
    
    printf("test_exec_basic: Starting RWX memory test\n");
    printf("  Architecture: ");
#if MELVIN_ARCH_X86_64
    printf("x86_64\n");
#elif MELVIN_ARCH_AARCH64
    printf("aarch64\n");
#endif
    printf("  Stub code size: %zu bytes\n", code_size);
    printf("  Stub bytes: ");
    for (size_t i = 0; i < code_size; i++) {
        printf("%02X ", stub_code[i]);
    }
    printf("\n");
    
    // Allocate RWX anonymous memory
    void *mem = mmap(NULL, code_size,
                     PROT_READ | PROT_WRITE | PROT_EXEC,
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     -1, 0);
    
    if (mem == MAP_FAILED) {
        perror("test_exec_basic: mmap failed");
        return 1;
    }
    
    printf("  Allocated RWX memory at: %p\n", mem);
    
    // Copy stub code into the RWX region
    memcpy(mem, stub_code, code_size);
    printf("  Copied stub code to RWX region\n");
    
    // Cast to function pointer and call
    typedef uint64_t (*stub_fn_t)(void);
    stub_fn_t fn = (stub_fn_t)mem;
    
    printf("  Calling stub function...\n");
    uint64_t result = fn();
    
    // Print diagnostics
    printf("test_exec_basic: stub returned 0x%llx\n",
           (unsigned long long)result);
    
    // Cleanup
    munmap(mem, code_size);
    
    // Check result
    if (result == 0x42ull) {
        printf("test_exec_basic: PASS\n");
        return 0;
    } else {
        printf("test_exec_basic: FAIL (expected 0x42, got 0x%llx)\n",
               (unsigned long long)result);
        return 2;
    }
}

