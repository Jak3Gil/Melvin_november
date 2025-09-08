#!/bin/bash

# ============================================================================
# MELVIN BRAIN ARCHITECTURE VALIDATION BUILD AND TEST SCRIPT
# ============================================================================

set -e  # Exit on any error

echo "🧪 MELVIN BRAIN ARCHITECTURE VALIDATION"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! command -v g++ &> /dev/null; then
        print_error "g++ compiler not found. Please install g++."
        exit 1
    fi
    
    if ! command -v pkg-config &> /dev/null; then
        print_warning "pkg-config not found. Some libraries may not be detected properly."
    fi
    
    # Check for compression libraries
    if ! pkg-config --exists zlib; then
        print_warning "zlib not found. Compression features may not work properly."
    fi
    
    if ! pkg-config --exists liblzma; then
        print_warning "liblzma not found. LZMA compression may not work."
    fi
    
    if ! pkg-config --exists libzstd; then
        print_warning "libzstd not found. ZSTD compression may not work."
    fi
    
    print_success "Dependencies check completed"
}

# Clean previous builds
clean_build() {
    print_status "Cleaning previous builds..."
    
    rm -f melvin_brain_tests
    rm -f melvin_logic_solver_test
    rm -f melvin_memory_tests
    rm -f run_melvin_tests
    rm -f melvin_optimized_v2_test
    rm -f *.o
    rm -rf melvin_test_memory
    rm -rf melvin_logic_memory
    rm -rf melvin_memory_test
    rm -rf melvin_comprehensive_test
    
    print_success "Clean completed"
}

# Build individual test programs
build_tests() {
    print_status "Building test programs..."
    
    # Compiler flags
    CXXFLAGS="-std=c++17 -O3 -Wall -Wextra -march=native -ffast-math"
    
    # Library flags - handle missing libraries gracefully
    LIBFLAGS=""
    COMPRESSION_DEFINES=""
    
    if pkg-config --exists zlib; then
        LIBFLAGS="$LIBFLAGS $(pkg-config --cflags --libs zlib)"
        COMPRESSION_DEFINES="$COMPRESSION_DEFINES -DHAVE_ZLIB"
        print_status "zlib compression library found"
    else
        print_warning "zlib not found - compression will be disabled"
    fi
    
    if pkg-config --exists liblzma; then
        LIBFLAGS="$LIBFLAGS $(pkg-config --cflags --libs liblzma)"
        COMPRESSION_DEFINES="$COMPRESSION_DEFINES -DHAVE_LZMA"
        print_status "lzma compression library found"
    else
        print_warning "liblzma not found - LZMA compression will be disabled"
    fi
    
    if pkg-config --exists libzstd; then
        LIBFLAGS="$LIBFLAGS $(pkg-config --cflags --libs libzstd)"
        COMPRESSION_DEFINES="$COMPRESSION_DEFINES -DHAVE_ZSTD"
        print_status "zstd compression library found"
    else
        print_warning "libzstd not found - ZSTD compression will be disabled"
    fi
    
    # Build comprehensive test runner
    print_status "Building comprehensive test runner..."
    g++ $CXXFLAGS $COMPRESSION_DEFINES -o run_melvin_tests run_melvin_tests.cpp melvin_optimized_v2.cpp $LIBFLAGS
    if [ $? -eq 0 ]; then
        print_success "Comprehensive test runner built successfully"
    else
        print_error "Failed to build comprehensive test runner"
        exit 1
    fi
    
    # Build brain architecture tests
    print_status "Building brain architecture tests..."
    g++ $CXXFLAGS $COMPRESSION_DEFINES -o melvin_brain_tests melvin_brain_tests.cpp melvin_optimized_v2.cpp $LIBFLAGS
    if [ $? -eq 0 ]; then
        print_success "Brain architecture tests built successfully"
    else
        print_error "Failed to build brain architecture tests"
        exit 1
    fi
    
    # Build logic solver tests
    print_status "Building logic solver tests..."
    g++ $CXXFLAGS $COMPRESSION_DEFINES -o melvin_logic_solver_test melvin_logic_solver_test.cpp melvin_optimized_v2.cpp $LIBFLAGS
    if [ $? -eq 0 ]; then
        print_success "Logic solver tests built successfully"
    else
        print_error "Failed to build logic solver tests"
        exit 1
    fi
    
    # Build memory tests
    print_status "Building memory tests..."
    g++ $CXXFLAGS $COMPRESSION_DEFINES -o melvin_memory_tests melvin_memory_tests.cpp melvin_optimized_v2.cpp $LIBFLAGS
    if [ $? -eq 0 ]; then
        print_success "Memory tests built successfully"
    else
        print_error "Failed to build memory tests"
        exit 1
    fi
    
    # Build main Melvin system test
    print_status "Building main Melvin system test..."
    g++ $CXXFLAGS $COMPRESSION_DEFINES -o melvin_optimized_v2_test melvin_optimized_v2.cpp $LIBFLAGS
    if [ $? -eq 0 ]; then
        print_success "Main Melvin system test built successfully"
    else
        print_error "Failed to build main Melvin system test"
        exit 1
    fi
    
    print_success "All test programs built successfully"
}

# Run individual test suites
run_tests() {
    print_status "Running comprehensive Melvin brain validation tests..."
    echo ""
    
    # Run comprehensive test suite
    print_status "=== RUNNING COMPREHENSIVE TEST SUITE ==="
    ./run_melvin_tests
    COMPREHENSIVE_EXIT_CODE=$?
    echo ""
    
    # Run brain architecture tests
    print_status "=== RUNNING BRAIN ARCHITECTURE TESTS ==="
    ./melvin_brain_tests
    BRAIN_EXIT_CODE=$?
    echo ""
    
    # Run logic solver tests
    print_status "=== RUNNING LOGIC SOLVER TESTS ==="
    ./melvin_logic_solver_test
    LOGIC_EXIT_CODE=$?
    echo ""
    
    # Run memory tests
    print_status "=== RUNNING MEMORY TESTS ==="
    ./melvin_memory_tests
    MEMORY_EXIT_CODE=$?
    echo ""
    
    # Run main system test
    print_status "=== RUNNING MAIN SYSTEM TEST ==="
    ./melvin_optimized_v2_test
    MAIN_EXIT_CODE=$?
    echo ""
    
    # Summary of test results
    print_status "=== TEST RESULTS SUMMARY ==="
    echo ""
    
    if [ $COMPREHENSIVE_EXIT_CODE -eq 0 ]; then
        print_success "Comprehensive Test Suite: PASSED"
    else
        print_error "Comprehensive Test Suite: FAILED"
    fi
    
    if [ $BRAIN_EXIT_CODE -eq 0 ]; then
        print_success "Brain Architecture Tests: PASSED"
    else
        print_error "Brain Architecture Tests: FAILED"
    fi
    
    if [ $LOGIC_EXIT_CODE -eq 0 ]; then
        print_success "Logic Solver Tests: PASSED"
    else
        print_error "Logic Solver Tests: FAILED"
    fi
    
    if [ $MEMORY_EXIT_CODE -eq 0 ]; then
        print_success "Memory Tests: PASSED"
    else
        print_error "Memory Tests: FAILED"
    fi
    
    if [ $MAIN_EXIT_CODE -eq 0 ]; then
        print_success "Main System Test: PASSED"
    else
        print_error "Main System Test: FAILED"
    fi
    
    # Overall result
    TOTAL_FAILED=$((COMPREHENSIVE_EXIT_CODE + BRAIN_EXIT_CODE + LOGIC_EXIT_CODE + MEMORY_EXIT_CODE + MAIN_EXIT_CODE))
    
    echo ""
    if [ $TOTAL_FAILED -eq 0 ]; then
        print_success "🎉 ALL TESTS PASSED! Melvin's brain architecture is working correctly."
        echo ""
        print_success "Melvin is successfully using his own brain to:"
        print_success "  ✅ Form memories and neural connections"
        print_success "  ✅ Learn through Hebbian mechanisms"
        print_success "  ✅ Solve logic puzzles using reasoning"
        print_success "  ✅ Distinguish between different problems (not just pattern matching)"
        print_success "  ✅ Maintain consistent brain state"
        print_success "  ✅ Achieve high performance and efficiency"
    else
        print_error "❌ SOME TESTS FAILED. Melvin's brain architecture needs investigation."
        echo ""
        print_warning "Please review the test output above to identify issues."
        print_warning "Common issues:"
        print_warning "  - Missing compression libraries (zlib, lzma, zstd)"
        print_warning "  - Memory formation problems"
        print_warning "  - Hebbian learning not working"
        print_warning "  - Performance below requirements"
    fi
    
    return $TOTAL_FAILED
}

# Generate test reports summary
generate_reports() {
    print_status "Generating test reports summary..."
    
    echo ""
    print_status "=== TEST REPORTS GENERATED ==="
    
    if [ -f "melvin_comprehensive_test_report.txt" ]; then
        print_success "📄 Comprehensive Test Report: melvin_comprehensive_test_report.txt"
    fi
    
    if [ -f "melvin_brain_test_report.txt" ]; then
        print_success "📄 Brain Architecture Report: melvin_brain_test_report.txt"
    fi
    
    if [ -f "melvin_logic_validation_report.txt" ]; then
        print_success "📄 Logic Solver Report: melvin_logic_validation_report.txt"
    fi
    
    if [ -f "melvin_memory_validation_report.txt" ]; then
        print_success "📄 Memory Validation Report: melvin_memory_validation_report.txt"
    fi
    
    echo ""
    print_status "Review these reports for detailed test results and brain state information."
}

# Main execution
main() {
    echo "Starting Melvin Brain Architecture Validation..."
    echo ""
    
    # Parse command line arguments
    CLEAN_ONLY=false
    BUILD_ONLY=false
    TEST_ONLY=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --clean-only)
                CLEAN_ONLY=true
                shift
                ;;
            --build-only)
                BUILD_ONLY=true
                shift
                ;;
            --test-only)
                TEST_ONLY=true
                shift
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --clean-only    Only clean previous builds"
                echo "  --build-only    Only build test programs"
                echo "  --test-only     Only run tests (assumes programs are built)"
                echo "  --help          Show this help message"
                echo ""
                echo "Default: Clean, build, and run all tests"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Execute based on options
    if [ "$CLEAN_ONLY" = true ]; then
        clean_build
        exit 0
    fi
    
    if [ "$BUILD_ONLY" = true ]; then
        check_dependencies
        clean_build
        build_tests
        exit 0
    fi
    
    if [ "$TEST_ONLY" = true ]; then
        run_tests
        generate_reports
        exit $?
    fi
    
    # Default: full build and test
    check_dependencies
    clean_build
    build_tests
    run_tests
    generate_reports
    
    echo ""
    print_status "Melvin Brain Architecture Validation completed!"
}

# Run main function with all arguments
main "$@"
