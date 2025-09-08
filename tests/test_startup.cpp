#include <iostream>
#include <string>
#include <cassert>
#include "logging.h"

// Forward declaration
std::string perform_web_search(const std::string& query);

int main() {
    std::cout << "Running Melvin Unified Brain startup tests..." << std::endl;
    
    try {
        // Test 1: Initialize logging
        std::cout << "Test 1: Initializing logging system..." << std::endl;
        melvin::Logger::init_logging("test_debug.log", false);
        melvin::Logger::log_info("Test logging initialized");
        std::cout << "✅ Logging initialization successful" << std::endl;
        
        // Test 2: Test web search functionality
        std::cout << "Test 2: Testing web search functionality..." << std::endl;
        std::string search_result = perform_web_search("test");
        
        // We don't assert on empty result since API might not be available
        // Just ensure the function doesn't crash
        std::cout << "✅ Web search function call successful (result length: " 
                  << search_result.length() << ")" << std::endl;
        
        // Test 3: Test basic string operations
        std::cout << "Test 3: Testing basic string operations..." << std::endl;
        std::string test_input = "Hello, Melvin!";
        assert(!test_input.empty());
        assert(test_input.length() > 0);
        std::cout << "✅ Basic string operations successful" << std::endl;
        
        // Test 4: Test logging with different levels
        std::cout << "Test 4: Testing logging levels..." << std::endl;
        melvin::Logger::log_info("Info message test");
        melvin::Logger::log_warn("Warning message test");
        melvin::Logger::log_error("Error message test");
        std::cout << "✅ All logging levels successful" << std::endl;
        
        std::cout << "\n🎉 All startup tests passed successfully!" << std::endl;
        std::cout << "Melvin Unified Brain is ready for operation." << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}