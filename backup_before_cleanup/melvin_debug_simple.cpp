#include "melvin_unified_brain.h"
#include <iostream>
#include <string>

int main() {
    std::cout << "🧠 Starting Melvin Debug Test" << std::endl;
    
    try {
        // Initialize brain
        MelvinUnifiedBrain brain("melvin_debug_memory");
        
        std::cout << "✅ Brain initialized successfully" << std::endl;
        
        // Test simple input
        std::string test_input = "Hello, how are you?";
        std::cout << "📝 Testing input: " << test_input << std::endl;
        
        std::string response = brain.process_input(test_input);
        
        std::cout << "✅ Response generated: " << response << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "🎉 Debug test completed successfully!" << std::endl;
    return 0;
}
