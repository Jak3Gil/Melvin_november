#include <iostream>
#include <string>

int main() {
    std::cout << "🔍 MELVIN DIAGNOSTIC TEST" << std::endl;
    std::cout << "=========================" << std::endl;
    
    try {
        std::cout << "✅ Program started successfully" << std::endl;
        
        // Test basic functionality
        std::string test_input;
        std::cout << "\nEnter a test question: ";
        std::getline(std::cin, test_input);
        
        std::cout << "You entered: " << test_input << std::endl;
        
        // Simple response logic
        if (test_input.find("name") != std::string::npos) {
            std::cout << "RESPONSE: I don't have access to your personal information." << std::endl;
        } else if (test_input.find("cancer") != std::string::npos) {
            std::cout << "RESPONSE: Cancer is a group of diseases with uncontrolled cell growth." << std::endl;
        } else {
            std::cout << "RESPONSE: That's an interesting question!" << std::endl;
        }
        
        std::cout << "\n✅ Test completed successfully" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
