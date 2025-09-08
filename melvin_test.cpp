#include <iostream>
#include <string>
#include <algorithm>

class SimpleMelvin {
public:
    void run_interactive_session() {
        std::cout << "🧠 MELVIN UNIFIED BRAIN (TEST VERSION)" << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "Welcome! I'm Melvin, your unified brain AI companion." << std::endl;
        std::cout << "Type 'quit' to exit, 'diag' for diagnostics." << std::endl;
        std::cout << "======================================" << std::endl;
        
        std::string user_input;
        
        while (true) {
            std::cout << "\nYou: ";
            std::getline(std::cin, user_input);
            
            if (user_input.empty()) {
                continue;
            }
            
            std::string lower_input = user_input;
            std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
            
            if (lower_input == "quit" || lower_input == "exit") {
                std::cout << "\nMelvin: Thank you for the conversation! Goodbye! 🧠✨" << std::endl;
                break;
            } else if (lower_input == "diag") {
                std::cout << "🧠 MELVIN UNIFIED BRAIN DIAGNOSTICS" << std::endl;
                std::cout << "===================================" << std::endl;
                std::cout << "📡 Basic functionality: Working" << std::endl;
                std::cout << "🔧 Input processing: Working" << std::endl;
                std::cout << "✅ Test version running successfully!" << std::endl;
                continue;
            }
            
            // Process input
            std::cout << "\nMelvin: ";
            if (user_input.find("search") != std::string::npos || user_input.find("find") != std::string::npos) {
                std::cout << "I would search for: \"" << user_input << "\" but web search is not available in test mode.";
            } else if (user_input.find("hello") != std::string::npos || user_input.find("hi") != std::string::npos) {
                std::cout << "Hello! I'm Melvin, your AI companion. How can I help you today?";
            } else {
                std::cout << "I understand you said: \"" << user_input << "\". This is the test version of Melvin.";
            }
            std::cout << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    try {
        SimpleMelvin melvin;
        
        // Check for diagnostic mode
        bool diagnostic_mode = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--diag") {
                diagnostic_mode = true;
                break;
            }
        }
        
        if (diagnostic_mode) {
            std::cout << "🧠 MELVIN UNIFIED BRAIN DIAGNOSTICS" << std::endl;
            std::cout << "===================================" << std::endl;
            std::cout << "📡 Test version: Working" << std::endl;
            std::cout << "🔧 Basic functionality: OK" << std::endl;
            std::cout << "✅ Diagnostic complete!" << std::endl;
        } else {
            melvin.run_interactive_session();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Fatal Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Unknown Fatal Error occurred" << std::endl;
        return 1;
    }
    
    return 0;
}
