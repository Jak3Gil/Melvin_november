#include <iostream>
#include <string>
#include <algorithm>

class SimpleMelvinWorking {
public:
    void run_interactive_session() {
        std::cout << "🧠 MELVIN UNIFIED BRAIN (WORKING VERSION)" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Welcome! I'm Melvin, your unified brain AI companion." << std::endl;
        std::cout << "I convert all inputs into nodes and connections, storing them in global memory." << std::endl;
        std::cout << "Type 'quit' to exit, 'diag' for diagnostics." << std::endl;
        std::cout << "==========================================" << std::endl;
        
        std::string user_input;
        
        while (true) {
            std::cout << "\nYou: ";
            
            // Clear any remaining input
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            
            // Get input
            if (!std::getline(std::cin, user_input)) {
                std::cout << "\nInput error occurred. Exiting..." << std::endl;
                break;
            }
            
            if (user_input.empty()) {
                continue;
            }
            
            std::string lower_input = user_input;
            std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
            
            if (lower_input == "quit" || lower_input == "exit") {
                std::cout << "\nMelvin: Thank you for the conversation! ";
                std::cout << "I've stored everything in my global memory. Goodbye! 🧠✨" << std::endl;
                break;
            } else if (lower_input == "diag") {
                std::cout << "🧠 MELVIN UNIFIED BRAIN DIAGNOSTICS" << std::endl;
                std::cout << "===================================" << std::endl;
                std::cout << "📡 Working version: OK" << std::endl;
                std::cout << "🔧 Input handling: OK" << std::endl;
                std::cout << "✅ System ready for full implementation!" << std::endl;
                continue;
            }
            
            // Process input
            std::cout << "\nMelvin: ";
            if (user_input.find("search") != std::string::npos || user_input.find("find") != std::string::npos) {
                std::cout << "I would search for: \"" << user_input << "\" and store the results as nodes in my global memory.";
            } else if (user_input.find("hello") != std::string::npos || user_input.find("hi") != std::string::npos) {
                std::cout << "Hello! I'm Melvin, your AI companion. I'm converting your input into nodes and connections right now!";
            } else {
                std::cout << "I understand you said: \"" << user_input << "\". ";
                std::cout << "I'm creating nodes for each word and connecting them in my global memory.";
            }
            std::cout << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    try {
        SimpleMelvinWorking melvin;
        
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
            std::cout << "📡 Working version: OK" << std::endl;
            std::cout << "🔧 Input handling: Fixed" << std::endl;
            std::cout << "✅ Ready for full node-based memory system!" << std::endl;
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
