#include "melvin_unified_brain.h"
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

class SimpleMelvinTest {
private:
    std::unique_ptr<MelvinUnifiedBrain> brain;
    
public:
    SimpleMelvinTest() {
        std::cout << "🧠 Initializing Simple Melvin Test (No Background)" << std::endl;
        
        // Initialize brain WITHOUT background scheduler
        brain = std::make_unique<MelvinUnifiedBrain>("melvin_test_memory");
        
        std::cout << "✅ Brain initialized successfully" << std::endl;
    }
    
    void run_test() {
        std::cout << "\n🧠 SIMPLE MELVIN TEST" << std::endl;
        std::cout << "====================" << std::endl;
        std::cout << "Type your questions (or 'quit' to exit):" << std::endl;
        
        std::string user_input;
        
        while (true) {
            std::cout << "\nYou: ";
            std::getline(std::cin, user_input);
            
            if (user_input.empty()) {
                continue;
            }
            
            if (user_input == "quit" || user_input == "exit") {
                break;
            }
            
            std::cout << "\nMelvin: ";
            
            try {
                std::string response = brain->process_input(user_input);
                std::cout << response << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "❌ Error processing input: " << e.what() << std::endl;
            }
            
            // Small delay
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        
        std::cout << "\n👋 Test completed!" << std::endl;
    }
};

int main() {
    try {
        SimpleMelvinTest melvin;
        melvin.run_test();
    } catch (const std::exception& e) {
        std::cerr << "❌ Fatal error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
