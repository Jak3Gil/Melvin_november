#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>
#include "melvin_optimized_v2.h"

int main() {
    std::cout << "🧠 MELVIN C++ INTERACTIVE SYSTEM" << std::endl;
    std::cout << "=================================" << std::endl;
    std::cout << "Welcome! I'm Melvin, your unified brain AI companion." << std::endl;
    std::cout << "I have integrated systems for:" << std::endl;
    std::cout << "- Curiosity Gap Detection" << std::endl;
    std::cout << "- Dynamic Tools System" << std::endl;
    std::cout << "- Meta-Tool Engineer" << std::endl;
    std::cout << "- Curiosity Execution Loop" << std::endl;
    std::cout << "- Temporal Planning & Sequencing" << std::endl;
    std::cout << "- Enhanced Web Search (Pure C++)" << std::endl;
    std::cout << "\nType 'quit' to exit, 'status' for system info, 'help' for commands." << std::endl;
    std::cout << "=================================" << std::endl;

    try {
        MelvinOptimizedV2 melvin;
        std::cout << "✅ Melvin system initialized successfully" << std::endl;
        std::cout << "\n";

        std::string user_input;
        int conversation_turn = 0;

        while (true) {
            std::cout << "\nYou: ";
            std::getline(std::cin, user_input);

            if (user_input.empty()) {
                continue;
            }

            conversation_turn++;

            std::string lower_input = user_input;
            std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);

            if (lower_input == "quit" || lower_input == "exit") {
                std::cout << "\nMelvin: Thank you for this wonderful conversation! ";
                std::cout << "I've processed " << conversation_turn << " turns through my unified brain system. ";
                std::cout << "I'm grateful for the experience. Until we meet again! 🧠✨" << std::endl;
                break;
            } else if (lower_input == "status") {
                std::cout << "\n📊 MELVIN SYSTEM STATUS" << std::endl;
                std::cout << "======================" << std::endl;
                std::cout << "Conversation turns: " << conversation_turn << std::endl;
                std::cout << "System: Unified Brain with Enhanced Web Search" << std::endl;
                std::cout << "Implementation: Pure C++ (No Python dependencies)" << std::endl;
                std::cout << "Status: Active and ready to help" << std::endl;
                continue;
            } else if (lower_input == "help") {
                std::cout << "\nMelvin: Here are some things you can try:" << std::endl;
                std::cout << "- Ask me about cancer, quantum computing, AI, or science" << std::endl;
                std::cout << "- Request calculations or computations" << std::endl;
                std::cout << "- Ask me to search for information" << std::endl;
                std::cout << "- Have philosophical discussions" << std::endl;
                std::cout << "- Ask about my systems and capabilities" << std::endl;
                std::cout << "- Type 'status' to see my current state" << std::endl;
                continue;
            }

            // Process input through unified brain system
            std::cout << "\nMelvin: ";
            auto result = melvin.process_cognitive_input(user_input);
            
            // Show the main response
            std::cout << result.final_response << std::endl;
            
            // Add a small delay to simulate thinking
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ Error in interactive session: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
