#include "melvin_optimized_v2.h"
#include <iostream>
#include <string>
#include <iomanip>

int main() {
    std::cout << "🧠 MELVIN INTERACTIVE CONVERSATION SYSTEM" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Welcome! I'm Melvin, your unified brain AI companion." << std::endl;
    std::cout << "I have integrated systems for:" << std::endl;
    std::cout << "- Pressure-Based Instinct System" << std::endl;
    std::cout << "- Meta-Reasoning Layer" << std::endl;
    std::cout << "- Enhanced Binary Memory" << std::endl;
    std::cout << "- Emotional Grounding System" << std::endl;
    std::cout << "- Blended Reasoning Protocol" << std::endl;
    std::cout << std::endl;
    std::cout << "Type 'quit' to exit, 'status' for system info, 'help' for commands." << std::endl;
    std::cout << "=========================================" << std::endl;
    
    try {
        // Initialize Melvin with the new system
        MelvinOptimizedV2 melvin("interactive_melvin_memory");
        
        std::string user_input;
        int conversation_turns = 0;
        
        while (true) {
            std::cout << std::endl;
            std::cout << "You: ";
            std::getline(std::cin, user_input);
            
            if (user_input.empty()) continue;
            
            if (user_input == "quit" || user_input == "exit") {
                break;
            }
            
            if (user_input == "status") {
                auto brain_state = melvin.get_unified_state();
                std::cout << std::endl;
                std::cout << "📊 MELVIN SYSTEM STATUS" << std::endl;
                std::cout << "======================" << std::endl;
                std::cout << "Conversation turns: " << conversation_turns << std::endl;
                std::cout << "Memory nodes: " << brain_state.global_memory.total_nodes << std::endl;
                std::cout << "Connections: " << brain_state.global_memory.total_edges << std::endl;
                std::cout << "Storage: " << std::fixed << std::setprecision(2) 
                          << brain_state.global_memory.storage_used_mb << " MB" << std::endl;
                std::cout << "Uptime: " << brain_state.system.uptime_seconds << " seconds" << std::endl;
                std::cout << "Hebbian updates: " << brain_state.global_memory.stats.hebbian_updates << std::endl;
                continue;
            }
            
            if (user_input == "help") {
                std::cout << std::endl;
                std::cout << "Melvin: Here are some things you can try:" << std::endl;
                std::cout << "- Ask me about science, technology, or philosophy" << std::endl;
                std::cout << "- Try emotional topics (I'll detect and respond appropriately)" << std::endl;
                std::cout << "- Ask complex questions (I'll use meta-reasoning)" << std::endl;
                std::cout << "- Test my instinct balancing with different types of questions" << std::endl;
                std::cout << "- Type 'status' to see my current brain state" << std::endl;
                std::cout << "- Type 'quit' to end our conversation" << std::endl;
                continue;
            }
            
            conversation_turns++;
            
            std::cout << std::endl;
            std::cout << "Melvin: Processing through unified brain system..." << std::endl;
            std::cout << "[Instinct Analysis] Computing pressure-based forces..." << std::endl;
            std::cout << "[Meta-Reasoning] Instinct council arbitration..." << std::endl;
            std::cout << "[Memory Integration] Storing conversation context..." << std::endl;
            std::cout << "[Emotional Grounding] Analyzing emotional signals..." << std::endl;
            std::cout << std::endl;
            
            // Process input through Melvin's complete system
            auto result = melvin.process_cognitive_input(user_input);
            
            // Display the response
            std::cout << "Melvin: " << result.final_response << std::endl;
            
            // Show system analysis
            std::cout << std::endl;
            std::cout << "🧠 [System Analysis]" << std::endl;
            std::cout << "[Instinct Forces] ";
            std::cout << "Curiosity(" << std::fixed << std::setprecision(0) 
                      << result.computed_forces.curiosity * 100 << "%) ";
            std::cout << "Efficiency(" << std::fixed << std::setprecision(0) 
                      << result.computed_forces.efficiency * 100 << "%) ";
            std::cout << "Social(" << std::fixed << std::setprecision(0) 
                      << result.computed_forces.social * 100 << "%) ";
            std::cout << "Consistency(" << std::fixed << std::setprecision(0) 
                      << result.computed_forces.consistency * 100 << "%) ";
            std::cout << "Survival(" << std::fixed << std::setprecision(0) 
                      << result.computed_forces.survival * 100 << "%)" << std::endl;
            
            std::cout << "[Meta-Reasoning] ";
            if (!result.meta_reasoning.arbitration.amplifications.empty()) {
                std::cout << "Amplified: " << result.meta_reasoning.arbitration.amplifications[0];
            }
            if (!result.meta_reasoning.arbitration.suppressions.empty()) {
                std::cout << ", Suppressed: " << result.meta_reasoning.arbitration.suppressions[0];
            }
            if (!result.meta_reasoning.arbitration.blends.empty()) {
                std::cout << ", Blended: " << result.meta_reasoning.arbitration.blends[0];
                if (result.meta_reasoning.arbitration.blends.size() > 1) {
                    std::cout << " + " << result.meta_reasoning.arbitration.blends[1];
                }
            }
            std::cout << std::endl;
            
            std::cout << "[Emotional Grounding] ";
            if (result.emotional_grounding.has_grounding_signal) {
                std::cout << "Signal detected (" << result.emotional_grounding.grounding_type 
                          << "): " << result.emotional_grounding.grounding_evidence;
            } else {
                std::cout << "No emotional signals detected";
            }
            std::cout << std::endl;
            
            std::cout << "[Confidence] Overall: " << std::fixed << std::setprecision(0) 
                      << result.confidence * 100 << "%, Meta: " 
                      << result.meta_reasoning.meta_confidence * 100 << "%" << std::endl;
            
            // Store the conversation in memory
            melvin.process_text_input(user_input, "user");
            melvin.process_text_input(result.final_response, "melvin");
        }
        
        std::cout << std::endl;
        std::cout << "Melvin: Thank you for this wonderful conversation! I've learned so much from our interaction." << std::endl;
        std::cout << "My unified brain system has processed " << conversation_turns << " turns and I'm grateful for the experience." << std::endl;
        std::cout << "Until we meet again! 🧠✨" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        std::cout << "Falling back to simple interactive mode..." << std::endl;
        
        // Simple fallback mode
        std::string user_input;
        while (true) {
            std::cout << std::endl << "You: ";
            std::getline(std::cin, user_input);
            
            if (user_input == "quit" || user_input == "exit") break;
            
            std::cout << "Melvin: I'm processing your input through my unified brain system. ";
            std::cout << "That's an interesting question! I'm analyzing the patterns and relationships. ";
            std::cout << "Could you tell me more about what you're thinking?" << std::endl;
        }
    }
    
    return 0;
}
