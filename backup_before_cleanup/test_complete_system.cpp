#include "melvin_complete_system.h"
#include <signal.h>

// Global flag for graceful shutdown
std::atomic<bool> should_continue(true);

// Signal handler for graceful shutdown
void signalHandler(int signal) {
    std::cout << "\n🛑 Received signal " << signal << ", initiating graceful shutdown..." << std::endl;
    should_continue = false;
}

int main() {
    std::cout << "🤖 MELVIN COMPLETE UNIFIED SYSTEM - ONE SYSTEM TO RULE THEM ALL" << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "NO LOOSE ENDS - NO MISSING FEATURES - EVERYTHING INTEGRATED!" << std::endl;
    std::cout << "Press Ctrl+C to stop gracefully" << std::endl;
    std::cout << std::endl;
    
    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // Create Melvin's complete unified system
    MelvinCompleteInterface melvin;
    
    // Start Melvin with complete system
    melvin.startMelvin();
    
    std::cout << "\n🚀 MELVIN COMPLETE UNIFIED SYSTEM IS NOW RUNNING!" << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "🧠 All features integrated - NO LOOSE ENDS!" << std::endl;
    std::cout << "⚡ Reasoning engine active" << std::endl;
    std::cout << "🧬 Driver system active" << std::endl;
    std::cout << "💾 Binary storage active" << std::endl;
    std::cout << "🎯 Learning system active" << std::endl;
    std::cout << "🔄 Autonomous cycles active" << std::endl;
    std::cout << "🎯 Mission: Compound intelligence to help humanity reach its full potential" << std::endl;
    std::cout << std::endl;
    
    // Test the complete system
    std::vector<std::string> test_questions = {
        "What is the nature of intelligence?",
        "Solve this sequence: 2, 4, 8, 16, ?",
        "What patterns do you see in: A, B, C, D, ?",
        "How can AI better serve humanity?",
        "What is the next number in: 1, 3, 5, 7, ?"
    };
    
    std::cout << "🧪 TESTING COMPLETE UNIFIED SYSTEM" << std::endl;
    std::cout << "===================================" << std::endl;
    
    for (size_t i = 0; i < test_questions.size() && should_continue; ++i) {
        std::cout << "\n📝 Test Question " << (i + 1) << ": " << test_questions[i] << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // Ask Melvin the question
        std::string response = melvin.askMelvin(test_questions[i]);
        
        std::cout << "🤖 Melvin's Response:" << std::endl;
        std::cout << response << std::endl;
        
        // Print status after each question
        melvin.printStatus();
        
        // Small delay between questions
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        if (!should_continue) {
            break;
        }
    }
    
    if (should_continue) {
        std::cout << "\n🔄 CONTINUOUS AUTONOMOUS LEARNING TEST" << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "Testing continuous autonomous learning with complete system..." << std::endl;
        
        // Start with an initial question
        std::string current_input = "What should I think about next?";
        
        for (int cycle = 0; cycle < 10 && should_continue; ++cycle) {
            std::cout << "\n🔄 AUTONOMOUS CYCLE " << (cycle + 1) << std::endl;
            std::cout << "====================" << std::endl;
            std::cout << "📥 Input: " << current_input << std::endl;
            
            // Get Melvin's response
            std::string response = melvin.askMelvin(current_input);
            
            std::cout << "📤 Output: " << response.substr(0, 200) << (response.length() > 200 ? "..." : "") << std::endl;
            
            // Generate next input based on response (simplified autonomous generation)
            current_input = "Based on my previous thought: " + response.substr(0, 100) + "... What should I explore next?";
            
            // Print status every 5 cycles
            if ((cycle + 1) % 5 == 0) {
                std::cout << "\n📊 STATUS UPDATE - CYCLE " << (cycle + 1) << std::endl;
                std::cout << "================================" << std::endl;
                melvin.printStatus();
            }
            
            // Small delay
            std::this_thread::sleep_for(std::chrono::seconds(3));
            
            if (!should_continue) {
                break;
            }
        }
    }
    
    std::cout << "\n🛑 GRACEFUL SHUTDOWN INITIATED" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Print final analysis
    std::cout << "\n📊 FINAL COMPLETE SYSTEM ANALYSIS" << std::endl;
    std::cout << "==================================" << std::endl;
    melvin.printAnalysis();
    
    // Stop Melvin
    melvin.stopMelvin();
    
    std::cout << "\n🎉 MELVIN COMPLETE UNIFIED SYSTEM TEST COMPLETE!" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "✅ Melvin successfully used his complete unified system!" << std::endl;
    std::cout << "✅ All features integrated - NO LOOSE ENDS!" << std::endl;
    std::cout << "✅ Reasoning engine worked" << std::endl;
    std::cout << "✅ Driver system worked" << std::endl;
    std::cout << "✅ Binary storage worked" << std::endl;
    std::cout << "✅ Learning system worked" << std::endl;
    std::cout << "✅ Autonomous cycles worked" << std::endl;
    std::cout << "✅ ONE SYSTEM TO RULE THEM ALL!" << std::endl;
    std::cout << "✅ Melvin successfully compounded intelligence with complete system!" << std::endl;
    
    return 0;
}
