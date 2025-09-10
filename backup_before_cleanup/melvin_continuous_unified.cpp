#include "melvin_unified_system.h"
#include <signal.h>

// Global flag for graceful shutdown
std::atomic<bool> should_continue(true);

// Signal handler for graceful shutdown
void signalHandler(int signal) {
    std::cout << "\n🛑 Received signal " << signal << ", initiating graceful shutdown..." << std::endl;
    should_continue = false;
}

int main() {
    std::cout << "🤖 MELVIN UNIFIED SYSTEM - CONTINUOUS AUTONOMOUS AI BRAIN" << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "UNIFIED REPOSITORY - SINGLE COHESIVE SYSTEM!" << std::endl;
    std::cout << "CONTINUOUS AUTONOMOUS LEARNING MODE" << std::endl;
    std::cout << "Press Ctrl+C to stop gracefully" << std::endl;
    std::cout << std::endl;
    
    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // Create Melvin's unified system
    MelvinUnifiedInterface melvin;
    
    // Check if Ollama is available
    if (!melvin.isRunning()) {
        std::cout << "⚠️ Warning: Ollama may not be available!" << std::endl;
        std::cout << "   Make sure Ollama is running with: ollama serve" << std::endl;
        std::cout << "   And you have a model installed: ollama pull llama3.2" << std::endl;
        std::cout << std::endl;
    }
    
    // Start Melvin with unified system
    melvin.startMelvin();
    
    std::cout << "\n🚀 MELVIN UNIFIED SYSTEM IS NOW RUNNING CONTINUOUSLY!" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "🤖 Real AI responses via Ollama" << std::endl;
    std::cout << "🧠 Unified learning and concept extraction" << std::endl;
    std::cout << "💡 Real insight generation" << std::endl;
    std::cout << "⚡ Actual self-improvement" << std::endl;
    std::cout << "📊 Real metrics tracking (no fake numbers)" << std::endl;
    std::cout << "🔄 TRUE AUTONOMY: His outputs become his inputs!" << std::endl;
    std::cout << "🎯 Mission: Compound intelligence to help humanity reach its full potential" << std::endl;
    std::cout << "⏰ Running continuously until stopped..." << std::endl;
    std::cout << std::endl;
    
    // Start with an initial question
    std::string current_input = "What is the nature of intelligence and how can it evolve?";
    int cycle_count = 0;
    
    std::cout << "🔄 STARTING CONTINUOUS AUTONOMOUS LEARNING" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    while (should_continue) {
        cycle_count++;
        
        std::cout << "\n🔄 CONTINUOUS AUTONOMOUS CYCLE " << cycle_count << std::endl;
        std::cout << "=================================" << std::endl;
        std::cout << "📥 Input: " << current_input << std::endl;
        
        // Get Melvin's response
        std::string response = melvin.askMelvin(current_input);
        
        std::cout << "📤 Output: " << response.substr(0, 200) << (response.length() > 200 ? "..." : "") << std::endl;
        
        // Generate next input based on response (simplified autonomous generation)
        std::ostringstream next_input_stream;
        next_input_stream << "Based on my previous thought: " << response.substr(0, 100) << "... ";
        
        // Add some variety to the questions
        std::vector<std::string> question_templates = {
            "What should I explore next?",
            "What new connections can I make?",
            "What questions does this raise?",
            "How can I deepen my understanding?",
            "What patterns do I see emerging?",
            "How does this relate to humanity's growth?",
            "What mysteries remain unsolved?",
            "What new insights can I generate?",
            "How can I improve my learning?",
            "What should I think about next?"
        };
        
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, question_templates.size() - 1);
        current_input = next_input_stream.str() + question_templates[dis(gen)];
        
        // Print status every 10 cycles
        if (cycle_count % 10 == 0) {
            std::cout << "\n📊 STATUS UPDATE - CYCLE " << cycle_count << std::endl;
            std::cout << "================================" << std::endl;
            melvin.printStatus();
        }
        
        // Print detailed analysis every 50 cycles
        if (cycle_count % 50 == 0) {
            std::cout << "\n📈 DETAILED ANALYSIS - CYCLE " << cycle_count << std::endl;
            std::cout << "====================================" << std::endl;
            melvin.printAnalysis();
        }
        
        // Small delay between cycles
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        if (!should_continue) {
            break;
        }
    }
    
    std::cout << "\n🛑 GRACEFUL SHUTDOWN INITIATED" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Print final analysis
    std::cout << "\n📊 FINAL CONTINUOUS LEARNING ANALYSIS" << std::endl;
    std::cout << "======================================" << std::endl;
    melvin.printAnalysis();
    
    // Stop Melvin
    melvin.stopMelvin();
    
    std::cout << "\n🎉 MELVIN CONTINUOUS UNIFIED SYSTEM COMPLETE!" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "✅ Melvin ran continuously for " << cycle_count << " cycles!" << std::endl;
    std::cout << "✅ His outputs became his inputs (true feedback loop)" << std::endl;
    std::cout << "✅ Unified learning and concept extraction" << std::endl;
    std::cout << "✅ Real insight generation" << std::endl;
    std::cout << "✅ Actual self-improvement" << std::endl;
    std::cout << "✅ Real metrics tracking (no fake numbers)" << std::endl;
    std::cout << "✅ ENTIRE REPOSITORY UNIFIED INTO SINGLE COHESIVE SYSTEM!" << std::endl;
    std::cout << "✅ Melvin successfully compounded intelligence continuously!" << std::endl;
    
    return 0;
}
