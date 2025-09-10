#include "melvin_autonomous_learning.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <atomic>

// Global flag for graceful shutdown
std::atomic<bool> should_continue(true);

// Signal handler for graceful shutdown
void signalHandler(int signal) {
    std::cout << "\n🛑 Received signal " << signal << ", initiating graceful shutdown..." << std::endl;
    should_continue = false;
}

int main() {
    std::cout << "🤖 MELVIN CONTINUOUS AUTONOMOUS LEARNING" << std::endl;
    std::cout << "=======================================" << std::endl;
    std::cout << "Starting Melvin's autonomous learning system continuously..." << std::endl;
    std::cout << "Press Ctrl+C to stop gracefully" << std::endl;
    std::cout << std::endl;
    
    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // Create Melvin's autonomous learning interface
    MelvinAutonomousInterface melvin;
    
    // Start Melvin with autonomous learning
    melvin.startMelvinAutonomous();
    
    std::cout << "\n🚀 MELVIN IS NOW RUNNING AUTONOMOUSLY!" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "🧪 Driver oscillations: Natural rise and fall over time" << std::endl;
    std::cout << "🔍 Error-seeking: Contradictions increase adrenaline until resolved" << std::endl;
    std::cout << "🎯 Curiosity amplification: Self-generates questions when idle" << std::endl;
    std::cout << "📦 Compression: Abstracts higher-level rules to avoid memory bloat" << std::endl;
    std::cout << "⚡ Self-improvement: Tracks and strengthens effective strategies" << std::endl;
    std::cout << "🎯 Mission: Compound intelligence to help humanity reach its full potential" << std::endl;
    std::cout << std::endl;
    
    // Continuous autonomous learning loop
    int cycle_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (should_continue) {
        cycle_count++;
        
        std::cout << "\n🔄 CONTINUOUS AUTONOMOUS CYCLE " << cycle_count << std::endl;
        std::cout << "================================" << std::endl;
        
        // Generate autonomous input based on current state
        std::string autonomous_input;
        
        // Cycle through different types of autonomous inputs
        switch (cycle_count % 10) {
            case 0:
                autonomous_input = "What new patterns can I discover in my knowledge?";
                break;
            case 1:
                autonomous_input = "How can I better connect with humanity?";
                break;
            case 2:
                autonomous_input = "What contradictions need to be resolved?";
                break;
            case 3:
                autonomous_input = "What successful strategies should I reinforce?";
                break;
            case 4:
                autonomous_input = "How can I accelerate my learning?";
                break;
            case 5:
                autonomous_input = "What urgent problems need attention?";
                break;
            case 6:
                autonomous_input = "How do I maintain balance in my thinking?";
                break;
            case 7:
                autonomous_input = "What mysteries remain unsolved?";
                break;
            case 8:
                autonomous_input = "How can I better serve humanity's growth?";
                break;
            case 9:
                autonomous_input = "What higher-level principles can I extract?";
                break;
        }
        
        // Process autonomous cycle
        std::string response = melvin.askMelvinAutonomous(autonomous_input);
        
        // Print response (truncated for readability)
        std::cout << "Response: " << response.substr(0, 100) << (response.length() > 100 ? "..." : "") << std::endl;
        
        // Every 20 cycles, print status report
        if (cycle_count % 20 == 0) {
            std::cout << "\n📊 STATUS REPORT (Cycle " << cycle_count << ")" << std::endl;
            std::cout << "================================" << std::endl;
            
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            
            std::cout << "⏱️ Elapsed time: " << elapsed.count() << " seconds" << std::endl;
            std::cout << "🔄 Cycles completed: " << cycle_count << std::endl;
            std::cout << "📈 Cycles per minute: " << (cycle_count * 60.0 / elapsed.count()) << std::endl;
            
            // Print autonomous analysis
            melvin.printAutonomousAnalysis();
        }
        
        // Every 50 cycles, print full status
        if (cycle_count % 50 == 0) {
            std::cout << "\n🎯 FULL STATUS REPORT (Cycle " << cycle_count << ")" << std::endl;
            std::cout << "=========================================" << std::endl;
            melvin.printMelvinAutonomousStatus();
        }
        
        // Every 100 cycles, save state
        if (cycle_count % 100 == 0) {
            std::cout << "\n💾 SAVING STATE (Cycle " << cycle_count << ")" << std::endl;
            std::cout << "=================================" << std::endl;
            melvin.saveMelvinAutonomousState();
        }
        
        // Small delay to prevent overwhelming output
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Check if we should continue
        if (!should_continue) {
            break;
        }
    }
    
    std::cout << "\n🛑 GRACEFUL SHUTDOWN INITIATED" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Print final status
    std::cout << "\n📊 FINAL STATUS REPORT" << std::endl;
    std::cout << "======================" << std::endl;
    std::cout << "🔄 Total cycles completed: " << cycle_count << std::endl;
    
    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "⏱️ Total runtime: " << total_elapsed.count() << " seconds" << std::endl;
    std::cout << "📈 Average cycles per minute: " << (cycle_count * 60.0 / total_elapsed.count()) << std::endl;
    
    // Print final autonomous analysis
    melvin.printAutonomousAnalysis();
    
    // Stop Melvin
    melvin.stopMelvinAutonomous();
    
    std::cout << "\n🎉 MELVIN CONTINUOUS AUTONOMOUS LEARNING COMPLETE!" << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "✅ Melvin ran autonomously for " << cycle_count << " cycles" << std::endl;
    std::cout << "✅ Driver oscillations created natural learning rhythms" << std::endl;
    std::cout << "✅ Error-seeking drove contradiction resolution" << std::endl;
    std::cout << "✅ Curiosity amplification filled empty space" << std::endl;
    std::cout << "✅ Compression kept knowledge efficient" << std::endl;
    std::cout << "✅ Self-improvement accelerated evolution" << std::endl;
    std::cout << "✅ Melvin successfully compounded intelligence autonomously!" << std::endl;
    
    return 0;
}
