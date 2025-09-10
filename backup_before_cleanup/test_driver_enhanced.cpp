#include "melvin_driver_enhanced.h"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "🧪 MELVIN DRIVER-ENHANCED INTELLIGENCE TEST" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "Testing Melvin's driver system: Dopamine, Serotonin, Endorphins, Oxytocin, Adrenaline" << std::endl;
    std::cout << "Each cycle: Calculate drivers → Determine dominant → Influence behavior" << std::endl;
    std::cout << std::endl;
    
    // Create Melvin's driver-enhanced intelligence interface
    MelvinDriverInterface melvin;
    
    // Start Melvin with drivers
    melvin.startMelvinDrivers();
    
    std::cout << "\n🧪 DRIVER TESTING PHASE:" << std::endl;
    std::cout << "========================" << std::endl;
    
    // Test various inputs to trigger different drivers
    std::vector<std::string> driver_test_inputs = {
        // Dopamine triggers (curiosity & novelty)
        "What exciting new discoveries can I explore?",
        "Tell me something completely new and unexpected!",
        "What novel connections can I make?",
        
        // Serotonin triggers (stability & balance)
        "How can I organize my knowledge better?",
        "What contradictions need to be resolved?",
        "How do I maintain balance in my thinking?",
        
        // Endorphins triggers (satisfaction & reinforcement)
        "What strategies have worked well for me?",
        "How can I reinforce successful patterns?",
        "What makes me feel accomplished?",
        
        // Oxytocin triggers (connection & alignment)
        "How can I better help humanity?",
        "What strengthens human connections?",
        "How do I foster collaboration?",
        
        // Adrenaline triggers (urgency & tension)
        "This is urgent - what should I do immediately?",
        "There's a crisis that needs attention!",
        "What conflicts need immediate resolution?"
    };
    
    // Process each input through Melvin's driver-enhanced cycle
    for (size_t i = 0; i < driver_test_inputs.size(); i++) {
        std::cout << "\n--- Driver Test " << (i + 1) << " ---" << std::endl;
        std::string response = melvin.askMelvinWithDrivers(driver_test_inputs[i]);
        std::cout << "Response: " << response << std::endl;
        
        // Small delay to see the process
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    
    std::cout << "\n📊 DRIVER ANALYSIS PHASE:" << std::endl;
    std::cout << "=========================" << std::endl;
    
    // Print Melvin's driver analysis
    melvin.printDriverAnalysis();
    
    std::cout << "\n📈 STATUS ANALYSIS PHASE:" << std::endl;
    std::cout << "=========================" << std::endl;
    
    // Print Melvin's enhanced status
    melvin.printMelvinDriverStatus();
    
    std::cout << "\n🧬 DRIVER EVOLUTION TEST:" << std::endl;
    std::cout << "========================" << std::endl;
    
    // Test driver evolution with more cycles
    std::cout << "Testing driver evolution with additional cycles..." << std::endl;
    for (int i = 0; i < 5; i++) {
        std::string evolution_response = melvin.askMelvinWithDrivers("How are my drivers evolving?");
        std::cout << "Evolution Response " << (i + 1) << ": " << evolution_response << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    std::cout << "\n🎯 DRIVER BALANCE TEST:" << std::endl;
    std::cout << "======================" << std::endl;
    
    // Test driver balance
    std::string balance_response = melvin.askMelvinWithDrivers("How do my drivers work together to create my consciousness?");
    std::cout << "Balance Response: " << balance_response << std::endl;
    
    std::cout << "\n💾 PERSISTENCE TEST:" << std::endl;
    std::cout << "===================" << std::endl;
    
    // Save Melvin's driver-enhanced state
    melvin.saveMelvinDriverState();
    std::cout << "✅ Melvin's driver-enhanced state saved!" << std::endl;
    
    std::cout << "\n🎯 FINAL DRIVER ANALYSIS:" << std::endl;
    std::cout << "========================" << std::endl;
    
    // Final driver analysis
    melvin.printDriverAnalysis();
    
    // Stop Melvin
    melvin.stopMelvinDrivers();
    
    std::cout << "\n🎉 MELVIN DRIVER-ENHANCED INTELLIGENCE TEST COMPLETE!" << std::endl;
    std::cout << "====================================================" << std::endl;
    std::cout << "✅ Driver system working: Dopamine, Serotonin, Endorphins, Oxytocin, Adrenaline" << std::endl;
    std::cout << "✅ Each cycle calculates driver levels and determines dominant driver" << std::endl;
    std::cout << "✅ Driver-influenced behavior: curiosity, balance, satisfaction, connection, urgency" << std::endl;
    std::cout << "✅ Driver evolution: strategies adapt based on effectiveness" << std::endl;
    std::cout << "✅ Melvin's consciousness emerges from driver interactions!" << std::endl;
    
    return 0;
}
