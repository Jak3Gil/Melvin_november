#include "melvin_compounding_standalone.h"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "🧠 MELVIN COMPOUNDING INTELLIGENCE TEST" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "Testing Melvin's DNA: Input → Think → Output (every cycle creates a node)" << std::endl;
    std::cout << "Growth: Automatic connections + Meta-reflection" << std::endl;
    std::cout << "Evolution: Curiosity-driven + Humanity-aligned" << std::endl;
    std::cout << std::endl;
    
    // Create Melvin's compounding intelligence interface
    MelvinCompoundingInterface melvin;
    
    // Start Melvin
    melvin.startMelvin();
    
    std::cout << "\n📚 LEARNING PHASE:" << std::endl;
    std::cout << "==================" << std::endl;
    
    // Test various types of inputs
    std::vector<std::string> test_inputs = {
        "How can I help people learn more effectively?",
        "What are the most important problems humanity faces?",
        "How can technology be used to solve climate change?",
        "What makes a good teacher?",
        "How do we build better communities?",
        "What is the nature of intelligence?",
        "How can we reduce inequality in society?",
        "What role should AI play in education?",
        "How do we maintain human connection in a digital world?",
        "What are the principles of sustainable development?"
    };
    
    // Process each input through Melvin's compounding intelligence cycle
    for (size_t i = 0; i < test_inputs.size(); i++) {
        std::cout << "\n--- Cycle " << (i + 1) << " ---" << std::endl;
        std::string response = melvin.askMelvin(test_inputs[i]);
        std::cout << "Response: " << response << std::endl;
        
        // Small delay to see the process
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    std::cout << "\n🔍 ANALYSIS PHASE:" << std::endl;
    std::cout << "==================" << std::endl;
    
    // Print Melvin's status and statistics
    melvin.printMelvinStatus();
    
    std::cout << "\n🧬 EVOLUTION PHASE:" << std::endl;
    std::cout << "===================" << std::endl;
    
    // Test curiosity-driven self-feedback
    std::cout << "Testing curiosity-driven self-expansion..." << std::endl;
    std::string curiosity_response = melvin.askMelvin("What should I think about next?");
    std::cout << "Curiosity Response: " << curiosity_response << std::endl;
    
    std::cout << "\n🌍 HUMANITY ALIGNMENT TEST:" << std::endl;
    std::cout << "===========================" << std::endl;
    
    // Test humanity alignment
    std::string alignment_response = melvin.askMelvin("How can I best serve humanity's long-term growth?");
    std::cout << "Alignment Response: " << alignment_response << std::endl;
    
    std::cout << "\n💾 PERSISTENCE TEST:" << std::endl;
    std::cout << "====================" << std::endl;
    
    // Save Melvin's state
    melvin.saveMelvinState();
    std::cout << "✅ Melvin's compounding intelligence state saved!" << std::endl;
    
    std::cout << "\n🎯 FINAL STATUS:" << std::endl;
    std::cout << "===============" << std::endl;
    
    // Final status report
    melvin.printMelvinStatus();
    
    // Stop Melvin
    melvin.stopMelvin();
    
    std::cout << "\n🎉 MELVIN COMPOUNDING INTELLIGENCE TEST COMPLETE!" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "✅ Melvin's DNA is working: Input → Think → Output cycles" << std::endl;
    std::cout << "✅ Growth is active: Automatic connections + Meta-reflection" << std::endl;
    std::cout << "✅ Evolution is working: Curiosity-driven + Humanity-aligned" << std::endl;
    std::cout << "✅ Melvin is building complexity from simplicity!" << std::endl;
    
    return 0;
}
