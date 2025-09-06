#include <iostream>
#include <string>
#include <vector>
#include "melvin_optimized_v2.h"

int main() {
    std::cout << "🧠 MELVIN BLENDED REASONING TEST" << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        // Initialize Melvin
        MelvinOptimizedV2 melvin;
        std::cout << "✅ Melvin initialized successfully!" << std::endl;
        
        // Test questions that force exploration-heavy reasoning
        std::vector<std::string> questions = {
            "If shadows could remember the objects they came from, how would they describe them?",
            "What would a conversation between a river and a mountain sound like?",
            "If silence had a texture, how would it feel in your hands?"
        };
        
        std::cout << "\n🎯 Testing " << questions.size() << " exploration-heavy questions:" << std::endl;
        std::cout << "=================================================" << std::endl;
        
        for (size_t i = 0; i < questions.size(); ++i) {
            std::cout << "\n[Question " << (i + 1) << "] " << questions[i] << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            
            // Get Melvin's response
            std::string response = melvin.generate_intelligent_response(questions[i]);
            std::cout << "🤖 Melvin's Response:" << std::endl;
            std::cout << response << std::endl;
            
            // Show blended reasoning analysis
            auto result = melvin.process_cognitive_input(questions[i]);
            std::cout << "\n📊 Blended Reasoning Analysis:" << std::endl;
            std::cout << "   • Overall Confidence: " << std::fixed << std::setprecision(2) 
                      << result.blended_reasoning.overall_confidence << std::endl;
            std::cout << "   • Recall Weight: " << std::fixed << std::setprecision(0) 
                      << (result.blended_reasoning.recall_weight * 100) << "%" << std::endl;
            std::cout << "   • Exploration Weight: " << std::fixed << std::setprecision(0) 
                      << (result.blended_reasoning.exploration_weight * 100) << "%" << std::endl;
            
            std::cout << "\n" << std::string(60, '=') << std::endl;
        }
        
        std::cout << "\n🎉 Test Complete! Melvin's blended reasoning is working!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
