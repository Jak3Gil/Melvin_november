#include <iostream>
#include <string>
#include <vector>
#include "melvin_optimized_v2.h"

void demonstrate_blended_reasoning() {
    std::cout << "🧠 MELVIN BLENDED REASONING PROTOCOL DEMONSTRATION" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    MelvinOptimizedV2 melvin;
    
    // Test cases for different confidence levels and reasoning types
    std::vector<std::string> test_cases = {
        "What happens if you plant a magnet in the ground?",
        "How does artificial intelligence work?",
        "What if gravity worked backwards?",
        "Explain quantum computing to a child",
        "What would happen if the sun disappeared?"
    };
    
    std::cout << "\n🔍 Testing Blended Reasoning Protocol:" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    for (size_t i = 0; i < test_cases.size(); ++i) {
        std::cout << "\n📝 Test Case " << (i + 1) << ": " << test_cases[i] << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // Process through blended reasoning
        std::string response = melvin.generate_intelligent_response(test_cases[i]);
        
        std::cout << "🤖 Melvin's Blended Reasoning Response:" << std::endl;
        std::cout << response << std::endl;
        
        // Update context for next iteration
        auto cognitive_result = melvin.process_cognitive_input(test_cases[i]);
        for (auto node_id : cognitive_result.activated_nodes) {
            melvin.update_conversation_context(node_id);
        }
    }
    
    std::cout << "\n🎯 Confidence-Based Weighting Examples:" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Demonstrate different confidence scenarios
    std::vector<std::string> confidence_tests = {
        "What is 2 + 2?",  // High confidence - should favor recall
        "What is the meaning of life?",  // Low confidence - should favor exploration
        "How do computers work?",  // Medium confidence - should be balanced
    };
    
    for (const auto& test : confidence_tests) {
        std::cout << "\n📊 Confidence Test: " << test << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        auto result = melvin.process_cognitive_input(test);
        
        std::cout << "🧠 Blended Reasoning Analysis:" << std::endl;
        std::cout << "   • Overall Confidence: " << std::fixed << std::setprecision(2) 
                  << result.blended_reasoning.overall_confidence << std::endl;
        std::cout << "   • Recall Weight: " << std::fixed << std::setprecision(0) 
                  << (result.blended_reasoning.recall_weight * 100) << "%" << std::endl;
        std::cout << "   • Exploration Weight: " << std::fixed << std::setprecision(0) 
                  << (result.blended_reasoning.exploration_weight * 100) << "%" << std::endl;
        
        std::cout << "   • Recall Track Confidence: " << std::fixed << std::setprecision(2) 
                  << result.blended_reasoning.recall_track.recall_confidence << std::endl;
        std::cout << "   • Exploration Track Confidence: " << std::fixed << std::setprecision(2) 
                  << result.blended_reasoning.exploration_track.exploration_confidence << std::endl;
        
        std::cout << "   • Integrated Response: " << result.blended_reasoning.integrated_response << std::endl;
    }
    
    std::cout << "\n📈 Final Brain State:" << std::endl;
    auto final_state = melvin.get_unified_state();
    std::cout << "   • Total nodes: " << final_state.global_memory.total_nodes << std::endl;
    std::cout << "   • Total connections: " << final_state.global_memory.total_edges << std::endl;
    std::cout << "   • Storage used: " << std::fixed << std::setprecision(2) 
              << final_state.global_memory.storage_used_mb << " MB" << std::endl;
    
    std::cout << "\n🎉 Blended Reasoning Protocol Demonstration Complete!" << std::endl;
    std::cout << "\nKey Features Demonstrated:" << std::endl;
    std::cout << "✅ Dual-track reasoning (Recall + Exploration)" << std::endl;
    std::cout << "✅ Confidence-based weighting" << std::endl;
    std::cout << "✅ Transparent reasoning paths" << std::endl;
    std::cout << "✅ Integrated response synthesis" << std::endl;
    std::cout << "✅ Context-aware processing" << std::endl;
}

int main() {
    try {
        demonstrate_blended_reasoning();
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during demonstration: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
