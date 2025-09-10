#include <iostream>
#include <string>
#include <vector>
#include "melvin_optimized_v2.h"

void demonstrate_cognitive_pipeline() {
    std::cout << "🧠 MELVIN COGNITIVE PROCESSING DEMONSTRATION" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    MelvinOptimizedV2 melvin;
    
    // Sample inputs to demonstrate different aspects
    std::vector<std::string> test_inputs = {
        "Hello, how are you today?",
        "What is artificial intelligence?",
        "How do neural networks learn?",
        "Explain machine learning algorithms",
        "What are the benefits of deep learning?"
    };
    
    std::cout << "\n🔍 Testing Cognitive Processing Pipeline:" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    for (size_t i = 0; i < test_inputs.size(); ++i) {
        std::cout << "\n📝 Input " << (i + 1) << ": " << test_inputs[i] << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // Process through cognitive pipeline
        auto result = melvin.process_cognitive_input(test_inputs[i]);
        
        std::cout << "🧠 Cognitive Analysis:" << std::endl;
        std::cout << "   • Activated nodes: " << result.activated_nodes.size() << std::endl;
        std::cout << "   • Interpretation clusters: " << result.clusters.size() << std::endl;
        std::cout << "   • Confidence score: " << std::fixed << std::setprecision(2) << result.confidence << std::endl;
        
        if (!result.clusters.empty()) {
            std::cout << "   • Top cluster: " << result.clusters[0].summary << std::endl;
        }
        
        std::cout << "   • Reasoning: " << result.reasoning << std::endl;
        std::cout << "   • Response: " << result.final_response << std::endl;
        
        // Update context for next iteration
        for (auto node_id : result.activated_nodes) {
            melvin.update_conversation_context(node_id);
        }
    }
    
    std::cout << "\n🤖 Testing Intelligent Response Generation:" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    std::string complex_query = "Can you explain how artificial intelligence and machine learning work together?";
    std::cout << "📝 Complex Query: " << complex_query << std::endl;
    
    std::string intelligent_response = melvin.generate_intelligent_response(complex_query);
    std::cout << "\n🧠 Melvin's Response with Thinking Process:" << std::endl;
    std::cout << intelligent_response << std::endl;
    
    std::cout << "\n📊 Final Brain State:" << std::endl;
    auto final_state = melvin.get_unified_state();
    std::cout << "   • Total nodes: " << final_state.global_memory.total_nodes << std::endl;
    std::cout << "   • Total connections: " << final_state.global_memory.total_edges << std::endl;
    std::cout << "   • Storage used: " << std::fixed << std::setprecision(2) 
              << final_state.global_memory.storage_used_mb << " MB" << std::endl;
    std::cout << "   • Hebbian updates: " << final_state.global_memory.stats.hebbian_updates << std::endl;
    
    std::cout << "\n🎉 Cognitive Processing Demonstration Complete!" << std::endl;
}

int main() {
    try {
        demonstrate_cognitive_pipeline();
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during demonstration: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
