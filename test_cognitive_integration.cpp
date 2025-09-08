#include <iostream>
#include <string>
#include "melvin_optimized_v2.h"

int main() {
    std::cout << "🧠 Testing Melvin Cognitive Integration..." << std::endl;
    
    try {
        // Initialize Melvin with cognitive processing
        MelvinOptimizedV2 melvin;
        
        // Test basic functionality
        std::cout << "📝 Testing basic text processing..." << std::endl;
        uint64_t node_id = melvin.process_text_input("Hello, this is a test of cognitive processing!");
        std::cout << "✅ Created node: " << std::hex << node_id << std::endl;
        
        // Test cognitive processing
        std::cout << "🧠 Testing cognitive processing..." << std::endl;
        std::string user_input = "How does artificial intelligence work?";
        auto cognitive_result = melvin.process_cognitive_input(user_input);
        
        std::cout << "📊 Cognitive Results:" << std::endl;
        std::cout << "   Activated nodes: " << cognitive_result.activated_nodes.size() << std::endl;
        std::cout << "   Clusters formed: " << cognitive_result.clusters.size() << std::endl;
        std::cout << "   Confidence: " << cognitive_result.confidence << std::endl;
        std::cout << "   Response: " << cognitive_result.final_response << std::endl;
        
        // Test intelligent response generation
        std::cout << "🤖 Testing intelligent response generation..." << std::endl;
        std::string intelligent_response = melvin.generate_intelligent_response("What is machine learning?");
        std::cout << "🧠 Intelligent Response:\n" << intelligent_response << std::endl;
        
        // Test conversation context
        std::cout << "💬 Testing conversation context..." << std::endl;
        melvin.update_conversation_context(node_id);
        std::cout << "✅ Context updated" << std::endl;
        
        // Test goal setting
        std::cout << "🎯 Testing goal setting..." << std::endl;
        std::vector<uint64_t> goals = {node_id};
        melvin.set_current_goals(goals);
        std::cout << "✅ Goals set" << std::endl;
        
        std::cout << "\n🎉 All cognitive integration tests passed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
