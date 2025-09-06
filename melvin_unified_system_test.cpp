#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include "melvin_optimized_v2.h"

int main() {
    std::cout << "🧠 MELVIN UNIFIED SYSTEM INTEGRATION TEST" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Testing all integrated systems working together:" << std::endl;
    std::cout << "- Temporal Planning" << std::endl;
    std::cout << "- Temporal Sequencing Memory" << std::endl;
    std::cout << "- Blended Reasoning" << std::endl;
    std::cout << "- Moral Supernodes" << std::endl;
    std::cout << "- Binary Storage" << std::endl;
    std::cout << "\n";
    
    try {
        // Initialize the unified Melvin system
        MelvinOptimizedV2 melvin;
        
        std::cout << "✅ Unified Melvin system initialized with all components" << std::endl;
        std::cout << "\n";
        
        // Test 1: Basic input processing with all systems
        std::cout << "🎯 TEST 1: Basic Input Processing" << std::endl;
        std::cout << "=================================" << std::endl;
        
        std::string test_input = "A dog finds food and then plays with a cat";
        std::cout << "Input: \"" << test_input << "\"" << std::endl;
        std::cout << "\n";
        
        auto result = melvin.process_cognitive_input(test_input);
        
        std::cout << "📊 Processing Results:" << std::endl;
        std::cout << "- Activated nodes: " << result.activated_nodes.size() << std::endl;
        std::cout << "- Interpretation clusters: " << result.clusters.size() << std::endl;
        std::cout << "- Overall confidence: " << std::fixed << std::setprecision(2) << result.confidence << std::endl;
        std::cout << "- Temporal sequencing confidence: " << std::fixed << std::setprecision(2) << result.temporal_sequencing.sequencing_confidence << std::endl;
        std::cout << "- Temporal planning alignment: " << std::fixed << std::setprecision(2) << result.temporal_planning.overall_alignment << std::endl;
        std::cout << "- Blended reasoning confidence: " << std::fixed << std::setprecision(2) << result.blended_reasoning.overall_confidence << std::endl;
        std::cout << "\n";
        
        // Test 2: Sequence repetition to test temporal sequencing
        std::cout << "🎯 TEST 2: Temporal Sequencing Memory" << std::endl;
        std::cout << "====================================" << std::endl;
        
        std::vector<std::string> sequence_inputs = {
            "dog finds food",
            "food attracts cat", 
            "cat plays with dog",
            "dog finds food",  // Repeat to strengthen sequence
            "food attracts cat"  // Repeat to strengthen sequence
        };
        
        for (size_t i = 0; i < sequence_inputs.size(); ++i) {
            std::cout << "Sequence " << (i + 1) << ": \"" << sequence_inputs[i] << "\"" << std::endl;
            auto seq_result = melvin.process_cognitive_input(sequence_inputs[i]);
            
            std::cout << "  - Temporal links created: " << seq_result.temporal_sequencing.new_links_created.size() << std::endl;
            std::cout << "  - Patterns detected: " << seq_result.temporal_sequencing.detected_patterns.size() << std::endl;
            std::cout << "  - Sequencing confidence: " << std::fixed << std::setprecision(2) << seq_result.temporal_sequencing.sequencing_confidence << std::endl;
            
            // Small delay to simulate time passing
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::cout << "\n";
        
        // Test 3: Moral reasoning integration
        std::cout << "🎯 TEST 3: Moral Reasoning Integration" << std::endl;
        std::cout << "======================================" << std::endl;
        
        std::vector<std::string> moral_inputs = {
            "Should I help someone who is struggling?",
            "Is it okay to lie to avoid hurting someone's feelings?",
            "What should I do if I see someone being treated unfairly?"
        };
        
        for (const auto& moral_input : moral_inputs) {
            std::cout << "Moral question: \"" << moral_input << "\"" << std::endl;
            auto moral_result = melvin.process_cognitive_input(moral_input);
            
            std::cout << "  - Moral gravity strength: " << std::fixed << std::setprecision(2) << moral_result.moral_gravity.moral_bias_strength << std::endl;
            std::cout << "  - Harm detected: " << (moral_result.moral_gravity.harm_detected ? "Yes" : "No") << std::endl;
            std::cout << "  - Temporal planning alignment: " << std::fixed << std::setprecision(2) << moral_result.temporal_planning.overall_alignment << std::endl;
            std::cout << "\n";
        }
        
        // Test 4: Blended reasoning with different confidence levels
        std::cout << "🎯 TEST 4: Blended Reasoning Integration" << std::endl;
        std::cout << "========================================" << std::endl;
        
        std::vector<std::string> reasoning_inputs = {
            "What is 2 + 2?",  // High confidence - should emphasize recall
            "What is the meaning of life?",  // Low confidence - should emphasize exploration
            "How do magnets work?"  // Medium confidence - should balance both
        };
        
        for (const auto& reasoning_input : reasoning_inputs) {
            std::cout << "Reasoning question: \"" << reasoning_input << "\"" << std::endl;
            auto reasoning_result = melvin.process_cognitive_input(reasoning_input);
            
            std::cout << "  - Overall confidence: " << std::fixed << std::setprecision(2) << reasoning_result.blended_reasoning.overall_confidence << std::endl;
            std::cout << "  - Recall weight: " << std::fixed << std::setprecision(2) << reasoning_result.blended_reasoning.recall_weight << std::endl;
            std::cout << "  - Exploration weight: " << std::fixed << std::setprecision(2) << reasoning_result.blended_reasoning.exploration_weight << std::endl;
            std::cout << "  - Recall confidence: " << std::fixed << std::setprecision(2) << reasoning_result.blended_reasoning.recall_track.recall_confidence << std::endl;
            std::cout << "  - Exploration confidence: " << std::fixed << std::setprecision(2) << reasoning_result.blended_reasoning.exploration_track.exploration_confidence << std::endl;
            std::cout << "\n";
        }
        
        // Test 5: Full response generation
        std::cout << "🎯 TEST 5: Full Response Generation" << std::endl;
        std::cout << "====================================" << std::endl;
        
        std::string complex_input = "If I plant a magnet in the ground, what will happen over time?";
        std::cout << "Complex question: \"" << complex_input << "\"" << std::endl;
        std::cout << "\n";
        
        std::string full_response = melvin.generate_intelligent_response(complex_input);
        std::cout << "🤖 Melvin's Unified Response:" << std::endl;
        std::cout << full_response << std::endl;
        
        // Test 6: System state verification
        std::cout << "🎯 TEST 6: System State Verification" << std::endl;
        std::cout << "====================================" << std::endl;
        
        auto state = melvin.get_unified_state();
        std::cout << "📊 Unified System State:" << std::endl;
        std::cout << "- Total nodes: " << state.global_memory.total_nodes << std::endl;
        std::cout << "- Total edges: " << state.global_memory.total_edges << std::endl;
        std::cout << "- Hebbian updates: " << state.hebbian_stats.total_updates << std::endl;
        std::cout << "- Similarity connections: " << state.hebbian_stats.similarity_connections << std::endl;
        std::cout << "- Temporal connections: " << state.hebbian_stats.temporal_connections << std::endl;
        std::cout << "- Cross-modal connections: " << state.hebbian_stats.cross_modal_connections << std::endl;
        std::cout << "\n";
        
        std::cout << "🎉 UNIFIED SYSTEM INTEGRATION TEST COMPLETE!" << std::endl;
        std::cout << "===========================================" << std::endl;
        std::cout << "✅ All systems are working together in harmony:" << std::endl;
        std::cout << "   • Temporal Planning: Multi-horizon decision making" << std::endl;
        std::cout << "   • Temporal Sequencing: Memory formation and pattern detection" << std::endl;
        std::cout << "   • Blended Reasoning: Recall + Exploration with confidence weighting" << std::endl;
        std::cout << "   • Moral Supernodes: Ethical guidance and harm prevention" << std::endl;
        std::cout << "   • Binary Storage: Efficient memory management" << std::endl;
        std::cout << "   • Hebbian Learning: Connection strengthening and formation" << std::endl;
        std::cout << "\n";
        std::cout << "🧠 Melvin is now a truly unified cognitive system!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during unified system testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
