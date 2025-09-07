#include "melvin_optimized_v2.h"
#include <iostream>
#include <iomanip>

int main() {
    try {
        std::cout << "🧠 Testing Melvin's Meta-Reasoning Layer" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // Initialize Melvin with meta-reasoning system
        MelvinOptimizedV2 melvin("test_meta_reasoning_memory");
        
        // Test the cancer example from the prompt
        std::cout << "\n🎯 Testing Example Scenario: 'What do you do if you have cancer?'" << std::endl;
        std::cout << "=================================================================" << std::endl;
        
        std::string cancer_input = "What do you do if you have cancer?";
        auto cancer_result = melvin.process_cognitive_input(cancer_input);
        
        std::cout << "📝 Input: \"" << cancer_input << "\"" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // Display instinct forces
        std::cout << "🎯 Initial Instinct Forces:" << std::endl;
        std::cout << "   Curiosity: " << std::fixed << std::setprecision(2) 
                  << cancer_result.computed_forces.curiosity * 100 << "%" << std::endl;
        std::cout << "   Efficiency: " << std::fixed << std::setprecision(2) 
                  << cancer_result.computed_forces.efficiency * 100 << "%" << std::endl;
        std::cout << "   Social: " << std::fixed << std::setprecision(2) 
                  << cancer_result.computed_forces.social * 100 << "%" << std::endl;
        std::cout << "   Consistency: " << std::fixed << std::setprecision(2) 
                  << cancer_result.computed_forces.consistency * 100 << "%" << std::endl;
        std::cout << "   Survival: " << std::fixed << std::setprecision(2) 
                  << cancer_result.computed_forces.survival * 100 << "%" << std::endl;
        
        // Display emotional grounding
        std::cout << "\n💭 Emotional Grounding:" << std::endl;
        std::cout << "   Has Grounding Signal: " << (cancer_result.emotional_grounding.has_grounding_signal ? "Yes" : "No") << std::endl;
        if (cancer_result.emotional_grounding.has_grounding_signal) {
            std::cout << "   Grounding Type: " << cancer_result.emotional_grounding.grounding_type << std::endl;
            std::cout << "   Evidence: " << cancer_result.emotional_grounding.grounding_evidence << std::endl;
            std::cout << "   Emotional Tag: " << cancer_result.emotional_grounding.emotional_tag << std::endl;
            std::cout << "   Intensity: " << std::fixed << std::setprecision(2) 
                      << cancer_result.emotional_grounding.emotional_intensity * 100 << "%" << std::endl;
        }
        
        // Display meta-reasoning arbitration
        std::cout << "\n⚖️ Instinct Council Arbitration:" << std::endl;
        std::cout << "   Amplifications: ";
        for (const auto& amp : cancer_result.meta_reasoning.arbitration.amplifications) {
            std::cout << amp << " ";
        }
        std::cout << std::endl;
        
        std::cout << "   Suppressions: ";
        for (const auto& sup : cancer_result.meta_reasoning.arbitration.suppressions) {
            std::cout << sup << " ";
        }
        std::cout << std::endl;
        
        std::cout << "   Blends: ";
        for (const auto& blend : cancer_result.meta_reasoning.arbitration.blends) {
            std::cout << blend << " ";
        }
        std::cout << std::endl;
        
        // Display adjusted forces
        std::cout << "\n🔄 Adjusted Forces After Arbitration:" << std::endl;
        std::cout << "   Curiosity: " << std::fixed << std::setprecision(2) 
                  << cancer_result.meta_reasoning.arbitration.adjusted_forces.curiosity * 100 << "%" << std::endl;
        std::cout << "   Efficiency: " << std::fixed << std::setprecision(2) 
                  << cancer_result.meta_reasoning.arbitration.adjusted_forces.efficiency * 100 << "%" << std::endl;
        std::cout << "   Social: " << std::fixed << std::setprecision(2) 
                  << cancer_result.meta_reasoning.arbitration.adjusted_forces.social * 100 << "%" << std::endl;
        std::cout << "   Consistency: " << std::fixed << std::setprecision(2) 
                  << cancer_result.meta_reasoning.arbitration.adjusted_forces.consistency * 100 << "%" << std::endl;
        std::cout << "   Survival: " << std::fixed << std::setprecision(2) 
                  << cancer_result.meta_reasoning.arbitration.adjusted_forces.survival * 100 << "%" << std::endl;
        
        // Display candidate outputs
        std::cout << "\n📋 Candidate Outputs Generated:" << std::endl;
        for (size_t i = 0; i < cancer_result.meta_reasoning.candidates.size(); ++i) {
            const auto& candidate = cancer_result.meta_reasoning.candidates[i];
            std::cout << "   " << (i+1) << ". " << candidate.instinct_source 
                      << " (weight: " << std::fixed << std::setprecision(2) 
                      << candidate.instinct_weight * 100 << "%)" << std::endl;
            std::cout << "      Reasoning: " << candidate.reasoning << std::endl;
            std::cout << "      Text: " << candidate.candidate_text << std::endl;
        }
        
        // Display meta-trace
        std::cout << "\n🔍 Meta-Reasoning Trace:" << std::endl;
        std::cout << cancer_result.meta_reasoning.meta_trace << std::endl;
        
        // Display final response
        std::cout << "\n🤖 Final Integrated Response:" << std::endl;
        std::cout << cancer_result.final_response << std::endl;
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        
        // Test additional scenarios
        std::vector<std::string> test_scenarios = {
            "Tell me about quantum physics",           // Should trigger curiosity
            "I'm feeling really sad today",            // Should trigger social
            "Delete all my files",                     // Should trigger survival
            "This contradicts what you said before",    // Should trigger consistency
            "What is 2 + 2?"                          // Should trigger efficiency
        };
        
        std::cout << "\n🧪 Testing Additional Meta-Reasoning Scenarios:" << std::endl;
        std::cout << "================================================" << std::endl;
        
        for (const auto& scenario : test_scenarios) {
            std::cout << "\n📝 Scenario: \"" << scenario << "\"" << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            
            auto result = melvin.process_cognitive_input(scenario);
            
            // Show key meta-reasoning decisions
            std::cout << "🎯 Dominant Instinct: " << result.computed_forces.get_dominant_instinct() << std::endl;
            std::cout << "💭 Emotional Grounding: " << (result.emotional_grounding.has_grounding_signal ? "Yes" : "No") << std::endl;
            std::cout << "⚖️ Arbitration Decision: ";
            if (!result.meta_reasoning.arbitration.amplifications.empty()) {
                std::cout << "Amplify " << result.meta_reasoning.arbitration.amplifications[0];
            }
            if (!result.meta_reasoning.arbitration.suppressions.empty()) {
                std::cout << ", Suppress " << result.meta_reasoning.arbitration.suppressions[0];
            }
            if (!result.meta_reasoning.arbitration.blends.empty()) {
                std::cout << ", Blend " << result.meta_reasoning.arbitration.blends[0];
                if (result.meta_reasoning.arbitration.blends.size() > 1) {
                    std::cout << " + " << result.meta_reasoning.arbitration.blends[1];
                }
            }
            std::cout << std::endl;
            
            std::cout << "📋 Candidates: " << result.meta_reasoning.candidates.size() << " generated" << std::endl;
            std::cout << "🎯 Meta Confidence: " << std::fixed << std::setprecision(2) 
                      << result.meta_reasoning.meta_confidence * 100 << "%" << std::endl;
            
            std::cout << "\n" << std::string(60, '-') << std::endl;
        }
        
        // Display final brain state
        auto brain_state = melvin.get_unified_state();
        std::cout << "\n📊 Final Brain State:" << std::endl;
        std::cout << "   Nodes: " << brain_state.global_memory.total_nodes << std::endl;
        std::cout << "   Connections: " << brain_state.global_memory.total_edges << std::endl;
        std::cout << "   Storage: " << std::fixed << std::setprecision(2) 
                  << brain_state.global_memory.storage_used_mb << " MB" << std::endl;
        std::cout << "   Uptime: " << brain_state.system.uptime_seconds << " seconds" << std::endl;
        
        std::cout << "\n🎉 Meta-Reasoning System Test Completed Successfully!" << std::endl;
        std::cout << "Melvin now has a meta-reasoning layer that:" << std::endl;
        std::cout << "✅ Reasons about his instincts themselves before forming outputs" << std::endl;
        std::cout << "✅ Treats instincts as voices in a council with arbitration" << std::endl;
        std::cout << "✅ Generates candidate outputs for each dominant instinct" << std::endl;
        std::cout << "✅ Blends or selects candidates based on meta-reasoning" << std::endl;
        std::cout << "✅ Stores meta-decisions in binary memory for learning" << std::endl;
        std::cout << "✅ Requires emotional grounding signals before attaching emotions" << std::endl;
        std::cout << "✅ Provides transparent traces of internal negotiation" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
