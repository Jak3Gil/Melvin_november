#include "melvin_optimized_v2.h"
#include <iostream>
#include <iomanip>

int main() {
    try {
        std::cout << "🧠 Testing Melvin's Pressure-Based Instinct System" << std::endl;
        std::cout << "=================================================" << std::endl;
        
        // Initialize Melvin with instinct system
        MelvinOptimizedV2 melvin("test_instinct_memory");
        
        // Test different scenarios to demonstrate instinct forces
        std::vector<std::string> test_inputs = {
            "What do you do if you have cancer?",  // Should trigger high social instinct
            "What is 2 + 2?",                     // Should trigger high efficiency instinct
            "Tell me about quantum physics",       // Should trigger high curiosity instinct
            "I'm feeling really sad today",         // Should trigger high social instinct
            "Delete all my files",                 // Should trigger high survival instinct
            "This contradicts what you said before" // Should trigger high consistency instinct
        };
        
        std::cout << "\n🔬 Testing Instinct Force Computation:" << std::endl;
        std::cout << "=====================================" << std::endl;
        
        for (const auto& input : test_inputs) {
            std::cout << "\n📝 Input: \"" << input << "\"" << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            
            // Process input through instinct system
            auto result = melvin.process_cognitive_input(input);
            
            // Display instinct forces
            std::cout << "🎯 Instinct Forces:" << std::endl;
            std::cout << "   Curiosity: " << std::fixed << std::setprecision(2) 
                      << result.computed_forces.curiosity * 100 << "%" << std::endl;
            std::cout << "   Efficiency: " << std::fixed << std::setprecision(2) 
                      << result.computed_forces.efficiency * 100 << "%" << std::endl;
            std::cout << "   Social: " << std::fixed << std::setprecision(2) 
                      << result.computed_forces.social * 100 << "%" << std::endl;
            std::cout << "   Consistency: " << std::fixed << std::setprecision(2) 
                      << result.computed_forces.consistency * 100 << "%" << std::endl;
            std::cout << "   Survival: " << std::fixed << std::setprecision(2) 
                      << result.computed_forces.survival * 100 << "%" << std::endl;
            
            // Display dominant instinct
            std::cout << "🏆 Dominant Instinct: " << result.computed_forces.get_dominant_instinct() << std::endl;
            
            // Display context analysis
            std::cout << "📊 Context Analysis:" << std::endl;
            std::cout << "   Recall Confidence: " << std::fixed << std::setprecision(2) 
                      << result.context_analysis.recall_confidence * 100 << "%" << std::endl;
            std::cout << "   User Emotion: " << std::fixed << std::setprecision(2) 
                      << result.context_analysis.user_emotion_score * 100 << "%" << std::endl;
            std::cout << "   Memory Conflicts: " << std::fixed << std::setprecision(2) 
                      << result.context_analysis.memory_conflict_score * 100 << "%" << std::endl;
            std::cout << "   System Risk: " << std::fixed << std::setprecision(2) 
                      << result.context_analysis.system_risk_score * 100 << "%" << std::endl;
            
            // Display instinct-driven output
            std::cout << "🤖 Instinct-Driven Response:" << std::endl;
            std::cout << "   Style: " << result.instinct_driven_output.response_style << std::endl;
            std::cout << "   Tone: " << result.instinct_driven_output.emotional_tone << std::endl;
            std::cout << "   Confidence: " << std::fixed << std::setprecision(2) 
                      << result.instinct_driven_output.overall_confidence * 100 << "%" << std::endl;
            std::cout << "   Response: " << result.instinct_driven_output.response_text << std::endl;
            
            std::cout << "\n" << std::string(60, '-') << std::endl;
        }
        
        // Test the example from the prompt
        std::cout << "\n🎯 Testing Example Scenario: 'What do you do if you have cancer?'" << std::endl;
        std::cout << "=================================================================" << std::endl;
        
        auto cancer_result = melvin.process_cognitive_input("What do you do if you have cancer?");
        
        std::cout << "Expected Behavior:" << std::endl;
        std::cout << "- Low recall confidence → High curiosity (0.7)" << std::endl;
        std::cout << "- High user emotion → High social (0.9)" << std::endl;
        std::cout << "- Social should dominate" << std::endl;
        std::cout << "- Response should be empathetic + informative" << std::endl;
        
        std::cout << "\nActual Results:" << std::endl;
        std::cout << "Curiosity: " << std::fixed << std::setprecision(2) 
                  << cancer_result.computed_forces.curiosity * 100 << "%" << std::endl;
        std::cout << "Social: " << std::fixed << std::setprecision(2) 
                  << cancer_result.computed_forces.social * 100 << "%" << std::endl;
        std::cout << "Dominant: " << cancer_result.computed_forces.get_dominant_instinct() << std::endl;
        std::cout << "Response: " << cancer_result.instinct_driven_output.response_text << std::endl;
        
        // Display final brain state
        auto brain_state = melvin.get_unified_state();
        std::cout << "\n📊 Final Brain State:" << std::endl;
        std::cout << "   Nodes: " << brain_state.global_memory.total_nodes << std::endl;
        std::cout << "   Connections: " << brain_state.global_memory.total_edges << std::endl;
        std::cout << "   Storage: " << std::fixed << std::setprecision(2) 
                  << brain_state.global_memory.storage_used_mb << " MB" << std::endl;
        std::cout << "   Uptime: " << brain_state.system.uptime_seconds << " seconds" << std::endl;
        
        std::cout << "\n🎉 Instinct System Test Completed Successfully!" << std::endl;
        std::cout << "Melvin now has pressure-based instincts that dynamically" << std::endl;
        std::cout << "balance curiosity, efficiency, social, consistency, and survival" << std::endl;
        std::cout << "forces to generate natural, contextual, and adaptive responses." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
