#include "melvin_instinct_engine.h"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

// ============================================================================
// INSTINCT ENGINE DEMONSTRATION PROGRAM
// ============================================================================

class InstinctEngineDemo {
private:
    InstinctEngine instinct_engine;
    
public:
    InstinctEngineDemo() {
        std::cout << "🧠 Melvin Instinct Engine Demonstration" << std::endl;
        std::cout << "=====================================" << std::endl;
    }
    
    void run_full_demonstration() {
        std::cout << "\n🚀 Starting Full Instinct Engine Demonstration..." << std::endl;
        
        // Test 1: Low confidence scenario (Curiosity vs Efficiency)
        test_low_confidence_scenario();
        
        // Test 2: High resource load scenario
        test_high_resource_load_scenario();
        
        // Test 3: Contradiction detection scenario
        test_contradiction_scenario();
        
        // Test 4: User interaction scenario
        test_user_interaction_scenario();
        
        // Test 5: Memory risk scenario
        test_memory_risk_scenario();
        
        // Test 6: Reinforcement learning demonstration
        test_reinforcement_learning();
        
        // Test 7: Complex multi-factor scenario
        test_complex_scenario();
        
        // Test 8: Statistics and monitoring
        test_statistics_monitoring();
        
        std::cout << "\n✅ Demonstration Complete!" << std::endl;
    }
    
private:
    void test_low_confidence_scenario() {
        std::cout << "\n📊 Test 1: Low Confidence Scenario (Curiosity vs Efficiency)" << std::endl;
        std::cout << "------------------------------------------------------------" << std::endl;
        
        // Create low confidence, moderate resource load scenario
        ContextState context = instinct_engine.analyze_context(
            0.25f,  // Low confidence
            0.4f,   // Moderate resource load
            false,  // No contradictions
            true,   // User interaction
            false,  // No memory risk
            0.8f,   // High novelty
            800     // Moderate complexity
        );
        
        InstinctBias bias = instinct_engine.get_instinct_bias(context);
        
        std::cout << "Context:" << std::endl;
        std::cout << "- Confidence Level: " << std::fixed << std::setprecision(2) << context.confidence_level << std::endl;
        std::cout << "- Resource Load: " << context.resource_load << std::endl;
        std::cout << "- Novelty Level: " << context.novelty_level << std::endl;
        std::cout << "- User Interaction: " << (context.user_interaction ? "Yes" : "No") << std::endl;
        
        std::cout << "\nInstinct Contributions:" << std::endl;
        for (const auto& contribution : bias.instinct_contributions) {
            std::cout << "- " << instinct_engine.instinct_type_to_string(contribution.first) 
                      << ": " << std::fixed << std::setprecision(3) << contribution.second << std::endl;
        }
        
        std::cout << "\nFinal Decision Bias:" << std::endl;
        std::cout << "- Recall Track: " << std::fixed << std::setprecision(1) 
                  << (bias.recall_weight * 100) << "%" << std::endl;
        std::cout << "- Exploration Track: " << (bias.exploration_weight * 100) << "%" << std::endl;
        
        std::cout << "\nReasoning: " << bias.reasoning << std::endl;
        
        // Demonstrate reinforcement
        std::cout << "\nApplying reinforcement (Curiosity success):" << std::endl;
        instinct_engine.reinforce_instinct(InstinctType::CURIOSITY, 0.2f, 
                                         "Successful exploration in low confidence scenario", 12345);
    }
    
    void test_high_resource_load_scenario() {
        std::cout << "\n⚡ Test 2: High Resource Load Scenario" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        
        ContextState context = instinct_engine.analyze_context(
            0.6f,   // Moderate confidence
            0.9f,   // High resource load
            false,  // No contradictions
            false,  // No user interaction
            false,  // No memory risk
            0.3f,   // Low novelty
            2000    // High complexity
        );
        
        InstinctBias bias = instinct_engine.get_instinct_bias(context);
        
        std::cout << "Context:" << std::endl;
        std::cout << "- Resource Load: " << std::fixed << std::setprecision(2) << context.resource_load << std::endl;
        std::cout << "- Input Complexity: " << context.input_complexity << std::endl;
        
        std::cout << "\nInstinct Contributions:" << std::endl;
        for (const auto& contribution : bias.instinct_contributions) {
            std::cout << "- " << instinct_engine.instinct_type_to_string(contribution.first) 
                      << ": " << std::fixed << std::setprecision(3) << contribution.second << std::endl;
        }
        
        std::cout << "\nFinal Decision Bias:" << std::endl;
        std::cout << "- Recall Track: " << std::fixed << std::setprecision(1) 
                  << (bias.recall_weight * 100) << "%" << std::endl;
        std::cout << "- Exploration Track: " << (bias.exploration_weight * 100) << "%" << std::endl;
        
        std::cout << "\nReasoning: " << bias.reasoning << std::endl;
        
        // Demonstrate reinforcement
        std::cout << "\nApplying reinforcement (Efficiency success):" << std::endl;
        instinct_engine.reinforce_instinct(InstinctType::EFFICIENCY, 0.15f, 
                                         "Efficient processing under high load", 12346);
    }
    
    void test_contradiction_scenario() {
        std::cout << "\n🔄 Test 3: Contradiction Detection Scenario" << std::endl;
        std::cout << "--------------------------------------------" << std::endl;
        
        ContextState context = instinct_engine.analyze_context(
            0.7f,   // High confidence
            0.3f,   // Low resource load
            true,   // Has contradictions
            false,  // No user interaction
            false,  // No memory risk
            0.4f,   // Moderate novelty
            600     // Moderate complexity
        );
        
        InstinctBias bias = instinct_engine.get_instinct_bias(context);
        
        std::cout << "Context:" << std::endl;
        std::cout << "- Confidence Level: " << std::fixed << std::setprecision(2) << context.confidence_level << std::endl;
        std::cout << "- Has Contradictions: " << (context.has_contradictions ? "Yes" : "No") << std::endl;
        
        std::cout << "\nInstinct Contributions:" << std::endl;
        for (const auto& contribution : bias.instinct_contributions) {
            std::cout << "- " << instinct_engine.instinct_type_to_string(contribution.first) 
                      << ": " << std::fixed << std::setprecision(3) << contribution.second << std::endl;
        }
        
        std::cout << "\nFinal Decision Bias:" << std::endl;
        std::cout << "- Recall Track: " << std::fixed << std::setprecision(1) 
                  << (bias.recall_weight * 100) << "%" << std::endl;
        std::cout << "- Exploration Track: " << (bias.exploration_weight * 100) << "%" << std::endl;
        
        std::cout << "\nReasoning: " << bias.reasoning << std::endl;
        
        // Demonstrate reinforcement
        std::cout << "\nApplying reinforcement (Consistency success):" << std::endl;
        instinct_engine.reinforce_instinct(InstinctType::CONSISTENCY, 0.1f, 
                                         "Successfully resolved contradictions", 12347);
    }
    
    void test_user_interaction_scenario() {
        std::cout << "\n👤 Test 4: User Interaction Scenario" << std::endl;
        std::cout << "-------------------------------------" << std::endl;
        
        ContextState context = instinct_engine.analyze_context(
            0.5f,   // Moderate confidence
            0.2f,   // Low resource load
            false,  // No contradictions
            true,   // User interaction
            false,  // No memory risk
            0.6f,   // Moderate novelty
            400     // Low complexity
        );
        
        InstinctBias bias = instinct_engine.get_instinct_bias(context);
        
        std::cout << "Context:" << std::endl;
        std::cout << "- User Interaction: " << (context.user_interaction ? "Yes" : "No") << std::endl;
        std::cout << "- Resource Load: " << std::fixed << std::setprecision(2) << context.resource_load << std::endl;
        
        std::cout << "\nInstinct Contributions:" << std::endl;
        for (const auto& contribution : bias.instinct_contributions) {
            std::cout << "- " << instinct_engine.instinct_type_to_string(contribution.first) 
                      << ": " << std::fixed << std::setprecision(3) << contribution.second << std::endl;
        }
        
        std::cout << "\nFinal Decision Bias:" << std::endl;
        std::cout << "- Recall Track: " << std::fixed << std::setprecision(1) 
                  << (bias.recall_weight * 100) << "%" << std::endl;
        std::cout << "- Exploration Track: " << (bias.exploration_weight * 100) << "%" << std::endl;
        
        std::cout << "\nReasoning: " << bias.reasoning << std::endl;
        
        // Demonstrate reinforcement
        std::cout << "\nApplying reinforcement (Social success):" << std::endl;
        instinct_engine.reinforce_instinct(InstinctType::SOCIAL, 0.12f, 
                                         "Positive user interaction", 12348);
    }
    
    void test_memory_risk_scenario() {
        std::cout << "\n⚠️ Test 5: Memory Risk Scenario" << std::endl;
        std::cout << "--------------------------------" << std::endl;
        
        ContextState context = instinct_engine.analyze_context(
            0.4f,   // Moderate confidence
            0.8f,   // High resource load
            false,  // No contradictions
            false,  // No user interaction
            true,   // Memory risk detected
            0.2f,   // Low novelty
            3000    // Very high complexity
        );
        
        InstinctBias bias = instinct_engine.get_instinct_bias(context);
        
        std::cout << "Context:" << std::endl;
        std::cout << "- Memory Risk: " << (context.memory_risk ? "Yes" : "No") << std::endl;
        std::cout << "- Resource Load: " << std::fixed << std::setprecision(2) << context.resource_load << std::endl;
        std::cout << "- Input Complexity: " << context.input_complexity << std::endl;
        
        std::cout << "\nInstinct Contributions:" << std::endl;
        for (const auto& contribution : bias.instinct_contributions) {
            std::cout << "- " << instinct_engine.instinct_type_to_string(contribution.first) 
                      << ": " << std::fixed << std::setprecision(3) << contribution.second << std::endl;
        }
        
        std::cout << "\nFinal Decision Bias:" << std::endl;
        std::cout << "- Recall Track: " << std::fixed << std::setprecision(1) 
                  << (bias.recall_weight * 100) << "%" << std::endl;
        std::cout << "- Exploration Track: " << (bias.exploration_weight * 100) << "%" << std::endl;
        
        std::cout << "\nReasoning: " << bias.reasoning << std::endl;
        
        // Demonstrate reinforcement
        std::cout << "\nApplying reinforcement (Survival success):" << std::endl;
        instinct_engine.reinforce_instinct(InstinctType::SURVIVAL, 0.18f, 
                                         "Successfully avoided memory corruption", 12349);
    }
    
    void test_reinforcement_learning() {
        std::cout << "\n🎯 Test 6: Reinforcement Learning Demonstration" << std::endl;
        std::cout << "----------------------------------------------" << std::endl;
        
        // Show initial weights
        auto stats_before = instinct_engine.get_instinct_statistics();
        std::cout << "Initial Instinct Weights:" << std::endl;
        for (const auto& weight : stats_before.current_weights) {
            std::cout << "- " << instinct_engine.instinct_type_to_string(weight.first) 
                      << ": " << std::fixed << std::setprecision(3) << weight.second << std::endl;
        }
        
        // Apply multiple reinforcements
        std::cout << "\nApplying multiple reinforcements..." << std::endl;
        
        // Simulate successful Curiosity-driven exploration
        instinct_engine.reinforce_instinct(InstinctType::CURIOSITY, 0.2f, "Successful exploration", 20001);
        instinct_engine.reinforce_instinct(InstinctType::CURIOSITY, 0.15f, "Novel discovery", 20002);
        instinct_engine.reinforce_instinct(InstinctType::CURIOSITY, 0.1f, "Learning success", 20003);
        
        // Simulate failed Efficiency attempts
        instinct_engine.reinforce_instinct(InstinctType::EFFICIENCY, -0.1f, "Inefficient processing", 20004);
        instinct_engine.reinforce_instinct(InstinctType::EFFICIENCY, -0.05f, "Resource waste", 20005);
        
        // Simulate successful Social interactions
        instinct_engine.reinforce_instinct(InstinctType::SOCIAL, 0.12f, "User satisfaction", 20006);
        instinct_engine.reinforce_instinct(InstinctType::SOCIAL, 0.08f, "Helpful response", 20007);
        
        // Show updated weights
        auto stats_after = instinct_engine.get_instinct_statistics();
        std::cout << "\nUpdated Instinct Weights:" << std::endl;
        for (const auto& weight : stats_after.current_weights) {
            std::cout << "- " << instinct_engine.instinct_type_to_string(weight.first) 
                      << ": " << std::fixed << std::setprecision(3) << weight.second << std::endl;
        }
        
        std::cout << "\nTotal Reinforcements Applied: " << stats_after.total_reinforcements << std::endl;
    }
    
    void test_complex_scenario() {
        std::cout << "\n🧩 Test 7: Complex Multi-Factor Scenario" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // Create a complex scenario with multiple conflicting factors
        ContextState context = instinct_engine.analyze_context(
            0.35f,  // Low confidence (triggers Curiosity)
            0.75f,  // High resource load (triggers Efficiency)
            true,   // Has contradictions (triggers Consistency)
            true,   // User interaction (triggers Social)
            false,  // No memory risk
            0.7f,   // High novelty (triggers Curiosity)
            1500    // High complexity (triggers Efficiency)
        );
        
        InstinctBias bias = instinct_engine.get_instinct_bias(context);
        
        std::cout << "Complex Context:" << std::endl;
        std::cout << "- Confidence: " << std::fixed << std::setprecision(2) << context.confidence_level << std::endl;
        std::cout << "- Resource Load: " << context.resource_load << std::endl;
        std::cout << "- Contradictions: " << (context.has_contradictions ? "Yes" : "No") << std::endl;
        std::cout << "- User Interaction: " << (context.user_interaction ? "Yes" : "No") << std::endl;
        std::cout << "- Novelty: " << context.novelty_level << std::endl;
        std::cout << "- Complexity: " << context.input_complexity << std::endl;
        
        std::cout << "\nInstinct Contributions (Competing Drives):" << std::endl;
        for (const auto& contribution : bias.instinct_contributions) {
            std::cout << "- " << instinct_engine.instinct_type_to_string(contribution.first) 
                      << ": " << std::fixed << std::setprecision(3) << contribution.second << std::endl;
        }
        
        std::cout << "\nConflict Resolution:" << std::endl;
        std::cout << "- Recall Track: " << std::fixed << std::setprecision(1) 
                  << (bias.recall_weight * 100) << "%" << std::endl;
        std::cout << "- Exploration Track: " << (bias.exploration_weight * 100) << "%" << std::endl;
        std::cout << "- Overall Instinct Strength: " << std::fixed << std::setprecision(3) 
                  << bias.overall_strength << std::endl;
        
        std::cout << "\nReasoning: " << bias.reasoning << std::endl;
        
        // Generate instinct tags
        std::vector<InstinctTag> tags = instinct_engine.generate_instinct_tags(bias, "Complex multi-factor decision");
        std::cout << "\nGenerated Instinct Tags:" << std::endl;
        std::cout << instinct_engine.format_instinct_tags(tags) << std::endl;
    }
    
    void test_statistics_monitoring() {
        std::cout << "\n📈 Test 8: Statistics and Monitoring" << std::endl;
        std::cout << "------------------------------------" << std::endl;
        
        auto stats = instinct_engine.get_instinct_statistics();
        std::cout << instinct_engine.format_instinct_statistics(stats) << std::endl;
        
        // Demonstrate context formatting
        ContextState sample_context = instinct_engine.analyze_context(0.5f, 0.5f, false, true, false, 0.5f, 1000);
        std::cout << "\nSample Context State:" << std::endl;
        std::cout << InstinctIntegrationHelper::format_context_state(sample_context) << std::endl;
        
        // Demonstrate bias formatting
        InstinctBias sample_bias = instinct_engine.get_instinct_bias(sample_context);
        std::cout << "\nSample Instinct Bias:" << std::endl;
        std::cout << InstinctIntegrationHelper::format_instinct_bias(sample_bias) << std::endl;
    }
};

// ============================================================================
// MAIN DEMONSTRATION FUNCTION
// ============================================================================

int main() {
    std::cout << "🧠 Melvin Instinct Engine - Unified Brain DNA" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "This demonstration shows how Melvin's five core instincts" << std::endl;
    std::cout << "(Survival, Curiosity, Efficiency, Social, Consistency)" << std::endl;
    std::cout << "compete and collaborate to bias reasoning, learning, and tool use." << std::endl;
    
    InstinctEngineDemo demo;
    demo.run_full_demonstration();
    
    std::cout << "\n🎯 Key Insights:" << std::endl;
    std::cout << "- Instincts dynamically adjust based on context" << std::endl;
    std::cout << "- Low confidence → Curiosity dominance" << std::endl;
    std::cout << "- High resource load → Efficiency dominance" << std::endl;
    std::cout << "- Contradictions → Consistency dominance" << std::endl;
    std::cout << "- User interaction → Social dominance" << std::endl;
    std::cout << "- Memory risk → Survival dominance" << std::endl;
    std::cout << "- Reinforcement learning adapts instinct strengths over time" << std::endl;
    std::cout << "- Softmax normalization resolves instinct conflicts" << std::endl;
    
    std::cout << "\n🚀 The Instinct Engine is ready for integration with Melvin's unified brain!" << std::endl;
    
    return 0;
}
