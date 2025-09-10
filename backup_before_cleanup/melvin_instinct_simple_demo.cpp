#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <iomanip>
#include <sstream>

// ============================================================================
// SIMPLIFIED INSTINCT ENGINE DEMONSTRATION
// ============================================================================

enum class InstinctType {
    SURVIVAL = 0,
    CURIOSITY = 1,
    EFFICIENCY = 2,
    SOCIAL = 3,
    CONSISTENCY = 4
};

struct InstinctWeight {
    InstinctType instinct;
    float weight;
    float base_weight;
    
    InstinctWeight() : instinct(InstinctType::SURVIVAL), weight(0.0f), base_weight(0.0f) {}
    InstinctWeight(InstinctType inst, float w, float base_w) 
        : instinct(inst), weight(w), base_weight(base_w) {}
};

struct ContextState {
    float confidence_level;
    float resource_load;
    bool has_contradictions;
    bool user_interaction;
    bool memory_risk;
    float novelty_level;
    
    ContextState() : confidence_level(0.5f), resource_load(0.0f), has_contradictions(false),
                    user_interaction(false), memory_risk(false), novelty_level(0.5f) {}
};

struct InstinctBias {
    float recall_weight;
    float exploration_weight;
    std::map<InstinctType, float> instinct_contributions;
    std::string reasoning;
    
    InstinctBias() : recall_weight(0.5f), exploration_weight(0.5f) {}
};

class InstinctEngine {
private:
    std::map<InstinctType, InstinctWeight> instinct_weights;
    
public:
    InstinctEngine() {
        // Initialize with default weights
        instinct_weights[InstinctType::SURVIVAL] = InstinctWeight(InstinctType::SURVIVAL, 0.8f, 0.8f);
        instinct_weights[InstinctType::CURIOSITY] = InstinctWeight(InstinctType::CURIOSITY, 0.6f, 0.6f);
        instinct_weights[InstinctType::EFFICIENCY] = InstinctWeight(InstinctType::EFFICIENCY, 0.5f, 0.5f);
        instinct_weights[InstinctType::SOCIAL] = InstinctWeight(InstinctType::SOCIAL, 0.4f, 0.4f);
        instinct_weights[InstinctType::CONSISTENCY] = InstinctWeight(InstinctType::CONSISTENCY, 0.7f, 0.7f);
    }
    
    InstinctBias get_instinct_bias(const ContextState& context) {
        InstinctBias bias;
        
        // Calculate influences for all instincts
        std::map<InstinctType, float> influences;
        
        for (const auto& pair : instinct_weights) {
            InstinctType instinct = pair.first;
            float base_weight = pair.second.weight;
            float influence = base_weight;
            
            // Apply context-specific multipliers
            switch (instinct) {
                case InstinctType::SURVIVAL:
                    if (context.memory_risk) influence *= 1.5f;
                    if (context.resource_load > 0.7f) influence *= 1.3f;
                    break;
                    
                case InstinctType::CURIOSITY:
                    if (context.confidence_level < 0.4f) influence *= 1.4f;
                    if (context.novelty_level > 0.8f) influence *= 1.3f;
                    if (context.resource_load > 0.7f) influence *= 0.7f;
                    break;
                    
                case InstinctType::EFFICIENCY:
                    if (context.resource_load > 0.7f) influence *= 1.5f;
                    break;
                    
                case InstinctType::SOCIAL:
                    if (context.user_interaction) influence *= 1.6f;
                    break;
                    
                case InstinctType::CONSISTENCY:
                    if (context.has_contradictions) influence *= 1.4f;
                    if (context.confidence_level > 0.7f) influence *= 1.2f;
                    break;
            }
            
            influences[instinct] = std::max(0.0f, std::min(1.0f, influence));
        }
        
        // Calculate weighted contributions to recall vs exploration
        float recall_contribution = 0.0f;
        float exploration_contribution = 0.0f;
        float total_weight = 0.0f;
        
        for (const auto& influence : influences) {
            InstinctType instinct = influence.first;
            float strength = influence.second;
            
            bias.instinct_contributions[instinct] = strength;
            total_weight += strength;
            
            // Instincts that favor recall track
            if (instinct == InstinctType::CONSISTENCY || instinct == InstinctType::EFFICIENCY) {
                recall_contribution += strength;
            }
            // Instincts that favor exploration track
            else if (instinct == InstinctType::CURIOSITY) {
                exploration_contribution += strength;
            }
            // Instincts that can go either way based on context
            else if (instinct == InstinctType::SURVIVAL) {
                if (context.memory_risk || context.resource_load > 0.7f) {
                    recall_contribution += strength; // Conservative approach
                } else {
                    exploration_contribution += strength; // Safe to explore
                }
            }
            else if (instinct == InstinctType::SOCIAL) {
                if (context.user_interaction) {
                    exploration_contribution += strength; // Engage with user
                } else {
                    recall_contribution += strength; // Use known patterns
                }
            }
        }
        
        // Normalize contributions
        if (total_weight > 0.0f) {
            recall_contribution /= total_weight;
            exploration_contribution /= total_weight;
        }
        
        // Apply softmax normalization for final weights
        float softmax_sum = std::exp(recall_contribution) + std::exp(exploration_contribution);
        bias.recall_weight = std::exp(recall_contribution) / softmax_sum;
        bias.exploration_weight = std::exp(exploration_contribution) / softmax_sum;
        
        // Generate reasoning explanation
        std::stringstream reasoning;
        reasoning << "Instinct Analysis: ";
        
        if (context.confidence_level < 0.4f) {
            reasoning << "Low confidence triggers Curiosity (" << std::fixed << std::setprecision(2) 
                     << influences[InstinctType::CURIOSITY] << "), ";
        }
        if (context.resource_load > 0.7f) {
            reasoning << "High resource load triggers Efficiency (" << std::fixed << std::setprecision(2) 
                     << influences[InstinctType::EFFICIENCY] << "), ";
        }
        if (context.has_contradictions) {
            reasoning << "Contradictions trigger Consistency (" << std::fixed << std::setprecision(2) 
                     << influences[InstinctType::CONSISTENCY] << "), ";
        }
        if (context.user_interaction) {
            reasoning << "User interaction triggers Social (" << std::fixed << std::setprecision(2) 
                     << influences[InstinctType::SOCIAL] << "), ";
        }
        if (context.memory_risk) {
            reasoning << "Memory risk triggers Survival (" << std::fixed << std::setprecision(2) 
                     << influences[InstinctType::SURVIVAL] << "), ";
        }
        
        reasoning << "Final bias: Recall=" << std::fixed << std::setprecision(2) << bias.recall_weight 
                 << ", Exploration=" << bias.exploration_weight;
        
        bias.reasoning = reasoning.str();
        
        return bias;
    }
    
    std::string instinct_type_to_string(InstinctType instinct) {
        switch (instinct) {
            case InstinctType::SURVIVAL: return "Survival";
            case InstinctType::CURIOSITY: return "Curiosity";
            case InstinctType::EFFICIENCY: return "Efficiency";
            case InstinctType::SOCIAL: return "Social";
            case InstinctType::CONSISTENCY: return "Consistency";
            default: return "Unknown";
        }
    }
};

// ============================================================================
// DEMONSTRATION FUNCTIONS
// ============================================================================

void demonstrate_low_confidence_scenario() {
    std::cout << "\n📊 Test 1: Low Confidence Scenario (Curiosity vs Efficiency)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    
    InstinctEngine engine;
    ContextState context;
    context.confidence_level = 0.25f;  // Low confidence
    context.resource_load = 0.4f;      // Moderate resource load
    context.user_interaction = true;   // User interaction
    context.novelty_level = 0.8f;     // High novelty
    
    InstinctBias bias = engine.get_instinct_bias(context);
    
    std::cout << "Context:" << std::endl;
    std::cout << "- Confidence Level: " << std::fixed << std::setprecision(2) << context.confidence_level << std::endl;
    std::cout << "- Resource Load: " << context.resource_load << std::endl;
    std::cout << "- Novelty Level: " << context.novelty_level << std::endl;
    std::cout << "- User Interaction: " << (context.user_interaction ? "Yes" : "No") << std::endl;
    
    std::cout << "\nInstinct Contributions:" << std::endl;
    for (const auto& contribution : bias.instinct_contributions) {
        std::cout << "- " << engine.instinct_type_to_string(contribution.first) 
                  << ": " << std::fixed << std::setprecision(3) << contribution.second << std::endl;
    }
    
    std::cout << "\nFinal Decision Bias:" << std::endl;
    std::cout << "- Recall Track: " << std::fixed << std::setprecision(1) 
              << (bias.recall_weight * 100) << "%" << std::endl;
    std::cout << "- Exploration Track: " << (bias.exploration_weight * 100) << "%" << std::endl;
    
    std::cout << "\nReasoning: " << bias.reasoning << std::endl;
}

void demonstrate_high_resource_load_scenario() {
    std::cout << "\n⚡ Test 2: High Resource Load Scenario" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    
    InstinctEngine engine;
    ContextState context;
    context.confidence_level = 0.6f;  // Moderate confidence
    context.resource_load = 0.9f;      // High resource load
    context.novelty_level = 0.3f;      // Low novelty
    
    InstinctBias bias = engine.get_instinct_bias(context);
    
    std::cout << "Context:" << std::endl;
    std::cout << "- Resource Load: " << std::fixed << std::setprecision(2) << context.resource_load << std::endl;
    std::cout << "- Confidence Level: " << context.confidence_level << std::endl;
    
    std::cout << "\nInstinct Contributions:" << std::endl;
    for (const auto& contribution : bias.instinct_contributions) {
        std::cout << "- " << engine.instinct_type_to_string(contribution.first) 
                  << ": " << std::fixed << std::setprecision(3) << contribution.second << std::endl;
    }
    
    std::cout << "\nFinal Decision Bias:" << std::endl;
    std::cout << "- Recall Track: " << std::fixed << std::setprecision(1) 
              << (bias.recall_weight * 100) << "%" << std::endl;
    std::cout << "- Exploration Track: " << (bias.exploration_weight * 100) << "%" << std::endl;
    
    std::cout << "\nReasoning: " << bias.reasoning << std::endl;
}

void demonstrate_complex_scenario() {
    std::cout << "\n🧩 Test 3: Complex Multi-Factor Scenario" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    InstinctEngine engine;
    ContextState context;
    context.confidence_level = 0.35f;  // Low confidence (triggers Curiosity)
    context.resource_load = 0.75f;      // High resource load (triggers Efficiency)
    context.has_contradictions = true;   // Has contradictions (triggers Consistency)
    context.user_interaction = true;    // User interaction (triggers Social)
    context.novelty_level = 0.7f;       // High novelty (triggers Curiosity)
    
    InstinctBias bias = engine.get_instinct_bias(context);
    
    std::cout << "Complex Context:" << std::endl;
    std::cout << "- Confidence: " << std::fixed << std::setprecision(2) << context.confidence_level << std::endl;
    std::cout << "- Resource Load: " << context.resource_load << std::endl;
    std::cout << "- Contradictions: " << (context.has_contradictions ? "Yes" : "No") << std::endl;
    std::cout << "- User Interaction: " << (context.user_interaction ? "Yes" : "No") << std::endl;
    std::cout << "- Novelty: " << context.novelty_level << std::endl;
    
    std::cout << "\nInstinct Contributions (Competing Drives):" << std::endl;
    for (const auto& contribution : bias.instinct_contributions) {
        std::cout << "- " << engine.instinct_type_to_string(contribution.first) 
                  << ": " << std::fixed << std::setprecision(3) << contribution.second << std::endl;
    }
    
    std::cout << "\nConflict Resolution:" << std::endl;
    std::cout << "- Recall Track: " << std::fixed << std::setprecision(1) 
              << (bias.recall_weight * 100) << "%" << std::endl;
    std::cout << "- Exploration Track: " << (bias.exploration_weight * 100) << "%" << std::endl;
    
    std::cout << "\nReasoning: " << bias.reasoning << std::endl;
}

int main() {
    std::cout << "🧠 Melvin Instinct Engine - Unified Brain DNA" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "This demonstration shows how Melvin's five core instincts" << std::endl;
    std::cout << "(Survival, Curiosity, Efficiency, Social, Consistency)" << std::endl;
    std::cout << "compete and collaborate to bias reasoning, learning, and tool use." << std::endl;
    
    // Run demonstrations
    demonstrate_low_confidence_scenario();
    demonstrate_high_resource_load_scenario();
    demonstrate_complex_scenario();
    
    std::cout << "\n🎯 Key Insights:" << std::endl;
    std::cout << "- Instincts dynamically adjust based on context" << std::endl;
    std::cout << "- Low confidence → Curiosity dominance" << std::endl;
    std::cout << "- High resource load → Efficiency dominance" << std::endl;
    std::cout << "- Contradictions → Consistency dominance" << std::endl;
    std::cout << "- User interaction → Social dominance" << std::endl;
    std::cout << "- Softmax normalization resolves instinct conflicts" << std::endl;
    
    std::cout << "\n🚀 The Instinct Engine is ready for integration with Melvin's unified brain!" << std::endl;
    
    return 0;
}
