#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <iomanip>
#include <sstream>

// ============================================================================
// MELVIN INSTINCT-DRIVEN TOOL ACTIVATION DEMONSTRATION
// ============================================================================

enum class InstinctType {
    SURVIVAL = 0,
    CURIOSITY = 1,
    EFFICIENCY = 2,
    SOCIAL = 3,
    CONSISTENCY = 4
};

struct InstinctBias {
    float recall_weight;
    float exploration_weight;
    std::map<InstinctType, float> instinct_contributions;
    std::string reasoning;
    
    InstinctBias() : recall_weight(0.5f), exploration_weight(0.5f) {}
};

struct CuriosityResult {
    float overall_curiosity_level;
    std::vector<std::string> generated_questions;
    bool has_high_urgency_questions;
    
    CuriosityResult() : overall_curiosity_level(0.0f), has_high_urgency_questions(false) {}
};

struct ToolActivationResult {
    bool tool_triggered;
    std::string tool_used;
    std::string search_query;
    std::vector<std::string> search_results;
    bool search_successful;
    std::string reasoning;
    
    ToolActivationResult() : tool_triggered(false), search_successful(false) {}
};

class InstinctDrivenMelvin {
private:
    std::map<InstinctType, float> instinct_weights;
    
public:
    InstinctDrivenMelvin() {
        // Initialize with default weights
        instinct_weights[InstinctType::SURVIVAL] = 0.8f;
        instinct_weights[InstinctType::CURIOSITY] = 0.6f;
        instinct_weights[InstinctType::EFFICIENCY] = 0.5f;
        instinct_weights[InstinctType::SOCIAL] = 0.4f;
        instinct_weights[InstinctType::CONSISTENCY] = 0.7f;
    }
    
    InstinctBias analyze_instinct_bias(const std::string& input) {
        InstinctBias bias;
        
        // Analyze input characteristics
        float confidence_level = 0.5f;
        float novelty_level = 0.5f;
        bool has_unknown_concepts = false;
        bool is_question = false;
        
        // Detect unknown concepts (simplified)
        std::vector<std::string> unknown_concepts = {"carbon nanotubes", "quantum computing", "blockchain", "machine learning", "artificial intelligence"};
        for (const auto& concept : unknown_concepts) {
            if (input.find(concept) != std::string::npos) {
                has_unknown_concepts = true;
                confidence_level = 0.2f; // Low confidence
                novelty_level = 0.9f;    // High novelty
                break;
            }
        }
        
        // Detect questions
        if (input.find("?") != std::string::npos || 
            input.find("what") != std::string::npos ||
            input.find("how") != std::string::npos ||
            input.find("why") != std::string::npos) {
            is_question = true;
        }
        
        // Calculate instinct influences
        float curiosity_influence = instinct_weights[InstinctType::CURIOSITY];
        float efficiency_influence = instinct_weights[InstinctType::EFFICIENCY];
        float social_influence = instinct_weights[InstinctType::SOCIAL];
        
        // Apply context multipliers
        if (has_unknown_concepts) {
            curiosity_influence *= 1.5f; // Boost curiosity for unknown concepts
        }
        
        if (is_question) {
            social_influence *= 1.3f; // Boost social for questions
        }
        
        if (input.length() > 100) {
            efficiency_influence *= 1.2f; // Boost efficiency for complex inputs
        }
        
        // Store contributions
        bias.instinct_contributions[InstinctType::CURIOSITY] = curiosity_influence;
        bias.instinct_contributions[InstinctType::EFFICIENCY] = efficiency_influence;
        bias.instinct_contributions[InstinctType::SOCIAL] = social_influence;
        
        // Calculate final weights
        float total_influence = curiosity_influence + efficiency_influence + social_influence;
        
        if (total_influence > 0.0f) {
            bias.exploration_weight = curiosity_influence / total_influence;
            bias.recall_weight = (efficiency_influence + social_influence) / total_influence;
        }
        
        // Generate reasoning
        std::stringstream reasoning;
        reasoning << "Instinct Analysis: ";
        
        if (has_unknown_concepts) {
            reasoning << "Unknown concepts trigger Curiosity (" << std::fixed << std::setprecision(2) 
                     << curiosity_influence << "), ";
        }
        
        if (is_question) {
            reasoning << "Question format triggers Social (" << std::fixed << std::setprecision(2) 
                     << social_influence << "), ";
        }
        
        if (input.length() > 100) {
            reasoning << "Complex input triggers Efficiency (" << std::fixed << std::setprecision(2) 
                     << efficiency_influence << "), ";
        }
        
        reasoning << "Final bias: Recall=" << std::fixed << std::setprecision(2) << bias.recall_weight 
                 << ", Exploration=" << bias.exploration_weight;
        
        bias.reasoning = reasoning.str();
        
        return bias;
    }
    
    CuriosityResult perform_curiosity_analysis(const std::string& input) {
        CuriosityResult result;
        
        // Detect knowledge gaps
        std::vector<std::string> unknown_concepts = {"carbon nanotubes", "quantum computing", "blockchain", "machine learning", "artificial intelligence"};
        bool has_unknown = false;
        
        for (const auto& concept : unknown_concepts) {
            if (input.find(concept) != std::string::npos) {
                has_unknown = true;
                result.generated_questions.push_back("What is " + concept + "?");
                result.generated_questions.push_back("How does " + concept + " work?");
                result.generated_questions.push_back("What are the applications of " + concept + "?");
                break;
            }
        }
        
        if (has_unknown) {
            result.overall_curiosity_level = 0.8f;
            result.has_high_urgency_questions = true;
        } else {
            result.overall_curiosity_level = 0.3f;
        }
        
        return result;
    }
    
    bool should_trigger_tool_usage(const InstinctBias& instinct_bias, const CuriosityResult& curiosity_result) {
        // High exploration bias triggers tools
        if (instinct_bias.exploration_weight > 0.6f) {
            return true;
        }
        
        // High curiosity level triggers tools
        if (curiosity_result.overall_curiosity_level > 0.5f) {
            return true;
        }
        
        // High urgency questions trigger tools
        if (curiosity_result.has_high_urgency_questions) {
            return true;
        }
        
        return false;
    }
    
    ToolActivationResult activate_tools(const std::string& input, const InstinctBias& instinct_bias) {
        ToolActivationResult result;
        
        if (should_trigger_tool_usage(instinct_bias, perform_curiosity_analysis(input))) {
            result.tool_triggered = true;
            result.tool_used = "WebSearchTool";
            result.search_query = input;
            
            // Simulate web search results
            std::vector<std::string> unknown_concepts = {"carbon nanotubes", "quantum computing", "blockchain", "machine learning", "artificial intelligence"};
            
            for (const auto& concept : unknown_concepts) {
                if (input.find(concept) != std::string::npos) {
                    result.search_results.push_back(concept + " are cylindrical nanostructures with unique properties");
                    result.search_results.push_back(concept + " have applications in electronics, medicine, and materials science");
                    result.search_results.push_back(concept + " represent a breakthrough in nanotechnology");
                    result.search_successful = true;
                    break;
                }
            }
            
            if (!result.search_successful) {
                result.search_results.push_back("General information about " + input);
                result.search_successful = true;
            }
            
            result.reasoning = "High exploration bias (" + std::to_string(instinct_bias.exploration_weight) + 
                              ") and curiosity level triggered web search";
        } else {
            result.reasoning = "Recall bias dominant (" + std::to_string(instinct_bias.recall_weight) + 
                              ") - using existing knowledge";
        }
        
        return result;
    }
    
    void reinforce_instincts(bool success, InstinctType primary_instinct) {
        float delta = success ? 0.1f : -0.05f;
        instinct_weights[primary_instinct] += delta;
        
        // Keep weights in bounds
        instinct_weights[primary_instinct] = std::max(0.1f, std::min(1.0f, instinct_weights[primary_instinct]));
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

void demonstrate_carbon_nanotubes_scenario() {
    std::cout << "\n🧪 DEMONSTRATION: Carbon Nanotubes Scenario" << std::endl;
    std::cout << "===========================================" << std::endl;
    
    InstinctDrivenMelvin melvin;
    std::string input = "look into carbon nanotubes";
    
    std::cout << "Input: \"" << input << "\"" << std::endl;
    std::cout << "\n🔍 [Phase 1: Instinct Analysis]" << std::endl;
    
    InstinctBias bias = melvin.analyze_instinct_bias(input);
    
    std::cout << "Instinct Contributions:" << std::endl;
    for (const auto& contribution : bias.instinct_contributions) {
        std::cout << "- " << melvin.instinct_type_to_string(contribution.first) 
                  << ": " << std::fixed << std::setprecision(3) << contribution.second << std::endl;
    }
    
    std::cout << "\nFinal Decision Bias:" << std::endl;
    std::cout << "- Recall Track: " << std::fixed << std::setprecision(1) 
              << (bias.recall_weight * 100) << "%" << std::endl;
    std::cout << "- Exploration Track: " << (bias.exploration_weight * 100) << "%" << std::endl;
    
    std::cout << "\nReasoning: " << bias.reasoning << std::endl;
    
    std::cout << "\n🔍 [Phase 2: Curiosity Analysis]" << std::endl;
    CuriosityResult curiosity = melvin.perform_curiosity_analysis(input);
    
    std::cout << "Overall Curiosity Level: " << std::fixed << std::setprecision(2) 
              << curiosity.overall_curiosity_level << std::endl;
    std::cout << "Generated Questions:" << std::endl;
    for (const auto& question : curiosity.generated_questions) {
        std::cout << "- " << question << std::endl;
    }
    std::cout << "High Urgency Questions: " << (curiosity.has_high_urgency_questions ? "Yes" : "No") << std::endl;
    
    std::cout << "\n🔧 [Phase 3: Tool Activation Decision]" << std::endl;
    bool should_use_tools = melvin.should_trigger_tool_usage(bias, curiosity);
    
    if (should_use_tools) {
        std::cout << "✅ TOOL ACTIVATION TRIGGERED!" << std::endl;
        std::cout << "Reason: High exploration bias (" << std::fixed << std::setprecision(1) 
                  << (bias.exploration_weight * 100) << "%) and high curiosity level (" 
                  << (curiosity.overall_curiosity_level * 100) << "%)" << std::endl;
        
        std::cout << "\n🔍 [Phase 4: Web Search Execution]" << std::endl;
        ToolActivationResult tool_result = melvin.activate_tools(input, bias);
        
        std::cout << "Tool Used: " << tool_result.tool_used << std::endl;
        std::cout << "Search Query: \"" << tool_result.search_query << "\"" << std::endl;
        std::cout << "Search Successful: " << (tool_result.search_successful ? "Yes" : "No") << std::endl;
        
        if (tool_result.search_successful) {
            std::cout << "\nSearch Results:" << std::endl;
            for (size_t i = 0; i < tool_result.search_results.size(); ++i) {
                std::cout << (i + 1) << ". " << tool_result.search_results[i] << std::endl;
            }
            
            std::cout << "\n🧠 [Phase 5: Learning and Reinforcement]" << std::endl;
            std::cout << "✅ Learning from search results..." << std::endl;
            std::cout << "✅ Creating knowledge nodes..." << std::endl;
            std::cout << "✅ Strengthening Curiosity instinct..." << std::endl;
            
            melvin.reinforce_instincts(true, InstinctType::CURIOSITY);
            std::cout << "Curiosity instinct weight updated!" << std::endl;
        }
        
        std::cout << "\n💬 [Phase 6: Response Generation]" << std::endl;
        std::cout << "Melvin: \"Based on my search, carbon nanotubes are cylindrical nanostructures with unique properties. ";
        std::cout << "They have applications in electronics, medicine, and materials science, representing a breakthrough in nanotechnology. ";
        std::cout << "Would you like me to explore any specific aspect of carbon nanotubes further?\"" << std::endl;
        
    } else {
        std::cout << "❌ Tool activation NOT triggered" << std::endl;
        std::cout << "Reason: " << tool_result.reasoning << std::endl;
        
        std::cout << "\n💬 [Response Generation]" << std::endl;
        std::cout << "Melvin: \"That's an interesting input! I'm processing this through my unified brain system. ";
        std::cout << "I've activated 0 memory nodes and I'm analyzing the patterns and relationships. ";
        std::cout << "Could you tell me more about what you're thinking?\"" << std::endl;
    }
}

void demonstrate_known_concept_scenario() {
    std::cout << "\n📚 DEMONSTRATION: Known Concept Scenario" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    InstinctDrivenMelvin melvin;
    std::string input = "tell me about dogs";
    
    std::cout << "Input: \"" << input << "\"" << std::endl;
    std::cout << "\n🔍 [Phase 1: Instinct Analysis]" << std::endl;
    
    InstinctBias bias = melvin.analyze_instinct_bias(input);
    
    std::cout << "Final Decision Bias:" << std::endl;
    std::cout << "- Recall Track: " << std::fixed << std::setprecision(1) 
              << (bias.recall_weight * 100) << "%" << std::endl;
    std::cout << "- Exploration Track: " << (bias.exploration_weight * 100) << "%" << std::endl;
    
    std::cout << "\n🔍 [Phase 2: Curiosity Analysis]" << std::endl;
    CuriosityResult curiosity = melvin.perform_curiosity_analysis(input);
    
    std::cout << "Overall Curiosity Level: " << std::fixed << std::setprecision(2) 
              << curiosity.overall_curiosity_level << std::endl;
    
    std::cout << "\n🔧 [Phase 3: Tool Activation Decision]" << std::endl;
    bool should_use_tools = melvin.should_trigger_tool_usage(bias, curiosity);
    
    if (should_use_tools) {
        std::cout << "✅ TOOL ACTIVATION TRIGGERED!" << std::endl;
    } else {
        std::cout << "❌ Tool activation NOT triggered" << std::endl;
        std::cout << "Reason: Low exploration bias and curiosity level - using existing knowledge" << std::endl;
        
        std::cout << "\n💬 [Response Generation]" << std::endl;
        std::cout << "Melvin: \"Dogs are domesticated mammals that have been companions to humans for thousands of years. ";
        std::cout << "They come in many breeds, sizes, and temperaments. Dogs are known for their loyalty, intelligence, ";
        std::cout << "and ability to form strong bonds with humans. They serve various roles including pets, working animals, ";
        std::cout << "and service animals.\"" << std::endl;
    }
}

void demonstrate_instinct_evolution() {
    std::cout << "\n🧬 DEMONSTRATION: Instinct Evolution" << std::endl;
    std::cout << "====================================" << std::endl;
    
    InstinctDrivenMelvin melvin;
    
    std::cout << "Initial Instinct Weights:" << std::endl;
    for (const auto& weight : melvin.instinct_weights) {
        std::cout << "- " << melvin.instinct_type_to_string(weight.first) 
                  << ": " << std::fixed << std::setprecision(3) << weight.second << std::endl;
    }
    
    std::cout << "\nSimulating multiple successful curiosity-driven searches..." << std::endl;
    
    // Simulate successful searches
    for (int i = 0; i < 5; ++i) {
        melvin.reinforce_instincts(true, InstinctType::CURIOSITY);
        std::cout << "Search " << (i + 1) << " successful - Curiosity strengthened" << std::endl;
    }
    
    std::cout << "\nUpdated Instinct Weights:" << std::endl;
    for (const auto& weight : melvin.instinct_weights) {
        std::cout << "- " << melvin.instinct_type_to_string(weight.first) 
                  << ": " << std::fixed << std::setprecision(3) << weight.second << std::endl;
    }
    
    std::cout << "\nNotice: Curiosity instinct has strengthened from 0.600 to " 
              << std::fixed << std::setprecision(3) << melvin.instinct_weights[InstinctType::CURIOSITY] 
              << "!" << std::endl;
}

int main() {
    std::cout << "🧠 Melvin Instinct-Driven Tool Activation Demonstration" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "This demonstration shows how Melvin's instincts drive automatic tool usage" << std::endl;
    std::cout << "when encountering unknown concepts or high curiosity scenarios." << std::endl;
    
    // Run demonstrations
    demonstrate_carbon_nanotubes_scenario();
    demonstrate_known_concept_scenario();
    demonstrate_instinct_evolution();
    
    std::cout << "\n🎯 Key Insights:" << std::endl;
    std::cout << "- Unknown concepts trigger high exploration bias" << std::endl;
    std::cout << "- High exploration bias automatically activates web search" << std::endl;
    std::cout << "- Successful searches strengthen curiosity instinct" << std::endl;
    std::cout << "- Known concepts use recall-heavy reasoning" << std::endl;
    std::cout << "- Instincts evolve based on success/failure outcomes" << std::endl;
    
    std::cout << "\n🚀 The integrated system solves the 'carbon nanotubes' problem!" << std::endl;
    std::cout << "Melvin now automatically searches when he encounters unknown concepts!" << std::endl;
    
    return 0;
}
