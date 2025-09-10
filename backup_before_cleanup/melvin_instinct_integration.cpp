// ============================================================================
// INSTINCT-DRIVEN COGNITIVE PROCESSING IMPLEMENTATION
// ============================================================================

#include "melvin_optimized_v2.h"

// Initialize instinct engine in constructor
CognitiveProcessor::CognitiveProcessor(std::unique_ptr<PureBinaryStorage>& storage) 
    : binary_storage(std::move(storage)) {
    // Initialize instinct engine
    instinct_engine = std::make_unique<InstinctEngine>();
    
    // Initialize other systems...
    initialize_moral_supernodes();
    initialize_response_templates();
    initialize_basic_tools();
    
    // Initialize IDs
    next_curiosity_node_id = 1;
    next_tool_node_id = 1;
    next_experience_node_id = 1;
    next_toolchain_id = 1;
    next_executed_curiosity_node_id = 1;
}

InstinctBias CognitiveProcessor::get_instinct_bias_for_input(const std::string& input, const std::vector<ActivationNode>& activations) {
    // Analyze context from input and activations
    float confidence_level = 0.5f; // Default
    float resource_load = 0.0f;     // Default
    bool has_contradictions = false;
    bool user_interaction = true;   // Assume user interaction for now
    bool memory_risk = false;
    float novelty_level = 0.5f;
    uint64_t input_complexity = input.length();
    
    // Calculate confidence based on activated nodes
    if (activations.empty()) {
        confidence_level = 0.2f; // Very low confidence
        novelty_level = 0.8f;    // High novelty
    } else if (activations.size() < 3) {
        confidence_level = 0.4f; // Low confidence
        novelty_level = 0.6f;    // Moderate novelty
    } else {
        confidence_level = 0.7f; // High confidence
        novelty_level = 0.3f;    // Low novelty
    }
    
    // Detect contradictions (simplified)
    if (input.find("but") != std::string::npos || 
        input.find("however") != std::string::npos ||
        input.find("contradict") != std::string::npos) {
        has_contradictions = true;
    }
    
    // Detect memory risk (simplified)
    if (input.length() > 1000 || activations.size() > 100) {
        memory_risk = true;
        resource_load = 0.8f;
    }
    
    // Create context state
    ContextState context = instinct_engine->analyze_context(
        confidence_level, resource_load, has_contradictions,
        user_interaction, memory_risk, novelty_level, input_complexity
    );
    
    // Get instinct bias
    return instinct_engine->get_instinct_bias(context);
}

bool CognitiveProcessor::should_trigger_tool_usage(const InstinctBias& instinct_bias, const CuriosityGapDetectionResult& curiosity_result) {
    // Check if exploration weight is high enough to trigger tools
    if (instinct_bias.exploration_weight > 0.6f) {
        return true;
    }
    
    // Check if curiosity level is high
    if (curiosity_result.overall_curiosity_level > 0.5f) {
        return true;
    }
    
    // Check if there are high-urgency curiosity questions
    for (const auto& question : curiosity_result.generated_questions) {
        if (question.urgency > 0.7f && question.requires_external_help) {
            return true;
        }
    }
    
    // Check if there are knowledge gaps that need external help
    for (const auto& gap : curiosity_result.detected_gaps) {
        if (gap.confidence_level > 0.6f && gap.gap_type == "missing_explanation") {
            return true;
        }
    }
    
    return false;
}

void CognitiveProcessor::reinforce_instincts_from_outcome(const std::string& input, bool success, const std::string& reason) {
    // Determine which instinct was primarily responsible
    InstinctType primary_instinct = InstinctType::CURIOSITY; // Default
    
    // Analyze input to determine primary instinct
    if (input.find("search") != std::string::npos || 
        input.find("find") != std::string::npos ||
        input.find("look") != std::string::npos) {
        primary_instinct = InstinctType::CURIOSITY;
    } else if (input.find("help") != std::string::npos ||
               input.find("explain") != std::string::npos) {
        primary_instinct = InstinctType::SOCIAL;
    } else if (input.find("optimize") != std::string::npos ||
               input.find("efficient") != std::string::npos) {
        primary_instinct = InstinctType::EFFICIENCY;
    } else if (input.find("consistent") != std::string::npos ||
               input.find("contradict") != std::string::npos) {
        primary_instinct = InstinctType::CONSISTENCY;
    } else if (input.find("safe") != std::string::npos ||
               input.find("risk") != std::string::npos) {
        primary_instinct = InstinctType::SURVIVAL;
    }
    
    // Apply reinforcement
    float delta = success ? 0.1f : -0.05f;
    instinct_engine->reinforce_instinct(primary_instinct, delta, reason);
}

// Modified process_input to include instinct-driven tool activation
ProcessingResult CognitiveProcessor::process_input_with_instincts(const std::string& user_input) {
    ProcessingResult result;
    
    // Phase 1: Parse to activations
    auto activations = parse_to_activations(user_input);
    for (const auto& activation : activations) {
        result.activated_nodes.push_back(activation.node_id);
    }
    
    // Phase 2: Get instinct bias
    InstinctBias instinct_bias = get_instinct_bias_for_input(user_input, activations);
    
    // Phase 3: Apply moral gravity
    result.moral_gravity = apply_moral_gravity(user_input, activations);
    
    // Phase 4: Apply context bias
    apply_context_bias(activations);
    
    // Phase 5: Connection traversal
    std::vector<ConnectionWalk> all_walks;
    for (const auto& activation : activations) {
        auto walks = traverse_connections(activation.node_id);
        all_walks.insert(all_walks.end(), walks.begin(), walks.end());
    }
    
    // Phase 6: Hypothesis synthesis
    result.clusters = synthesize_hypotheses(activations);
    
    // Phase 7: Generate candidates
    auto candidates = generate_candidates(result.clusters);
    
    // Phase 8: Perform curiosity & knowledge gap detection
    result.curiosity_gap_detection = perform_curiosity_gap_detection(user_input, activations, result.clusters);
    
    // Phase 9: INSTINCT-DRIVEN TOOL ACTIVATION
    bool should_use_tools = should_trigger_tool_usage(instinct_bias, result.curiosity_gap_detection);
    
    if (should_use_tools) {
        std::cout << "\n🧠 [Instinct Analysis] High exploration bias detected (" 
                  << std::fixed << std::setprecision(1) << (instinct_bias.exploration_weight * 100) 
                  << "%) - Triggering tool usage!" << std::endl;
        std::cout << "Reasoning: " << instinct_bias.reasoning << std::endl;
        
        // Perform dynamic tools evaluation with instinct bias
        result.dynamic_tools = perform_dynamic_tools_evaluation(user_input, activations, result.curiosity_gap_detection);
        
        // If web search is recommended, perform it
        if (result.dynamic_tools.tool_evaluation.needs_new_tool || 
            !result.dynamic_tools.tool_evaluation.recommended_tools.empty()) {
            
            std::cout << "🔍 [Tool Activation] Performing web search for: " << user_input << std::endl;
            
            // Perform web search
            auto search_result = perform_web_search(user_input, next_curiosity_node_id);
            
            if (search_result.search_successful && !search_result.results.empty()) {
                std::cout << "✅ [Search Success] Found " << search_result.results.size() 
                          << " results. Learning from search..." << std::endl;
                
                // Create knowledge nodes from search results
                auto new_nodes = create_knowledge_nodes_from_search_results(search_result.results, user_input);
                result.activated_nodes.insert(result.activated_nodes.end(), new_nodes.begin(), new_nodes.end());
                
                // Record successful search experience
                auto experience = record_search_experience(web_search_tool.tool_id, next_curiosity_node_id, 
                                                          user_input, search_result);
                result.dynamic_tools.new_experiences.push_back(experience);
                
                // Reinforce curiosity instinct
                reinforce_instincts_from_outcome(user_input, true, "Successful web search and learning");
                
                std::cout << "🧠 [Instinct Reinforcement] Curiosity instinct strengthened!" << std::endl;
            } else {
                std::cout << "❌ [Search Failed] " << search_result.error_message << std::endl;
                
                // Reinforce curiosity instinct negatively
                reinforce_instincts_from_outcome(user_input, false, "Failed web search");
            }
        }
    } else {
        std::cout << "\n🧠 [Instinct Analysis] Recall bias dominant (" 
                  << std::fixed << std::setprecision(1) << (instinct_bias.recall_weight * 100) 
                  << "%) - Using existing knowledge" << std::endl;
        
        // Perform standard dynamic tools evaluation
        result.dynamic_tools = perform_dynamic_tools_evaluation(user_input, activations, result.curiosity_gap_detection);
    }
    
    // Phase 10: Perform meta-tool engineering
    result.meta_tool_engineer = perform_meta_tool_engineering(user_input, activations, result.dynamic_tools);
    
    // Phase 11: Perform curiosity execution loop
    result.curiosity_execution = perform_curiosity_execution_loop(user_input, activations, 
                                                                 result.curiosity_gap_detection, 
                                                                 result.dynamic_tools, 
                                                                 result.meta_tool_engineer);
    
    // Phase 12: Perform temporal planning
    result.temporal_planning = perform_temporal_planning(user_input, activations, result.moral_gravity);
    
    // Phase 13: Perform temporal sequencing
    result.temporal_sequencing = perform_temporal_sequencing(activations, 
                                                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                                                std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0);
    
    // Phase 14: Perform blended reasoning with instinct bias
    result.blended_reasoning = perform_blended_reasoning(user_input, activations);
    
    // Modify blended reasoning weights based on instinct bias
    result.blended_reasoning.recall_weight = (result.blended_reasoning.recall_weight * 0.7f) + 
                                            (instinct_bias.recall_weight * 0.3f);
    result.blended_reasoning.exploration_weight = (result.blended_reasoning.exploration_weight * 0.7f) + 
                                                 (instinct_bias.exploration_weight * 0.3f);
    
    // Normalize weights
    float total_weight = result.blended_reasoning.recall_weight + result.blended_reasoning.exploration_weight;
    if (total_weight > 0.0f) {
        result.blended_reasoning.recall_weight /= total_weight;
        result.blended_reasoning.exploration_weight /= total_weight;
    }
    
    // Phase 15: Generate final response
    auto best_candidate = select_best_response(candidates);
    result.final_response = best_candidate.text;
    result.confidence = best_candidate.confidence;
    result.reasoning = best_candidate.reasoning;
    
    // Add instinct analysis to reasoning
    result.reasoning += "\n[Instinct Analysis] " + instinct_bias.reasoning;
    
    return result;
}
