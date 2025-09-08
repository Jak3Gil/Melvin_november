#include "melvin_instinct_engine.h"
#include <sstream>
#include <iomanip>
#include <random>

// ============================================================================
// INSTINCT ENGINE IMPLEMENTATION
// ============================================================================

InstinctEngine::InstinctEngine() {
    initialize_instincts();
}

InstinctEngine::~InstinctEngine() {
    // Cleanup if needed
}

void InstinctEngine::initialize_instincts() {
    std::lock_guard<std::mutex> lock(instinct_mutex);
    
    // Initialize with default weights as specified
    instinct_weights[InstinctType::SURVIVAL] = InstinctWeight(InstinctType::SURVIVAL, 0.8f, 0.8f);
    instinct_weights[InstinctType::CURIOSITY] = InstinctWeight(InstinctType::CURIOSITY, 0.6f, 0.6f);
    instinct_weights[InstinctType::EFFICIENCY] = InstinctWeight(InstinctType::EFFICIENCY, 0.5f, 0.5f);
    instinct_weights[InstinctType::SOCIAL] = InstinctWeight(InstinctType::SOCIAL, 0.4f, 0.4f);
    instinct_weights[InstinctType::CONSISTENCY] = InstinctWeight(InstinctType::CONSISTENCY, 0.7f, 0.7f);
    
    // Initialize activation timestamps
    for (auto& pair : instinct_weights) {
        activation_timestamps[pair.first] = std::vector<double>();
    }
}

InstinctBias InstinctEngine::get_instinct_bias(const ContextState& context) {
    std::lock_guard<std::mutex> lock(instinct_mutex);
    
    // Calculate influences for all instincts
    auto influences = calculate_all_instinct_influences(context);
    
    // Resolve conflicts and generate bias
    InstinctBias bias = resolve_instinct_conflicts(influences, context);
    
    // Update activation counts and timestamps
    double current_time = context.timestamp;
    for (const auto& influence : influences) {
        if (influence.second > 0.1f) { // Only count significant activations
            instinct_weights[influence.first].activation_count++;
            instinct_weights[influence.first].last_activation_time = current_time;
            activation_timestamps[influence.first].push_back(current_time);
            
            // Clean old timestamps
            auto& timestamps = activation_timestamps[influence.first];
            timestamps.erase(std::remove_if(timestamps.begin(), timestamps.end(),
                [current_time](double ts) { return current_time - ts > INSTINCT_DECAY_TIME; }),
                timestamps.end());
        }
    }
    
    return bias;
}

void InstinctEngine::reinforce_instinct(InstinctType instinct, float delta, const std::string& reason, uint64_t node_id) {
    std::lock_guard<std::mutex> instinct_lock(instinct_mutex);
    std::lock_guard<std::mutex> reinforcement_lock(reinforcement_mutex);
    
    // Record reinforcement signal
    double current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    
    ReinforcementSignal signal(instinct, delta, reason, current_time, node_id);
    reinforcement_history.push_back(signal);
    
    // Clean old reinforcement history
    if (reinforcement_history.size() > MAX_REINFORCEMENT_HISTORY) {
        reinforcement_history.erase(reinforcement_history.begin(), 
            reinforcement_history.begin() + (reinforcement_history.size() - MAX_REINFORCEMENT_HISTORY));
    }
    
    // Apply reinforcement to instinct weight
    if (instinct_weights.find(instinct) != instinct_weights.end()) {
        InstinctWeight& weight = instinct_weights[instinct];
        
        // Apply reinforcement with decay
        weight.reinforcement += delta * REINFORCEMENT_STRENGTH;
        weight.reinforcement *= REINFORCEMENT_DECAY;
        
        // Update weight based on reinforcement
        float new_weight = weight.base_weight + weight.reinforcement;
        new_weight = std::max(MIN_INSTINCT_WEIGHT, std::min(MAX_INSTINCT_WEIGHT, new_weight));
        weight.weight = new_weight;
    }
}

float InstinctEngine::calculate_instinct_influence(InstinctType instinct, const ContextState& context) {
    if (instinct_weights.find(instinct) == instinct_weights.end()) {
        return 0.0f;
    }
    
    const InstinctWeight& weight = instinct_weights.at(instinct);
    float base_influence = weight.weight;
    
    // Apply context-specific multipliers
    switch (instinct) {
        case InstinctType::SURVIVAL:
            // Increase when memory risk is high
            if (context.memory_risk) {
                base_influence *= 1.5f;
            }
            // Increase when resource load is high
            if (context.resource_load > RESOURCE_LOAD_THRESHOLD) {
                base_influence *= 1.3f;
            }
            break;
            
        case InstinctType::CURIOSITY:
            // Increase when confidence is low (need to explore)
            if (context.confidence_level < CONFIDENCE_THRESHOLD_LOW) {
                base_influence *= 1.4f;
            }
            // Increase when novelty is high
            if (context.novelty_level > NOVELTY_THRESHOLD_HIGH) {
                base_influence *= 1.3f;
            }
            // Decrease when resource load is high (efficiency takes priority)
            if (context.resource_load > RESOURCE_LOAD_THRESHOLD) {
                base_influence *= 0.7f;
            }
            break;
            
        case InstinctType::EFFICIENCY:
            // Increase when resource load is high
            if (context.resource_load > RESOURCE_LOAD_THRESHOLD) {
                base_influence *= 1.5f;
            }
            // Increase when input complexity is high
            if (context.input_complexity > 1000) {
                base_influence *= 1.2f;
            }
            break;
            
        case InstinctType::SOCIAL:
            // Increase when user interaction is present
            if (context.user_interaction) {
                base_influence *= 1.6f;
            }
            break;
            
        case InstinctType::CONSISTENCY:
            // Increase when contradictions are detected
            if (context.has_contradictions) {
                base_influence *= 1.4f;
            }
            // Increase when confidence is high (maintain consistency)
            if (context.confidence_level > CONFIDENCE_THRESHOLD_HIGH) {
                base_influence *= 1.2f;
            }
            break;
    }
    
    // Apply temporal decay
    float decay_factor = calculate_instinct_decay(instinct, context.timestamp);
    base_influence *= decay_factor;
    
    return std::max(0.0f, std::min(1.0f, base_influence));
}

std::map<InstinctType, float> InstinctEngine::calculate_all_instinct_influences(const ContextState& context) {
    std::map<InstinctType, float> influences;
    
    for (const auto& pair : instinct_weights) {
        influences[pair.first] = calculate_instinct_influence(pair.first, context);
    }
    
    return influences;
}

InstinctBias InstinctEngine::resolve_instinct_conflicts(const std::map<InstinctType, float>& influences, const ContextState& context) {
    InstinctBias bias;
    
    // Calculate weighted contributions to recall vs exploration
    float recall_contribution = 0.0f;
    float exploration_contribution = 0.0f;
    float total_weight = 0.0f;
    
    // Map instincts to their preferred tracks
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
            if (context.memory_risk || context.resource_load > RESOURCE_LOAD_THRESHOLD) {
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
    std::vector<float> weights = {recall_contribution, exploration_contribution};
    float softmax_sum = 0.0f;
    for (float w : weights) {
        softmax_sum += std::exp(w);
    }
    
    bias.recall_weight = std::exp(recall_contribution) / softmax_sum;
    bias.exploration_weight = std::exp(exploration_contribution) / softmax_sum;
    bias.overall_strength = total_weight;
    
    // Generate reasoning explanation
    std::stringstream reasoning;
    reasoning << "Instinct Analysis: ";
    
    if (context.confidence_level < CONFIDENCE_THRESHOLD_LOW) {
        reasoning << "Low confidence triggers Curiosity (" << influences.at(InstinctType::CURIOSITY) << "), ";
    }
    if (context.resource_load > RESOURCE_LOAD_THRESHOLD) {
        reasoning << "High resource load triggers Efficiency (" << influences.at(InstinctType::EFFICIENCY) << "), ";
    }
    if (context.has_contradictions) {
        reasoning << "Contradictions trigger Consistency (" << influences.at(InstinctType::CONSISTENCY) << "), ";
    }
    if (context.user_interaction) {
        reasoning << "User interaction triggers Social (" << influences.at(InstinctType::SOCIAL) << "), ";
    }
    if (context.memory_risk) {
        reasoning << "Memory risk triggers Survival (" << influences.at(InstinctType::SURVIVAL) << "), ";
    }
    
    reasoning << "Final bias: Recall=" << std::fixed << std::setprecision(2) << bias.recall_weight 
             << ", Exploration=" << bias.exploration_weight;
    
    bias.reasoning = reasoning.str();
    
    return bias;
}

float InstinctEngine::apply_softmax_normalization(const std::vector<float>& values) {
    if (values.empty()) return 0.0f;
    
    float max_val = *std::max_element(values.begin(), values.end());
    float sum = 0.0f;
    
    for (float val : values) {
        sum += std::exp(val - max_val);
    }
    
    return std::exp(values[0] - max_val) / sum;
}

ContextState InstinctEngine::analyze_context(float confidence, float resource_load, bool contradictions, 
                                            bool user_interaction, bool memory_risk, float novelty, uint64_t complexity) {
    ContextState context;
    context.confidence_level = confidence;
    context.resource_load = resource_load;
    context.has_contradictions = contradictions;
    context.user_interaction = user_interaction;
    context.memory_risk = memory_risk;
    context.novelty_level = novelty;
    context.input_complexity = complexity;
    context.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    
    return context;
}

void InstinctEngine::update_instinct_weight(InstinctType instinct, float new_weight) {
    std::lock_guard<std::mutex> lock(instinct_mutex);
    
    if (instinct_weights.find(instinct) != instinct_weights.end()) {
        instinct_weights[instinct].weight = std::max(MIN_INSTINCT_WEIGHT, 
                                                   std::min(MAX_INSTINCT_WEIGHT, new_weight));
    }
}

void InstinctEngine::apply_reinforcement_decay() {
    std::lock_guard<std::mutex> lock(instinct_mutex);
    
    for (auto& pair : instinct_weights) {
        pair.second.reinforcement *= REINFORCEMENT_DECAY;
    }
}

void InstinctEngine::normalize_instinct_weights() {
    std::lock_guard<std::mutex> lock(instinct_mutex);
    
    float total_weight = 0.0f;
    for (const auto& pair : instinct_weights) {
        total_weight += pair.second.weight;
    }
    
    if (total_weight > 0.0f) {
        for (auto& pair : instinct_weights) {
            pair.second.weight /= total_weight;
        }
    }
}

std::vector<InstinctTag> InstinctEngine::generate_instinct_tags(const InstinctBias& bias, const std::string& context) {
    std::vector<InstinctTag> tags;
    double current_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    
    for (const auto& contribution : bias.instinct_contributions) {
        if (contribution.second > 0.1f) { // Only tag significant contributions
            InstinctTag tag(contribution.first, contribution.second, current_time, context);
            tags.push_back(tag);
        }
    }
    
    return tags;
}

std::string InstinctEngine::format_instinct_tags(const std::vector<InstinctTag>& tags) {
    if (tags.empty()) return "No instinct tags";
    
    std::stringstream ss;
    ss << "Instinct Tags: ";
    for (size_t i = 0; i < tags.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << "[" << instinct_type_to_string(tags[i].instinct) 
           << "+" << std::fixed << std::setprecision(2) << tags[i].strength << "]";
    }
    return ss.str();
}

std::string InstinctEngine::instinct_type_to_string(InstinctType instinct) {
    switch (instinct) {
        case InstinctType::SURVIVAL: return "Survival";
        case InstinctType::CURIOSITY: return "Curiosity";
        case InstinctType::EFFICIENCY: return "Efficiency";
        case InstinctType::SOCIAL: return "Social";
        case InstinctType::CONSISTENCY: return "Consistency";
        default: return "Unknown";
    }
}

InstinctType InstinctEngine::string_to_instinct_type(const std::string& str) {
    if (str == "Survival") return InstinctType::SURVIVAL;
    if (str == "Curiosity") return InstinctType::CURIOSITY;
    if (str == "Efficiency") return InstinctType::EFFICIENCY;
    if (str == "Social") return InstinctType::SOCIAL;
    if (str == "Consistency") return InstinctType::CONSISTENCY;
    return InstinctType::SURVIVAL; // Default
}

float InstinctEngine::calculate_instinct_decay(InstinctType instinct, double current_time) {
    if (activation_timestamps.find(instinct) == activation_timestamps.end()) {
        return 1.0f;
    }
    
    const auto& timestamps = activation_timestamps[instinct];
    if (timestamps.empty()) {
        return 1.0f;
    }
    
    // Calculate decay based on time since last activation
    double last_activation = timestamps.back();
    double time_since_activation = current_time - last_activation;
    
    // Exponential decay with half-life of 1 hour
    float decay_factor = std::exp(-time_since_activation / INSTINCT_DECAY_TIME);
    return std::max(0.1f, decay_factor);
}

InstinctEngine::InstinctStats InstinctEngine::get_instinct_statistics() {
    std::lock_guard<std::mutex> instinct_lock(instinct_mutex);
    std::lock_guard<std::mutex> reinforcement_lock(reinforcement_mutex);
    
    InstinctStats stats;
    
    for (const auto& pair : instinct_weights) {
        stats.current_weights[pair.first] = pair.second.weight;
        stats.activation_counts[pair.first] = pair.second.activation_count;
        stats.average_reinforcement[pair.first] = pair.second.reinforcement;
    }
    
    stats.total_reinforcements = reinforcement_history.size();
    if (!reinforcement_history.empty()) {
        stats.last_reinforcement_time = reinforcement_history.back().timestamp;
    }
    
    return stats;
}

std::string InstinctEngine::format_instinct_statistics(const InstinctStats& stats) {
    std::stringstream ss;
    ss << "=== Instinct Engine Statistics ===\n";
    
    for (const auto& weight : stats.current_weights) {
        ss << instinct_type_to_string(weight.first) << ": " 
           << std::fixed << std::setprecision(3) << weight.second
           << " (activations: " << stats.activation_counts.at(weight.first)
           << ", reinforcement: " << stats.average_reinforcement.at(weight.first) << ")\n";
    }
    
    ss << "Total reinforcements: " << stats.total_reinforcements << "\n";
    ss << "Last reinforcement: " << std::fixed << std::setprecision(1) 
       << stats.last_reinforcement_time << "s ago\n";
    
    return ss.str();
}

void InstinctEngine::demonstrate_instinct_conflict(const std::string& scenario_description) {
    std::cout << "\n=== Instinct Conflict Demonstration ===" << std::endl;
    std::cout << "Scenario: " << scenario_description << std::endl;
    
    // Create a low-confidence, high-resource-load scenario
    ContextState context = analyze_context(0.3f, 0.8f, false, true, false, 0.7f, 1500);
    
    InstinctBias bias = get_instinct_bias(context);
    
    std::cout << "Context Analysis:" << std::endl;
    std::cout << "- Confidence: " << context.confidence_level << std::endl;
    std::cout << "- Resource Load: " << context.resource_load << std::endl;
    std::cout << "- User Interaction: " << (context.user_interaction ? "Yes" : "No") << std::endl;
    std::cout << "- Novelty: " << context.novelty_level << std::endl;
    
    std::cout << "\nInstinct Contributions:" << std::endl;
    for (const auto& contribution : bias.instinct_contributions) {
        std::cout << "- " << instinct_type_to_string(contribution.first) 
                  << ": " << std::fixed << std::setprecision(3) << contribution.second << std::endl;
    }
    
    std::cout << "\nFinal Bias:" << std::endl;
    std::cout << "- Recall Weight: " << std::fixed << std::setprecision(3) << bias.recall_weight << std::endl;
    std::cout << "- Exploration Weight: " << bias.exploration_weight << std::endl;
    std::cout << "- Reasoning: " << bias.reasoning << std::endl;
    
    std::cout << "\n=== End Demonstration ===" << std::endl;
}

InstinctBias InstinctEngine::simulate_decision_scenario(const ContextState& context, const std::string& scenario) {
    std::cout << "\n=== Decision Scenario Simulation ===" << std::endl;
    std::cout << "Scenario: " << scenario << std::endl;
    
    InstinctBias bias = get_instinct_bias(context);
    
    std::cout << "Instinct Analysis:" << std::endl;
    std::cout << bias.reasoning << std::endl;
    
    std::cout << "Decision Bias:" << std::endl;
    std::cout << "- Recall Track: " << std::fixed << std::setprecision(1) 
              << (bias.recall_weight * 100) << "%" << std::endl;
    std::cout << "- Exploration Track: " << (bias.exploration_weight * 100) << "%" << std::endl;
    
    return bias;
}

// ============================================================================
// INSTINCT INTEGRATION HELPER IMPLEMENTATION
// ============================================================================

InstinctBias InstinctIntegrationHelper::influence_blended_reasoning(const InstinctBias& instinct_bias, 
                                                                   float base_recall_weight, float base_exploration_weight) {
    InstinctBias influenced_bias = instinct_bias;
    
    // Blend instinct bias with base weights
    influenced_bias.recall_weight = (base_recall_weight * 0.7f) + (instinct_bias.recall_weight * 0.3f);
    influenced_bias.exploration_weight = (base_exploration_weight * 0.7f) + (instinct_bias.exploration_weight * 0.3f);
    
    // Normalize to ensure they sum to 1.0
    float total = influenced_bias.recall_weight + influenced_bias.exploration_weight;
    if (total > 0.0f) {
        influenced_bias.recall_weight /= total;
        influenced_bias.exploration_weight /= total;
    }
    
    return influenced_bias;
}

std::vector<InstinctTag> InstinctIntegrationHelper::extract_instinct_tags_from_node(uint64_t node_id) {
    // This would integrate with Melvin's memory system
    // For now, return empty vector as placeholder
    return std::vector<InstinctTag>();
}

void InstinctIntegrationHelper::attach_instinct_tags_to_node(uint64_t node_id, const std::vector<InstinctTag>& tags) {
    // This would integrate with Melvin's memory system
    // For now, just a placeholder
}

ContextState InstinctIntegrationHelper::build_context_from_confidence(float confidence) {
    ContextState context;
    context.confidence_level = confidence;
    context.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    return context;
}

ContextState InstinctIntegrationHelper::build_context_from_resource_load(float resource_load) {
    ContextState context;
    context.resource_load = resource_load;
    context.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    return context;
}

ContextState InstinctIntegrationHelper::build_context_from_contradictions(bool has_contradictions) {
    ContextState context;
    context.has_contradictions = has_contradictions;
    context.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    return context;
}

ContextState InstinctIntegrationHelper::build_context_from_user_interaction(bool user_interaction) {
    ContextState context;
    context.user_interaction = user_interaction;
    context.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    return context;
}

ContextState InstinctIntegrationHelper::build_context_from_memory_risk(bool memory_risk) {
    ContextState context;
    context.memory_risk = memory_risk;
    context.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
    return context;
}

std::string InstinctIntegrationHelper::format_instinct_bias(const InstinctBias& bias) {
    std::stringstream ss;
    ss << "Instinct Bias Analysis:\n";
    ss << "- Recall Weight: " << std::fixed << std::setprecision(3) << bias.recall_weight << "\n";
    ss << "- Exploration Weight: " << bias.exploration_weight << "\n";
    ss << "- Overall Strength: " << bias.overall_strength << "\n";
    ss << "- Reasoning: " << bias.reasoning << "\n";
    
    ss << "Instinct Contributions:\n";
    for (const auto& contribution : bias.instinct_contributions) {
        ss << "- " << InstinctEngine().instinct_type_to_string(contribution.first) 
           << ": " << std::fixed << std::setprecision(3) << contribution.second << "\n";
    }
    
    return ss.str();
}

std::string InstinctIntegrationHelper::format_context_state(const ContextState& context) {
    std::stringstream ss;
    ss << "Context State:\n";
    ss << "- Confidence: " << std::fixed << std::setprecision(3) << context.confidence_level << "\n";
    ss << "- Resource Load: " << context.resource_load << "\n";
    ss << "- Contradictions: " << (context.has_contradictions ? "Yes" : "No") << "\n";
    ss << "- User Interaction: " << (context.user_interaction ? "Yes" : "No") << "\n";
    ss << "- Memory Risk: " << (context.memory_risk ? "Yes" : "No") << "\n";
    ss << "- Novelty: " << context.novelty_level << "\n";
    ss << "- Complexity: " << context.input_complexity << "\n";
    ss << "- Timestamp: " << std::fixed << std::setprecision(1) << context.timestamp << "\n";
    
    return ss.str();
}
