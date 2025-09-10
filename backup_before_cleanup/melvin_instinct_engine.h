#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <mutex>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <random>

// ============================================================================
// INSTINCT ENGINE STRUCTURES
// ============================================================================

enum class InstinctType : uint8_t {
    SURVIVAL = 0,
    CURIOSITY = 1,
    EFFICIENCY = 2,
    SOCIAL = 3,
    CONSISTENCY = 4
};

struct InstinctWeight {
    InstinctType instinct;
    float weight;           // 0.0 - 1.0
    float base_weight;      // Default weight for this instinct
    float reinforcement;    // Accumulated reinforcement over time
    uint64_t activation_count;
    double last_activation_time;
    
    InstinctWeight() : instinct(InstinctType::SURVIVAL), weight(0.0f), base_weight(0.0f), 
                      reinforcement(0.0f), activation_count(0), last_activation_time(0.0) {}
    
    InstinctWeight(InstinctType inst, float w, float base_w) 
        : instinct(inst), weight(w), base_weight(base_w), reinforcement(0.0f), 
          activation_count(0), last_activation_time(0.0) {}
};

struct ContextState {
    float confidence_level;     // 0.0 - 1.0
    float resource_load;        // 0.0 - 1.0 (CPU/memory usage)
    bool has_contradictions;    // Detected contradictions in reasoning
    bool user_interaction;      // Direct user interaction
    bool memory_risk;          // Risk of memory corruption/overflow
    float novelty_level;       // 0.0 - 1.0 (how novel the input is)
    uint64_t input_complexity; // Complexity of current input
    double timestamp;          // Current timestamp
    
    ContextState() : confidence_level(0.5f), resource_load(0.0f), has_contradictions(false),
                    user_interaction(false), memory_risk(false), novelty_level(0.5f),
                    input_complexity(0), timestamp(0.0) {}
};

struct InstinctBias {
    float recall_weight;        // Bias toward recall track (0.0 - 1.0)
    float exploration_weight;   // Bias toward exploration track (0.0 - 1.0)
    std::map<InstinctType, float> instinct_contributions;
    std::string reasoning;      // Explanation of bias calculation
    float overall_strength;     // Overall instinct activation strength
    
    InstinctBias() : recall_weight(0.5f), exploration_weight(0.5f), overall_strength(0.0f) {}
};

struct InstinctTag {
    InstinctType instinct;
    float strength;             // How strongly this instinct influenced the decision
    double timestamp;          // When this tag was created
    std::string context;       // Context where instinct was activated
    
    InstinctTag() : instinct(InstinctType::SURVIVAL), strength(0.0f), timestamp(0.0) {}
    InstinctTag(InstinctType inst, float str, double time, const std::string& ctx)
        : instinct(inst), strength(str), timestamp(time), context(ctx) {}
};

struct ReinforcementSignal {
    InstinctType instinct;
    float delta;               // Positive for success, negative for failure
    std::string reason;        // Why this reinforcement was applied
    double timestamp;          // When reinforcement occurred
    uint64_t node_id;          // Associated memory node
    
    ReinforcementSignal() : instinct(InstinctType::SURVIVAL), delta(0.0f), timestamp(0.0), node_id(0) {}
    ReinforcementSignal(InstinctType inst, float d, const std::string& r, double time, uint64_t nid)
        : instinct(inst), delta(d), reason(r), timestamp(time), node_id(nid) {}
};

// ============================================================================
// INSTINCT ENGINE CLASS
// ============================================================================

class InstinctEngine {
private:
    std::map<InstinctType, InstinctWeight> instinct_weights;
    std::mutex instinct_mutex;
    
    // Reinforcement history for learning
    std::vector<ReinforcementSignal> reinforcement_history;
    std::mutex reinforcement_mutex;
    static constexpr size_t MAX_REINFORCEMENT_HISTORY = 10000;
    
    // Instinct activation patterns
    std::map<InstinctType, std::vector<double>> activation_timestamps;
    static constexpr double INSTINCT_DECAY_TIME = 3600.0; // 1 hour decay
    
    // Context sensitivity parameters
    static constexpr float CONFIDENCE_THRESHOLD_LOW = 0.4f;
    static constexpr float CONFIDENCE_THRESHOLD_HIGH = 0.7f;
    static constexpr float RESOURCE_LOAD_THRESHOLD = 0.7f;
    static constexpr float NOVELTY_THRESHOLD_HIGH = 0.8f;
    
    // Reinforcement parameters
    static constexpr float REINFORCEMENT_STRENGTH = 0.1f;
    static constexpr float REINFORCEMENT_DECAY = 0.95f;
    static constexpr float MIN_INSTINCT_WEIGHT = 0.1f;
    static constexpr float MAX_INSTINCT_WEIGHT = 1.0f;
    
public:
    InstinctEngine();
    ~InstinctEngine();
    
    // Core instinct engine methods
    void initialize_instincts();
    InstinctBias get_instinct_bias(const ContextState& context);
    void reinforce_instinct(InstinctType instinct, float delta, const std::string& reason, uint64_t node_id = 0);
    
    // Instinct influence calculation
    float calculate_instinct_influence(InstinctType instinct, const ContextState& context);
    std::map<InstinctType, float> calculate_all_instinct_influences(const ContextState& context);
    
    // Conflict resolution
    InstinctBias resolve_instinct_conflicts(const std::map<InstinctType, float>& influences, const ContextState& context);
    float apply_softmax_normalization(const std::vector<float>& values);
    
    // Context analysis
    ContextState analyze_context(float confidence, float resource_load, bool contradictions, 
                               bool user_interaction, bool memory_risk, float novelty, uint64_t complexity);
    
    // Instinct weight management
    void update_instinct_weight(InstinctType instinct, float new_weight);
    void apply_reinforcement_decay();
    void normalize_instinct_weights();
    
    // Memory node tagging
    std::vector<InstinctTag> generate_instinct_tags(const InstinctBias& bias, const std::string& context);
    std::string format_instinct_tags(const std::vector<InstinctTag>& tags);
    
    // Utility functions
    std::string instinct_type_to_string(InstinctType instinct);
    InstinctType string_to_instinct_type(const std::string& str);
    float calculate_instinct_decay(InstinctType instinct, double current_time);
    
    // Statistics and monitoring
    struct InstinctStats {
        std::map<InstinctType, float> current_weights;
        std::map<InstinctType, uint64_t> activation_counts;
        std::map<InstinctType, float> average_reinforcement;
        uint64_t total_reinforcements;
        double last_reinforcement_time;
    };
    
    InstinctStats get_instinct_statistics();
    std::string format_instinct_statistics(const InstinctStats& stats);
    
    // Demonstration and testing
    void demonstrate_instinct_conflict(const std::string& scenario_description);
    InstinctBias simulate_decision_scenario(const ContextState& context, const std::string& scenario);
};

// ============================================================================
// INSTINCT ENGINE INTEGRATION HELPERS
// ============================================================================

class InstinctIntegrationHelper {
public:
    // Integration with blended reasoning
    static InstinctBias influence_blended_reasoning(const InstinctBias& instinct_bias, 
                                                   float base_recall_weight, float base_exploration_weight);
    
    // Integration with memory nodes
    static std::vector<InstinctTag> extract_instinct_tags_from_node(uint64_t node_id);
    static void attach_instinct_tags_to_node(uint64_t node_id, const std::vector<InstinctTag>& tags);
    
    // Context state builders
    static ContextState build_context_from_confidence(float confidence);
    static ContextState build_context_from_resource_load(float resource_load);
    static ContextState build_context_from_contradictions(bool has_contradictions);
    static ContextState build_context_from_user_interaction(bool user_interaction);
    static ContextState build_context_from_memory_risk(bool memory_risk);
    
    // Formatting utilities
    static std::string format_instinct_bias(const InstinctBias& bias);
    static std::string format_context_state(const ContextState& context);
};
