#pragma once

#include "melvin_common.hpp"
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <memory>

namespace melvin {

// Forward declarations
class RuleEngine;
class ConfidenceCalculator;
class MLHooks;

// Learning rule structure
struct LearningRule {
    uint64_t id;
    std::string name;
    std::string description;
    std::vector<std::string> conditions;
    std::vector<std::string> actions;
    Confidence confidence;
    Priority priority;
    uint64_t usage_count;
    TimePoint created_at;
    TimePoint last_used;
    bool is_active;
    
    LearningRule() : id(0), confidence(0.5f), priority(128), usage_count(0), is_active(true) {}
};

// Decision context
struct DecisionContext {
    std::map<std::string, double> sensor_values;
    std::map<std::string, std::string> environment_state;
    std::vector<std::string> available_actions;
    TimePoint timestamp;
    uint64_t context_id;
    
    DecisionContext() : context_id(0) {}
};

// Decision result
struct DecisionResult {
    std::string selected_action;
    Confidence confidence;
    std::vector<std::string> reasoning;
    std::vector<LearningRule> applied_rules;
    Duration decision_time;
    bool was_learned;
    
    DecisionResult() : confidence(0.0), was_learned(false) {}
};

// Learning configuration
struct LearningConfig {
    bool enable_rule_learning;
    bool enable_confidence_adaptation;
    bool enable_ml_integration;
    double min_confidence_threshold;
    double learning_rate;
    double forgetting_rate;
    size_t max_rules;
    size_t max_rule_conditions;
    Duration rule_timeout;
    Duration context_timeout;
    
    LearningConfig() : enable_rule_learning(true), enable_confidence_adaptation(true),
                      enable_ml_integration(false), min_confidence_threshold(0.3),
                      learning_rate(0.1), forgetting_rate(0.05), max_rules(1000),
                      max_rule_conditions(10), rule_timeout(Minutes(60)), context_timeout(Seconds(30)) {}
};

// Learning statistics
struct LearningStatistics {
    size_t total_rules;
    size_t active_rules;
    size_t total_decisions;
    size_t successful_decisions;
    size_t learned_decisions;
    double average_confidence;
    double average_decision_time;
    std::map<std::string, size_t> action_frequencies;
    TimePoint last_calculation;
    
    LearningStatistics() : total_rules(0), active_rules(0), total_decisions(0),
                          successful_decisions(0), learned_decisions(0), average_confidence(0.0),
                          average_decision_time(0.0) {}
};

// Learning Engine - Main cognitive learning system
class LearningEngine {
public:
    static LearningEngine& instance();
    
    // Initialization and shutdown
    Result<void> init(const LearningConfig& config = LearningConfig());
    Result<void> shutdown();
    
    // Rule management
    Result<uint64_t> add_rule(const LearningRule& rule);
    Result<void> update_rule(uint64_t rule_id, const LearningRule& rule);
    Result<void> delete_rule(uint64_t rule_id);
    Result<LearningRule> get_rule(uint64_t rule_id) const;
    Result<std::vector<LearningRule>> get_rules_by_condition(const std::string& condition) const;
    Result<std::vector<LearningRule>> get_rules_by_action(const std::string& action) const;
    
    // Decision making
    Result<DecisionResult> make_decision(const DecisionContext& context);
    Result<DecisionResult> make_decision_with_feedback(const DecisionContext& context,
                                                     const std::string& feedback_action,
                                                     bool was_successful);
    
    // Learning and adaptation
    Result<void> learn_from_feedback(uint64_t rule_id, bool was_successful, 
                                   const DecisionContext& context);
    Result<void> adapt_confidence(uint64_t rule_id, double new_confidence);
    Result<void> create_rule_from_example(const DecisionContext& context, 
                                        const std::string& action, bool was_successful);
    Result<void> generalize_rules();
    Result<void> specialize_rules();
    
    // Rule evaluation
    Result<double> evaluate_rule(uint64_t rule_id, const DecisionContext& context) const;
    Result<bool> rule_matches_context(uint64_t rule_id, const DecisionContext& context) const;
    Result<std::vector<std::string>> get_rule_conditions(uint64_t rule_id) const;
    Result<std::vector<std::string>> get_rule_actions(uint64_t rule_id) const;
    
    // Context management
    Result<uint64_t> create_context(const DecisionContext& context);
    Result<DecisionContext> get_context(uint64_t context_id) const;
    Result<void> update_context(uint64_t context_id, const DecisionContext& context);
    Result<void> delete_context(uint64_t context_id);
    
    // ML integration
    Result<void> enable_ml_integration(bool enable);
    Result<void> set_ml_model(const std::string& model_path, const std::string& model_type);
    Result<void> train_ml_model(const std::vector<DecisionContext>& contexts,
                               const std::vector<std::string>& actions);
    Result<std::string> predict_with_ml(const DecisionContext& context);
    
    // Analysis and statistics
    LearningStatistics get_statistics() const;
    Result<std::vector<LearningRule>> get_most_used_rules(size_t count = 10) const;
    Result<std::vector<LearningRule>> get_most_confident_rules(size_t count = 10) const;
    Result<std::vector<std::string>> get_most_common_actions(size_t count = 10) const;
    Result<double> calculate_rule_effectiveness(uint64_t rule_id) const;
    
    // Persistence
    Result<void> save_rules(const std::string& filename);
    Result<void> load_rules(const std::string& filename);
    Result<void> export_rules(const std::string& filename, const std::string& format);
    
    // Event callbacks
    using RuleEventCallback = std::function<void(uint64_t, const std::string&)>;
    using DecisionEventCallback = std::function<void(const DecisionResult&)>;
    using LearningEventCallback = std::function<void(const std::string&)>;
    
    void set_rule_event_callback(RuleEventCallback callback);
    void set_decision_event_callback(DecisionEventCallback callback);
    void set_learning_event_callback(LearningEventCallback callback);
    
    // Configuration
    LearningConfig get_config() const;
    Result<void> update_config(const LearningConfig& config);
    
    // Maintenance
    Result<void> cleanup_old_rules();
    Result<void> optimize_rules();
    Result<void> validate_rules();
    Result<void> backup_rules();

private:
    LearningEngine() = default;
    ~LearningEngine() = default;
    LearningEngine(const LearningEngine&) = delete;
    LearningEngine& operator=(const LearningEngine&) = delete;
    
    // Internal methods
    Result<void> validate_rule(const LearningRule& rule) const;
    Result<void> validate_context(const DecisionContext& context) const;
    Result<std::vector<LearningRule>> find_applicable_rules(const DecisionContext& context) const;
    Result<DecisionResult> apply_rules(const std::vector<LearningRule>& rules, 
                                     const DecisionContext& context);
    Result<void> notify_rule_event(uint64_t rule_id, const std::string& event);
    Result<void> notify_decision_event(const DecisionResult& result);
    Result<void> notify_learning_event(const std::string& event);
    
    // Member variables
    std::unique_ptr<RuleEngine> rule_engine_;
    std::unique_ptr<ConfidenceCalculator> confidence_calculator_;
    std::unique_ptr<MLHooks> ml_hooks_;
    
    LearningConfig config_;
    LearningStatistics statistics_;
    
    std::map<uint64_t, DecisionContext> contexts_;
    std::map<uint64_t, LearningRule> rules_;
    
    RuleEventCallback rule_event_callback_;
    DecisionEventCallback decision_event_callback_;
    LearningEventCallback learning_event_callback_;
    
    mutable std::shared_mutex engine_mutex_;
    bool initialized_;
    
    // Background tasks
    std::thread learning_thread_;
    std::atomic<bool> running_;
    
    void learning_loop();
    void update_statistics();
    void cleanup_expired_contexts();
};

// Utility functions for learning operations
namespace learning_utils {
    
    // Rule matching
    bool rule_matches_context(const LearningRule& rule, const DecisionContext& context);
    double calculate_rule_confidence(const LearningRule& rule, const DecisionContext& context);
    
    // Rule generalization
    LearningRule generalize_rule(const LearningRule& rule1, const LearningRule& rule2);
    LearningRule specialize_rule(const LearningRule& rule, const DecisionContext& context);
    
    // Context analysis
    std::vector<std::string> extract_key_features(const DecisionContext& context);
    double calculate_context_similarity(const DecisionContext& ctx1, const DecisionContext& ctx2);
    
    // Decision optimization
    std::vector<LearningRule> rank_rules_by_relevance(const std::vector<LearningRule>& rules,
                                                     const DecisionContext& context);
    
} // namespace learning_utils

} // namespace melvin
