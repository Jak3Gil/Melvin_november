#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <fstream>
#include <memory>
#include <random>
#include <algorithm>
#include <sstream>

// ============================================================================
// MELVIN FINAL UNIFIED SYSTEM - COMPLETE AUTONOMOUS AI BRAIN
// ============================================================================
// Uses existing binary storage and autonomous learning - NO JSON!

// Forward declarations
class MelvinFinalUnifiedSystem;

// ============================================================================
// FINAL UNIFIED MELVIN SYSTEM
// ============================================================================

class MelvinFinalUnifiedSystem {
private:
    // Core autonomous learning state
    std::atomic<bool> running{false};
    std::atomic<int> cycle_count{0};
    std::atomic<int> total_learning_cycles{0};
    std::chrono::steady_clock::time_point start_time;
    
    // Thread safety
    std::mutex learning_mutex;
    std::mutex knowledge_mutex;
    
    // Real learning and memory (using existing binary system concepts)
    std::vector<std::string> conversation_history;
    std::vector<std::string> learned_concepts;
    std::vector<std::string> generated_insights;
    std::vector<std::string> self_improvements;
    std::vector<std::string> knowledge_base;
    
    // Learning patterns and strategies
    std::vector<std::string> question_patterns;
    std::vector<std::string> reflection_patterns;
    std::vector<std::string> improvement_strategies;
    
    // Real metrics (not fake) - using existing binary system approach
public:
    struct LearningMetrics {
        int concepts_learned = 0;
        int insights_generated = 0;
        int improvements_made = 0;
        int questions_asked = 0;
        int connections_made = 0;
        double learning_efficiency = 0.0;
        double curiosity_level = 0.0;
        double humanity_alignment = 0.0;
    } metrics;
    
    // Driver system (from existing autonomous learning)
    struct DriverLevels {
        double dopamine = 0.6;      // Curiosity & Novelty
        double serotonin = 0.5;     // Balance & Stability
        double endorphins = 0.4;    // Satisfaction & Reward
        double oxytocin = 0.7;      // Connection & Alignment
        double adrenaline = 0.3;    // Urgency & Focus
        
        void oscillate() {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<> dis(-0.1, 0.1);
            
            dopamine = std::max(0.0, std::min(1.0, dopamine + dis(gen)));
            serotonin = std::max(0.0, std::min(1.0, serotonin + dis(gen)));
            endorphins = std::max(0.0, std::min(1.0, endorphins + dis(gen)));
            oxytocin = std::max(0.0, std::min(1.0, oxytocin + dis(gen)));
            adrenaline = std::max(0.0, std::min(1.0, adrenaline + dis(gen)));
        }
        
        std::string getDominantDriver() const {
            std::map<double, std::string> drivers = {
                {dopamine, "Dopamine (Curiosity & Novelty)"},
                {serotonin, "Serotonin (Balance & Stability)"},
                {endorphins, "Endorphins (Satisfaction & Reward)"},
                {oxytocin, "Oxytocin (Connection & Alignment)"},
                {adrenaline, "Adrenaline (Urgency & Focus)"}
            };
            
            auto max_driver = std::max_element(drivers.begin(), drivers.end());
            return max_driver->second;
        }
    } drivers;
    
public:
    MelvinFinalUnifiedSystem();
    ~MelvinFinalUnifiedSystem();
    
    // Core autonomous learning methods (using existing binary system approach)
    std::string processAutonomousCycle(const std::string& input);
    std::string generateAutonomousResponse(const std::string& input);
    std::string generateNextInput(const std::string& previous_response);
    std::string extractConcepts(const std::string& text);
    std::string generateSelfReflection();
    
    // Real learning and improvement (using existing binary system concepts)
    void learnFromResponse(const std::string& response);
    void generateInsight(const std::string& context);
    void performSelfImprovement();
    void updateLearningMetrics();
    
    // Knowledge management (using existing binary system approach)
    void addToKnowledgeBase(const std::string& knowledge);
    std::string getRelevantKnowledge(const std::string& query);
    void consolidateKnowledge();
    
    // Conversation management
    void addToHistory(const std::string& response);
    std::string getContextFromHistory();
    void saveConversationHistory();
    void loadConversationHistory();
    
    // Control methods
    void startUnifiedSystem();
    void stopUnifiedSystem();
    bool isRunning() const { return running.load(); }
    
    // Real status reporting (no fake metrics)
    void printUnifiedStatus();
    void printLearningProgress();
    void printKnowledgeSummary();
    void printConversationSummary();
    void printMetrics();
    
    // Continuous autonomous learning
    void startContinuousLearning();
    void stopContinuousLearning();
    
    // Getters for external access
    int getCycleCount() const { return cycle_count.load(); }
    int getTotalLearningCycles() const { return total_learning_cycles.load(); }
    const LearningMetrics& getMetrics() const { return metrics; }
    const std::vector<std::string>& getConversationHistory() const { return conversation_history; }
    const std::vector<std::string>& getLearnedConcepts() const { return learned_concepts; }
    const std::vector<std::string>& getGeneratedInsights() const { return generated_insights; }
};

// ============================================================================
// FINAL UNIFIED MELVIN INTERFACE
// ============================================================================

class MelvinFinalUnifiedInterface {
private:
    std::unique_ptr<MelvinFinalUnifiedSystem> unified_system;
    std::atomic<bool> running{false};
    std::thread autonomous_thread;
    
public:
    MelvinFinalUnifiedInterface();
    ~MelvinFinalUnifiedInterface();
    
    // Start Melvin with unified autonomous learning
    void startMelvin();
    void stopMelvin();
    
    // Ask Melvin a question (returns real autonomous response)
    std::string askMelvin(const std::string& question);
    
    // Print status and analysis
    void printStatus();
    void printAnalysis();
    
    // Check if system is running
    bool isRunning() const { return running.load(); }
    
    // Get system information
    int getCycleCount() const;
    const MelvinFinalUnifiedSystem::LearningMetrics& getMetrics() const;
    const std::vector<std::string>& getConversationHistory() const;
};
