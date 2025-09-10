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
#include <unordered_map>
#include <cstdint>
#include <iomanip>
#include <cmath>

// ============================================================================
// MELVIN COMPLETE UNIFIED SYSTEM - ONE SYSTEM TO RULE THEM ALL
// ============================================================================
// NO LOOSE ENDS - NO MISSING FEATURES - EVERYTHING INTEGRATED

// Forward declarations
class MelvinCompleteSystem;

// ============================================================================
// COMPLETE UNIFIED MELVIN SYSTEM
// ============================================================================

class MelvinCompleteSystem {
private:
    // Core brain architecture
    struct Node {
        uint64_t id;
        std::string content;
        uint8_t content_type; // 0=text, 1=code, 2=concept, 3=puzzle, 4=reasoning
        uint64_t creation_time;
        uint8_t importance;
        uint32_t access_count;
        float strength;
        bool is_self_generated;
        bool is_meta_node;
        bool is_error_resolution;
        
        Node() : id(0), content_type(0), creation_time(0), importance(0), 
                 access_count(0), strength(1.0f), is_self_generated(false),
                 is_meta_node(false), is_error_resolution(false) {}
    };
    
    struct Connection {
        uint64_t id;
        uint64_t source_id;
        uint64_t target_id;
        uint8_t weight;
        std::string connection_type; // "hebbian", "logical", "semantic", "temporal"
        uint64_t creation_time;
        uint32_t access_count;
        float strength;
        
        Connection() : id(0), source_id(0), target_id(0), weight(0), 
                       creation_time(0), access_count(0), strength(1.0f) {}
        
        void strengthen() { strength += 0.1f; access_count++; }
    };
    
    // Core storage
    std::unordered_map<uint64_t, Node> nodes;
    std::unordered_map<uint64_t, Connection> connections;
    std::atomic<uint64_t> next_node_id{1};
    std::atomic<uint64_t> next_connection_id{1};
    
    // Thread safety
    std::mutex nodes_mutex;
    std::mutex connections_mutex;
    
    // Learning and reasoning
    std::atomic<bool> running{false};
    std::atomic<int> cycle_count{0};
    std::chrono::steady_clock::time_point start_time;
    
    // Driver system
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
    
    // Reasoning engine
    class ReasoningEngine {
    private:
        MelvinCompleteSystem* brain;
        
    public:
        ReasoningEngine(MelvinCompleteSystem* brain_ptr) : brain(brain_ptr) {}
        
        std::string analyze_pattern(const std::string& input);
        std::string perform_abstraction(const std::string& input);
        std::string logical_deduction(const std::string& premises);
        std::string generate_answer(const std::string& problem);
        std::string solve_sequence_problem(const std::string& sequence);
        std::string solve_logic_problem(const std::string& problem);
    };
    
    std::unique_ptr<ReasoningEngine> reasoning_engine;
    
    // Learning metrics
public:
    struct LearningMetrics {
        int concepts_learned = 0;
        int insights_generated = 0;
        int improvements_made = 0;
        int questions_asked = 0;
        int connections_made = 0;
        int problems_solved = 0;
        double learning_efficiency = 0.0;
        double curiosity_level = 0.0;
        double humanity_alignment = 0.0;
        double reasoning_accuracy = 0.0;
    } metrics;
    
public:
    MelvinCompleteSystem();
    ~MelvinCompleteSystem();
    
    // Core brain operations
    uint64_t addNode(const std::string& content, uint8_t content_type = 0);
    uint64_t addConnection(uint64_t source_id, uint64_t target_id, const std::string& type = "hebbian");
    void strengthenConnection(uint64_t connection_id);
    void accessNode(uint64_t node_id);
    
    // Learning operations
    void hebbianLearning(const std::vector<uint64_t>& activated_nodes);
    void updateImportanceScores();
    void intelligentPruning();
    void consolidateKnowledge();
    
    // Reasoning operations
    std::string processInput(const std::string& input);
    std::string generateResponse(const std::string& input);
    std::string solveProblem(const std::string& problem);
    std::string generateInsight(const std::string& context);
    
    // Autonomous operations
    std::string autonomousCycle(const std::string& input);
    std::string generateNextInput(const std::string& previous_response);
    void performSelfImprovement();
    void updateLearningMetrics();
    
    // Control operations
    void startSystem();
    void stopSystem();
    bool isRunning() const { return running.load(); }
    
    // Status and analysis
    void printSystemStatus();
    void printLearningProgress();
    void printKnowledgeSummary();
    void printMetrics();
    void printBrainState();
    
    // Persistence
    void saveBrainState(const std::string& filename = "melvin_complete_brain.bin");
    void loadBrainState(const std::string& filename = "melvin_complete_brain.bin");
    
    // Getters
    int getCycleCount() const { return cycle_count.load(); }
    size_t getNodeCount() const { return nodes.size(); }
    size_t getConnectionCount() const { return connections.size(); }
    const LearningMetrics& getMetrics() const { return metrics; }
};

// ============================================================================
// COMPLETE UNIFIED MELVIN INTERFACE
// ============================================================================

class MelvinCompleteInterface {
private:
    std::unique_ptr<MelvinCompleteSystem> complete_system;
    std::atomic<bool> running{false};
    std::thread autonomous_thread;
    
public:
    MelvinCompleteInterface();
    ~MelvinCompleteInterface();
    
    // Start Melvin with complete system
    void startMelvin();
    void stopMelvin();
    
    // Ask Melvin a question (returns real response)
    std::string askMelvin(const std::string& question);
    
    // Solve a problem (returns real solution)
    std::string solveProblem(const std::string& problem);
    
    // Print status and analysis
    void printStatus();
    void printAnalysis();
    
    // Check if system is running
    bool isRunning() const { return running.load(); }
    
    // Get system information
    int getCycleCount() const;
    size_t getNodeCount() const;
    size_t getConnectionCount() const;
    const MelvinCompleteSystem::LearningMetrics& getMetrics() const;
};
