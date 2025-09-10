#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <future>

// Include the real AI client
#include "ollama_client.h"

// ============================================================================
// CORE BRAIN ARCHITECTURE
// ============================================================================

struct Node {
    uint64_t id;
    std::string content;
    int importance = 0;
    int access_count = 0;
    double strength = 1.0;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_accessed;
    
    Node() : id(0), created_at(std::chrono::steady_clock::now()), 
             last_accessed(std::chrono::steady_clock::now()) {}
    
    Node(uint64_t node_id, const std::string& node_content, int imp = 0) 
        : id(node_id), content(node_content), importance(imp), 
          created_at(std::chrono::steady_clock::now()),
          last_accessed(std::chrono::steady_clock::now()) {}
};

struct Connection {
    uint64_t id;
    uint64_t from_node;
    uint64_t to_node;
    std::string connection_type;
    double strength = 1.0;
    int access_count = 0;
    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point last_accessed;
    
    Connection() : id(0), from_node(0), to_node(0), 
                   created_at(std::chrono::steady_clock::now()),
                   last_accessed(std::chrono::steady_clock::now()) {}
    
    Connection(uint64_t conn_id, uint64_t from, uint64_t to, const std::string& type)
        : id(conn_id), from_node(from), to_node(to), connection_type(type),
          created_at(std::chrono::steady_clock::now()),
          last_accessed(std::chrono::steady_clock::now()) {}
    
    void strengthen() {
        strength += 0.1f;
        access_count++;
        last_accessed = std::chrono::steady_clock::now();
    }
};

// ============================================================================
// DRIVER SYSTEM (MOTIVATIONAL FORCES)
// ============================================================================

class DriverLevels {
public:
    double dopamine = 0.5;      // Curiosity, exploration
    double serotonin = 0.5;     // Balance, stability
    double endorphins = 0.5;    // Satisfaction, reward
    double oxytocin = 0.5;      // Connection, empathy
    double adrenaline = 0.5;    // Urgency, focus
    
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
        std::map<double, std::string> driver_map = {
            {dopamine, "dopamine (curiosity)"},
            {serotonin, "serotonin (balance)"},
            {endorphins, "endorphins (satisfaction)"},
            {oxytocin, "oxytocin (connection)"},
            {adrenaline, "adrenaline (urgency)"}
        };
        
        return std::max_element(driver_map.begin(), driver_map.end())->second;
    }
};

// ============================================================================
// REASONING ENGINE
// ============================================================================

class ReasoningEngine {
public:
    std::string generate_answer(const std::string& question) {
        // Enhanced pattern recognition
        if (question.find("sequence") != std::string::npos || question.find("pattern") != std::string::npos) {
            return "I can see patterns in sequences. Let me analyze the relationship between elements.";
        } else if (question.find("intelligence") != std::string::npos) {
            return "Intelligence is the capacity to learn, adapt, and solve problems effectively.";
        } else if (question.find("humanity") != std::string::npos) {
            return "I aim to serve humanity by helping solve complex problems and advancing understanding.";
        } else if (question.find("quantum") != std::string::npos) {
            return "Quantum computing leverages quantum mechanical phenomena to process information.";
        } else if (question.find("ethical") != std::string::npos) {
            return "AI ethics involves ensuring AI systems are fair, transparent, and beneficial to humanity.";
        } else if (question.find("neural") != std::string::npos) {
            return "Neural networks learn through adjusting connection weights based on training data.";
        } else {
            return "I'm analyzing this question using my reasoning capabilities.";
        }
    }
    
    std::string recognize_pattern(const std::string& input) {
        if (input.find("2, 4, 8, 16") != std::string::npos) {
            return "Pattern: Each number doubles the previous (2^n)";
        } else if (input.find("1, 3, 5, 7") != std::string::npos) {
            return "Pattern: Odd numbers in sequence";
        } else if (input.find("A, B, C, D") != std::string::npos) {
            return "Pattern: Alphabetical sequence";
        }
        return "Pattern: Analyzing...";
    }
    
    std::string logical_deduction(const std::string& premise) {
        return "Based on the premise, I can deduce logical conclusions.";
    }
    
    std::string abstract_thinking(const std::string& concrete) {
        return "Abstracting from the concrete to find general principles.";
    }
};

// ============================================================================
// LEARNING METRICS
// ============================================================================

struct LearningMetrics {
    int concepts_learned = 0;
    int insights_generated = 0;
    int improvements_made = 0;
    int questions_asked = 0;
    int connections_made = 0;
    int problems_solved = 0;
    int ai_responses = 0;
    int fallback_responses = 0;
    double learning_efficiency = 0.0;
    double curiosity_level = 0.0;
    double reasoning_accuracy = 0.0;
    double humanity_alignment = 0.0;
};

// ============================================================================
// MELVIN ROBUST COMPLETE SYSTEM
// ============================================================================

class MelvinRobustCompleteSystem {
private:
    std::map<uint64_t, Node> nodes;
    std::map<uint64_t, Connection> connections;
    std::atomic<uint64_t> next_node_id{1};
    std::atomic<uint64_t> next_connection_id{1};
    std::atomic<int> cycle_count{0};
    
    DriverLevels drivers;
    std::unique_ptr<ReasoningEngine> reasoning_engine;
    LearningMetrics metrics;
    
    // ROBUST AI CLIENT WITH TIMEOUTS
    std::unique_ptr<OllamaClient> ollama_client;
    std::vector<std::string> conversation_history;
    
    std::mutex brain_mutex;
    std::atomic<bool> running{false};
    
    // Self-improvement tracking
    std::chrono::steady_clock::time_point last_improvement;
    
    // Timeout settings
    static constexpr int AI_TIMEOUT_SECONDS = 10;
    
public:
    MelvinRobustCompleteSystem();
    ~MelvinRobustCompleteSystem();
    
    // Core system operations
    void startSystem();
    void stopSystem();
    bool isRunning() const { return running.load(); }
    
    // Node and connection management
    uint64_t addNode(const std::string& content, int importance = 1);
    uint64_t addConnection(uint64_t from, uint64_t to, const std::string& type);
    void updateImportanceScores();
    void intelligentPruning();
    void consolidateKnowledge();
    
    // Learning operations
    void hebbianLearning(const std::vector<uint64_t>& activated_nodes);
    void performSelfImprovement();
    void updateLearningMetrics();
    
    // ROBUST AI operations with timeouts
    std::string getRobustAIResponse(const std::string& input);
    std::string generateNextRobustInput(const std::string& previous_response);
    std::string processInput(const std::string& input);
    std::string generateResponse(const std::string& input);
    std::string solveProblem(const std::string& problem);
    std::string generateInsight(const std::string& context);
    
    // Autonomous operations
    std::string autonomousCycle(const std::string& input);
    std::string generateNextInput(const std::string& previous_response);
    
    // Status and analysis
    void printSystemStatus();
    void printAnalysis();
    
    // Persistence
    void saveBrainState(const std::string& filename);
    void loadBrainState(const std::string& filename);
    void saveConversationHistory();
    void loadConversationHistory();
    
    // Helper methods
    std::string getFallbackResponse(const std::string& input);
    std::string getFallbackInput();
};

// ============================================================================
// ROBUST COMPLETE UNIFIED MELVIN INTERFACE
// ============================================================================

class MelvinRobustCompleteInterface {
private:
    std::unique_ptr<MelvinRobustCompleteSystem> complete_system;
    std::atomic<bool> running{false};
    
public:
    MelvinRobustCompleteInterface();
    ~MelvinRobustCompleteInterface();
    
    void startMelvin();
    void stopMelvin();
    std::string askMelvin(const std::string& question);
    std::string solveProblem(const std::string& problem);
    void printStatus();
    void printAnalysis();
};
