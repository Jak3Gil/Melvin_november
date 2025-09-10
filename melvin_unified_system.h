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
#include <curl/curl.h>
#include <json/json.h>

// ============================================================================
// MELVIN UNIFIED SYSTEM - COMPLETE AUTONOMOUS AI BRAIN
// ============================================================================

// Forward declarations
class OllamaClient;
class MelvinUnifiedSystem;

// ============================================================================
// OLLAMA CLIENT FOR REAL AI RESPONSES
// ============================================================================

class OllamaClient {
private:
    std::string base_url;
    std::string model;
    CURL* curl;
    
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data);
    
public:
    OllamaClient(const std::string& url = "http://localhost:11434", const std::string& model_name = "llama3.2");
    ~OllamaClient();
    
    std::string generateResponse(const std::string& prompt, const std::string& context = "");
    std::string generateResponseWithParams(const std::string& prompt, 
                                         const std::string& context = "",
                                         float temperature = 0.7f,
                                         int max_tokens = 1000);
    bool isAvailable();
    std::vector<std::string> getAvailableModels();
    void setModel(const std::string& model_name);
    
    // Specialized prompt generation
    std::string generateAutonomousThinkingPrompt(const std::string& input, 
                                                const std::string& driver_context,
                                                const std::string& previous_thoughts = "");
    std::string generateCuriosityQuestion(const std::string& current_knowledge);
    std::string generateSelfImprovementReflection(const std::string& recent_cycles);
    std::string generateMetaCognitiveAnalysis(const std::string& thought_patterns);
};

// ============================================================================
// UNIFIED MELVIN SYSTEM
// ============================================================================

class MelvinUnifiedSystem {
private:
    // Core AI components
    std::unique_ptr<OllamaClient> ollama;
    
    // Real learning and memory
    std::vector<std::string> conversation_history;
    std::vector<std::string> learned_concepts;
    std::vector<std::string> generated_insights;
    std::vector<std::string> self_improvements;
    std::vector<std::string> knowledge_base;
    
    // Autonomous learning state
    std::atomic<bool> running{false};
    std::atomic<int> cycle_count{0};
    std::atomic<int> total_learning_cycles{0};
    std::chrono::steady_clock::time_point start_time;
    
    // Thread safety
    std::mutex history_mutex;
    std::mutex learning_mutex;
    std::mutex knowledge_mutex;
    
    // Learning patterns and strategies
    std::vector<std::string> question_patterns;
    std::vector<std::string> reflection_patterns;
    std::vector<std::string> improvement_strategies;
    
    // Real metrics (not fake)
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
    
public:
    MelvinUnifiedSystem(const std::string& ollama_url = "http://localhost:11434", 
                       const std::string& model = "llama3.2");
    ~MelvinUnifiedSystem();
    
    // Core autonomous learning methods
    std::string processAutonomousCycle(const std::string& input);
    std::string generateAutonomousResponse(const std::string& input);
    std::string generateNextInput(const std::string& previous_response);
    std::string extractConcepts(const std::string& text);
    std::string generateSelfReflection();
    
    // Real learning and improvement
    void learnFromResponse(const std::string& response);
    void generateInsight(const std::string& context);
    void performSelfImprovement();
    void updateLearningMetrics();
    
    // Knowledge management
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
// UNIFIED MELVIN INTERFACE
// ============================================================================

class MelvinUnifiedInterface {
private:
    std::unique_ptr<MelvinUnifiedSystem> unified_system;
    std::atomic<bool> running{false};
    std::thread autonomous_thread;
    
public:
    MelvinUnifiedInterface(const std::string& ollama_url = "http://localhost:11434", 
                           const std::string& model = "llama3.2");
    ~MelvinUnifiedInterface();
    
    // Start Melvin with unified autonomous learning
    void startMelvin();
    void stopMelvin();
    
    // Ask Melvin a question (returns real AI response)
    std::string askMelvin(const std::string& question);
    
    // Print status and analysis
    void printStatus();
    void printAnalysis();
    
    // Check if system is running
    bool isRunning() const { return running.load(); }
    
    // Get system information
    int getCycleCount() const;
    const MelvinUnifiedSystem::LearningMetrics& getMetrics() const;
    const std::vector<std::string>& getConversationHistory() const;
};
