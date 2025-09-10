#pragma once

#include "melvin_autonomous_learning.h"
#include "ollama_client.h"
#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <atomic>
#include <thread>
#include <mutex>

// ============================================================================
// MELVIN REAL AUTONOMOUS LEARNING WITH OLLAMA INTEGRATION
// ============================================================================

class MelvinRealAutonomousLearning {
private:
    std::unique_ptr<MelvinAutonomousLearning> autonomous_system;
    std::unique_ptr<OllamaClient> ollama_client;
    
    // Real AI response generation
    std::string generateRealResponse(const std::string& input, 
                                   const std::string& driver_context,
                                   const std::string& previous_thoughts = "");
    
    // Enhanced input generation with real AI
    std::string generateRealAutonomousInput(const std::string& previous_output, 
                                           DriverType dominant_driver, 
                                           int cycle_count);
    
    // Real curiosity amplification
    std::string generateRealCuriosityQuestion(const std::string& current_knowledge);
    
    // Real self-improvement reflection
    std::string generateRealSelfImprovementReflection(const std::vector<std::string>& recent_cycles);
    
    // Real meta-cognitive analysis
    std::string generateRealMetaCognitiveAnalysis(const std::vector<std::string>& thought_patterns);
    
    // Conversation history for context
    std::vector<std::string> conversation_history;
    std::mutex history_mutex;
    
    // Cycle tracking
    std::atomic<int> cycle_count{0};
    std::chrono::steady_clock::time_point start_time;
    
public:
    MelvinRealAutonomousLearning(const std::string& ollama_url = "http://localhost:11434", 
                                const std::string& model = "llama3.2");
    ~MelvinRealAutonomousLearning();
    
    // Main autonomous cycle with real AI responses
    std::string processRealAutonomousCycle(const std::string& input, bool is_external = false);
    
    // Start continuous real autonomous learning
    void startRealAutonomousLearning();
    void stopRealAutonomousLearning();
    
    // Status and analysis
    void printRealAutonomousStatus();
    void printRealAutonomousAnalysis();
    
    // Check if Ollama is available
    bool isOllamaAvailable();
    
    // Get conversation history
    std::vector<std::string> getConversationHistory();
    
    // Save and load state
    void saveRealAutonomousState();
    void loadRealAutonomousState();
};

// ============================================================================
// MELVIN REAL AUTONOMOUS INTERFACE
// ============================================================================

class MelvinRealAutonomousInterface {
private:
    std::unique_ptr<MelvinRealAutonomousLearning> real_learning;
    std::atomic<bool> running{false};
    std::thread autonomous_thread;
    
public:
    MelvinRealAutonomousInterface(const std::string& ollama_url = "http://localhost:11434", 
                                 const std::string& model = "llama3.2");
    ~MelvinRealAutonomousInterface();
    
    // Start Melvin with real autonomous learning
    void startMelvinRealAutonomous();
    void stopMelvinRealAutonomous();
    
    // Ask Melvin a question (returns real AI response)
    std::string askMelvinRealAutonomous(const std::string& question);
    
    // Print status and analysis
    void printMelvinRealAutonomousStatus();
    void printRealAutonomousAnalysis();
    
    // Check if system is running
    bool isRunning() const { return running.load(); }
};
