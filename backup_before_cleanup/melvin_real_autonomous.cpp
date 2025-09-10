#include "melvin_real_autonomous.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <thread>
#include <fstream>

// ============================================================================
// MELVIN REAL AUTONOMOUS LEARNING IMPLEMENTATION
// ============================================================================

MelvinRealAutonomousLearning::MelvinRealAutonomousLearning(const std::string& ollama_url, 
                                                          const std::string& model) 
    : start_time(std::chrono::steady_clock::now()) {
    
    // Initialize autonomous learning system
    autonomous_system = std::make_unique<MelvinAutonomousLearning>();
    
    // Initialize Ollama client
    ollama_client = std::make_unique<OllamaClient>(ollama_url, model);
    
    std::cout << "🤖 Melvin Real Autonomous Learning initialized" << std::endl;
    std::cout << "🔗 Ollama URL: " << ollama_url << std::endl;
    std::cout << "🧠 Model: " << model << std::endl;
    
    // Check if Ollama is available
    if (ollama_client->isAvailable()) {
        std::cout << "✅ Ollama is available and ready!" << std::endl;
    } else {
        std::cout << "⚠️ Warning: Ollama may not be available at " << ollama_url << std::endl;
        std::cout << "   Make sure Ollama is running with: ollama serve" << std::endl;
    }
}

MelvinRealAutonomousLearning::~MelvinRealAutonomousLearning() {
    stopRealAutonomousLearning();
}

std::string MelvinRealAutonomousLearning::generateRealResponse(const std::string& input, 
                                                              const std::string& driver_context,
                                                              const std::string& previous_thoughts) {
    if (!ollama_client || !ollama_client->isAvailable()) {
        return "Error: Ollama not available. Please start Ollama with 'ollama serve'";
    }
    
    // Generate autonomous thinking prompt
    std::string prompt = ollama_client->generateAutonomousThinkingPrompt(input, driver_context, previous_thoughts);
    
    // Get real AI response
    std::string response = ollama_client->generateResponse(prompt);
    
    return response;
}

std::string MelvinRealAutonomousLearning::generateRealAutonomousInput(const std::string& previous_output, 
                                                                     DriverType dominant_driver, 
                                                                     int cycle_count) {
    if (!ollama_client || !ollama_client->isAvailable()) {
        // Fallback to simple pattern-based generation
        std::vector<std::string> fallback_inputs = {
            "What should I think about next?",
            "How can I improve my understanding?",
            "What new connections can I explore?",
            "How can I better serve humanity?",
            "What mysteries remain unsolved?"
        };
        return fallback_inputs[cycle_count % fallback_inputs.size()];
    }
    
    // Extract concepts from previous output
    std::istringstream iss(previous_output);
    std::string word;
    std::vector<std::string> concepts;
    
    while (iss >> word) {
        // Simple concept extraction
        if (word.length() > 3 && 
            (std::isupper(word[0]) || 
             word == "intelligence" || word == "learning" || word == "knowledge" ||
             word == "humanity" || word == "connection" || word == "pattern" ||
             word == "problem" || word == "solution" || word == "evolution" ||
             word == "curiosity" || word == "balance" || word == "growth")) {
            concepts.push_back(word);
        }
    }
    
    // Create context for input generation
    std::ostringstream context;
    context << "Previous output: " << previous_output.substr(0, 200) << "...\n";
    context << "Dominant driver: " << DriverLevels().getDriverName(dominant_driver) << "\n";
    context << "Cycle count: " << cycle_count << "\n";
    context << "Extracted concepts: ";
    for (const auto& concept : concepts) {
        context << concept << " ";
    }
    
    // Generate curiosity-driven question
    std::string prompt = ollama_client->generateCuriosityQuestion(context.str());
    std::string new_input = ollama_client->generateResponse(prompt);
    
    return new_input;
}

std::string MelvinRealAutonomousLearning::generateRealCuriosityQuestion(const std::string& current_knowledge) {
    if (!ollama_client || !ollama_client->isAvailable()) {
        return "What new patterns can I discover in my knowledge?";
    }
    
    std::string prompt = ollama_client->generateCuriosityQuestion(current_knowledge);
    return ollama_client->generateResponse(prompt);
}

std::string MelvinRealAutonomousLearning::generateRealSelfImprovementReflection(const std::vector<std::string>& recent_cycles) {
    if (!ollama_client || !ollama_client->isAvailable()) {
        return "I should focus on improving my learning efficiency and strengthening effective patterns.";
    }
    
    std::ostringstream cycles_context;
    for (size_t i = 0; i < std::min(recent_cycles.size(), size_t(5)); ++i) {
        cycles_context << "Cycle " << i << ": " << recent_cycles[i].substr(0, 100) << "...\n";
    }
    
    std::string prompt = ollama_client->generateSelfImprovementReflection(cycles_context.str());
    return ollama_client->generateResponse(prompt);
}

std::string MelvinRealAutonomousLearning::generateRealMetaCognitiveAnalysis(const std::vector<std::string>& thought_patterns) {
    if (!ollama_client || !ollama_client->isAvailable()) {
        return "I notice patterns in my thinking that suggest I should focus on higher-level principles.";
    }
    
    std::ostringstream patterns_context;
    for (size_t i = 0; i < std::min(thought_patterns.size(), size_t(5)); ++i) {
        patterns_context << "Pattern " << i << ": " << thought_patterns[i].substr(0, 100) << "...\n";
    }
    
    std::string prompt = ollama_client->generateMetaCognitiveAnalysis(patterns_context.str());
    return ollama_client->generateResponse(prompt);
}

std::string MelvinRealAutonomousLearning::processRealAutonomousCycle(const std::string& input, bool is_external) {
    cycle_count++;
    
    std::cout << "\n🔄 REAL AUTONOMOUS CYCLE " << cycle_count << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "📥 Input: " << input.substr(0, 100) << (input.length() > 100 ? "..." : "") << std::endl;
    
    // Process through autonomous system to get driver context
    uint64_t node_id = autonomous_system->processAutonomousCycle(input, is_external);
    
    // Get driver context from autonomous system
    std::ostringstream driver_context;
    driver_context << "Node ID: " << node_id << ", Cycle: " << cycle_count;
    
    // Get previous thoughts from conversation history
    std::string previous_thoughts;
    {
        std::lock_guard<std::mutex> lock(history_mutex);
        if (!conversation_history.empty()) {
            previous_thoughts = conversation_history.back();
        }
    }
    
    // Generate real AI response
    std::string real_response = generateRealResponse(input, driver_context.str(), previous_thoughts);
    
    std::cout << "🤖 Real AI Response: " << real_response.substr(0, 150) << (real_response.length() > 150 ? "..." : "") << std::endl;
    
    // Store in conversation history
    {
        std::lock_guard<std::mutex> lock(history_mutex);
        conversation_history.push_back(real_response);
        
        // Keep only last 10 responses to manage memory
        if (conversation_history.size() > 10) {
            conversation_history.erase(conversation_history.begin());
        }
    }
    
    return real_response;
}

void MelvinRealAutonomousLearning::startRealAutonomousLearning() {
    std::cout << "🚀 Starting Melvin Real Autonomous Learning..." << std::endl;
    std::cout << "🔗 Connected to Ollama for real AI responses" << std::endl;
    auto models = ollama_client->getAvailableModels();
    std::cout << "🧠 Using model: " << (models.empty() ? "default" : models[0]) << std::endl;
}

void MelvinRealAutonomousLearning::stopRealAutonomousLearning() {
    std::cout << "⏹️ Stopping Melvin Real Autonomous Learning..." << std::endl;
}

void MelvinRealAutonomousLearning::printRealAutonomousStatus() {
    std::cout << "\n📊 REAL AUTONOMOUS STATUS REPORT" << std::endl;
    std::cout << "=================================" << std::endl;
    std::cout << "🔄 Cycles completed: " << cycle_count.load() << std::endl;
    
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
    std::cout << "⏱️ Runtime: " << elapsed.count() << " seconds" << std::endl;
    
    std::cout << "🔗 Ollama Status: " << (ollama_client->isAvailable() ? "✅ Available" : "❌ Unavailable") << std::endl;
    
    {
        std::lock_guard<std::mutex> lock(history_mutex);
        std::cout << "💬 Conversation history: " << conversation_history.size() << " responses" << std::endl;
    }
    
    // Print autonomous system status
    autonomous_system->printAutonomousStatistics();
}

void MelvinRealAutonomousLearning::printRealAutonomousAnalysis() {
    printRealAutonomousStatus();
}

bool MelvinRealAutonomousLearning::isOllamaAvailable() {
    return ollama_client && ollama_client->isAvailable();
}

std::vector<std::string> MelvinRealAutonomousLearning::getConversationHistory() {
    std::lock_guard<std::mutex> lock(history_mutex);
    return conversation_history;
}

void MelvinRealAutonomousLearning::saveRealAutonomousState() {
    std::cout << "💾 Saving real autonomous state..." << std::endl;
    // Save conversation history to file
    std::ofstream file("melvin_conversation_history.txt");
    if (file.is_open()) {
        for (const auto& response : conversation_history) {
            file << response << "\n---\n";
        }
        file.close();
        std::cout << "✅ Conversation history saved to melvin_conversation_history.txt" << std::endl;
    }
}

void MelvinRealAutonomousLearning::loadRealAutonomousState() {
    std::cout << "📂 Loading real autonomous state..." << std::endl;
    // Load conversation history from file
    std::ifstream file("melvin_conversation_history.txt");
    if (file.is_open()) {
        std::string line;
        std::string current_response;
        while (std::getline(file, line)) {
            if (line == "---") {
                if (!current_response.empty()) {
                    conversation_history.push_back(current_response);
                    current_response.clear();
                }
            } else {
                current_response += line + "\n";
            }
        }
        file.close();
        std::cout << "✅ Loaded " << conversation_history.size() << " responses from history" << std::endl;
    }
}

// ============================================================================
// MELVIN REAL AUTONOMOUS INTERFACE IMPLEMENTATION
// ============================================================================

MelvinRealAutonomousInterface::MelvinRealAutonomousInterface(const std::string& ollama_url, 
                                                             const std::string& model) {
    real_learning = std::make_unique<MelvinRealAutonomousLearning>(ollama_url, model);
}

MelvinRealAutonomousInterface::~MelvinRealAutonomousInterface() {
    stopMelvinRealAutonomous();
}

void MelvinRealAutonomousInterface::startMelvinRealAutonomous() {
    if (running.load()) {
        std::cout << "⚠️ Melvin Real Autonomous is already running!" << std::endl;
        return;
    }
    
    running.store(true);
    real_learning->startRealAutonomousLearning();
    
    std::cout << "🚀 Melvin Real Autonomous Learning started!" << std::endl;
    std::cout << "🤖 Real AI responses enabled via Ollama" << std::endl;
}

void MelvinRealAutonomousInterface::stopMelvinRealAutonomous() {
    if (!running.load()) {
        return;
    }
    
    running.store(false);
    real_learning->stopRealAutonomousLearning();
    
    std::cout << "⏹️ Melvin Real Autonomous Learning stopped!" << std::endl;
}

std::string MelvinRealAutonomousInterface::askMelvinRealAutonomous(const std::string& question) {
    if (!running.load()) {
        return "Error: Melvin Real Autonomous is not running. Call startMelvinRealAutonomous() first.";
    }
    
    return real_learning->processRealAutonomousCycle(question, true);
}

void MelvinRealAutonomousInterface::printMelvinRealAutonomousStatus() {
    if (real_learning) {
        real_learning->printRealAutonomousStatus();
    }
}

void MelvinRealAutonomousInterface::printRealAutonomousAnalysis() {
    if (real_learning) {
        real_learning->printRealAutonomousAnalysis();
    }
}
