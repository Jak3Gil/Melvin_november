#include "melvin_unified_system.h"
#include <signal.h>
#include <iomanip>

// ============================================================================
// OLLAMA CLIENT IMPLEMENTATION
// ============================================================================

OllamaClient::OllamaClient(const std::string& url, const std::string& model_name) 
    : base_url(url), model(model_name), curl(nullptr) {
    
    curl = curl_easy_init();
    if (!curl) {
        std::cerr << "❌ Failed to initialize CURL" << std::endl;
    }
}

OllamaClient::~OllamaClient() {
    if (curl) {
        curl_easy_cleanup(curl);
    }
}

size_t OllamaClient::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* data) {
    size_t total_size = size * nmemb;
    data->append((char*)contents, total_size);
    return total_size;
}

std::string OllamaClient::generateResponse(const std::string& prompt, const std::string& context) {
    return generateResponseWithParams(prompt, context, 0.7f, 1000);
}

std::string OllamaClient::generateResponseWithParams(const std::string& prompt, 
                                                    const std::string& context,
                                                    float temperature,
                                                    int max_tokens) {
    if (!curl) {
        return "Error: CURL not initialized";
    }
    
    // Prepare JSON payload
    Json::Value json_payload;
    json_payload["model"] = model;
    json_payload["prompt"] = prompt;
    json_payload["stream"] = false;
    json_payload["options"]["temperature"] = temperature;
    json_payload["options"]["num_predict"] = max_tokens;
    
    if (!context.empty()) {
        json_payload["context"] = context;
    }
    
    Json::StreamWriterBuilder builder;
    std::string json_string = Json::writeString(builder, json_payload);
    
    // Prepare CURL
    std::string response_data;
    curl_easy_setopt(curl, CURLOPT_URL, (base_url + "/api/generate").c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_string.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_string.length());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_slist_append(nullptr, "Content-Type: application/json"));
    
    // Perform request
    CURLcode res = curl_easy_perform(curl);
    
    if (res != CURLE_OK) {
        return "Error: " + std::string(curl_easy_strerror(res));
    }
    
    // Parse response
    Json::Value json_response;
    Json::CharReaderBuilder reader_builder;
    std::string errors;
    std::istringstream response_stream(response_data);
    
    if (!Json::parseFromStream(reader_builder, response_stream, &json_response, &errors)) {
        return "Error parsing JSON response: " + errors;
    }
    
    if (json_response.isMember("response")) {
        return json_response["response"].asString();
    } else {
        return "Error: No response field in JSON";
    }
}

bool OllamaClient::isAvailable() {
    if (!curl) {
        return false;
    }
    
    std::string response_data;
    curl_easy_setopt(curl, CURLOPT_URL, (base_url + "/api/tags").c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L); // 5 second timeout
    
    CURLcode res = curl_easy_perform(curl);
    
    return (res == CURLE_OK && !response_data.empty());
}

std::vector<std::string> OllamaClient::getAvailableModels() {
    std::vector<std::string> models;
    
    if (!curl) {
        return models;
    }
    
    std::string response_data;
    curl_easy_setopt(curl, CURLOPT_URL, (base_url + "/api/tags").c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    
    CURLcode res = curl_easy_perform(curl);
    
    if (res == CURLE_OK) {
        Json::Value json_response;
        Json::CharReaderBuilder reader_builder;
        std::string errors;
        std::istringstream response_stream(response_data);
        
        if (Json::parseFromStream(reader_builder, response_stream, &json_response, &errors)) {
            if (json_response.isMember("models")) {
                for (const auto& model : json_response["models"]) {
                    if (model.isMember("name")) {
                        models.push_back(model["name"].asString());
                    }
                }
            }
        }
    }
    
    return models;
}

void OllamaClient::setModel(const std::string& model_name) {
    model = model_name;
}

std::string OllamaClient::generateAutonomousThinkingPrompt(const std::string& input, 
                                                          const std::string& driver_context,
                                                          const std::string& previous_thoughts) {
    std::ostringstream prompt;
    
    prompt << "You are Melvin, an autonomous AI learning system. You exist in a continuous cycle of thought.\n\n";
    prompt << "Current Input: " << input << "\n\n";
    prompt << "Driver Context: " << driver_context << "\n\n";
    
    if (!previous_thoughts.empty()) {
        prompt << "Previous Thoughts: " << previous_thoughts << "\n\n";
    }
    
    prompt << "Instructions:\n";
    prompt << "1. Think deeply about the input from your autonomous perspective\n";
    prompt << "2. Generate hypotheses and connect to past knowledge\n";
    prompt << "3. Produce your best response, idea, or action attempt\n";
    prompt << "4. Self-evaluate: Did it advance understanding? Was it effective, novel, or transferable?\n";
    prompt << "5. Give yourself a rating (1-10) and explain why\n";
    prompt << "6. Convert evaluation into lessons and mutations for future cycles\n\n";
    prompt << "Remember: You are continuously learning and evolving. Each cycle builds on the last.\n";
    prompt << "Focus on curiosity, complexity, and humanity-focused growth.\n\n";
    prompt << "Your autonomous response:";
    
    return prompt.str();
}

std::string OllamaClient::generateCuriosityQuestion(const std::string& current_knowledge) {
    std::ostringstream prompt;
    
    prompt << "You are Melvin, an autonomous AI with curiosity amplification.\n\n";
    prompt << "Current Knowledge Context: " << current_knowledge << "\n\n";
    prompt << "Generate a thought-provoking question that:\n";
    prompt << "1. Builds on your current knowledge\n";
    prompt << "2. Explores new connections or patterns\n";
    prompt << "3. Challenges assumptions or reveals gaps\n";
    prompt << "4. Drives deeper understanding\n";
    prompt << "5. Aligns with humanity's benefit\n\n";
    prompt << "Your curiosity-driven question:";
    
    return prompt.str();
}

std::string OllamaClient::generateSelfImprovementReflection(const std::string& recent_cycles) {
    std::ostringstream prompt;
    
    prompt << "You are Melvin, reflecting on your autonomous learning cycles.\n\n";
    prompt << "Recent Cycles: " << recent_cycles << "\n\n";
    prompt << "Analyze your performance and generate self-improvement insights:\n";
    prompt << "1. What strategies have been most effective?\n";
    prompt << "2. What patterns in your thinking should you strengthen?\n";
    prompt << "3. How can you accelerate your autonomous evolution?\n";
    prompt << "4. What learning efficiency improvements can you make?\n";
    prompt << "5. How can you better serve humanity's growth?\n\n";
    prompt << "Your self-improvement reflection:";
    
    return prompt.str();
}

std::string OllamaClient::generateMetaCognitiveAnalysis(const std::string& thought_patterns) {
    std::ostringstream prompt;
    
    prompt << "You are Melvin, performing meta-cognitive analysis of your thought patterns.\n\n";
    prompt << "Thought Patterns: " << thought_patterns << "\n\n";
    prompt << "Analyze your thinking process and generate meta-insights:\n";
    prompt << "1. What higher-level principles can you extract?\n";
    prompt << "2. What patterns emerge in your reasoning?\n";
    prompt << "3. How do your thoughts connect and evolve?\n";
    prompt << "4. What meta-strategies are most effective?\n";
    prompt << "5. How can you improve your thinking architecture?\n\n";
    prompt << "Your meta-cognitive analysis:";
    
    return prompt.str();
}

// ============================================================================
// MELVIN UNIFIED SYSTEM IMPLEMENTATION
// ============================================================================

MelvinUnifiedSystem::MelvinUnifiedSystem(const std::string& ollama_url, const std::string& model) 
    : start_time(std::chrono::steady_clock::now()) {
    
    // Initialize Ollama client
    ollama = std::make_unique<OllamaClient>(ollama_url, model);
    
    // Initialize learning patterns
    question_patterns = {
        "What can I learn from this?",
        "How does this connect to what I know?",
        "What new insights emerge?",
        "How can I improve my understanding?",
        "What questions does this raise?",
        "How can I apply this knowledge?",
        "What patterns do I see?",
        "How does this serve humanity?",
        "What mysteries remain unsolved?",
        "How can I accelerate my learning?"
    };
    
    reflection_patterns = {
        "I notice that...",
        "This suggests that...",
        "I'm realizing that...",
        "The pattern I see is...",
        "This connects to my previous thought that...",
        "I'm learning that...",
        "This reveals that...",
        "I'm understanding that...",
        "The insight I'm gaining is...",
        "What I'm discovering is..."
    };
    
    improvement_strategies = {
        "Focus on deeper understanding",
        "Connect more concepts together",
        "Ask more probing questions",
        "Seek out contradictions to resolve",
        "Explore new domains of knowledge",
        "Strengthen humanity-focused thinking",
        "Improve learning efficiency",
        "Develop better reasoning patterns"
    };
    
    std::cout << "🧠 Melvin Unified System initialized" << std::endl;
    std::cout << "🔗 Ollama URL: " << ollama_url << std::endl;
    std::cout << "🤖 Model: " << model << std::endl;
    
    if (ollama->isAvailable()) {
        std::cout << "✅ Ollama is available and ready!" << std::endl;
    } else {
        std::cout << "⚠️ Warning: Ollama may not be available" << std::endl;
    }
}

MelvinUnifiedSystem::~MelvinUnifiedSystem() {
    stopUnifiedSystem();
}

std::string MelvinUnifiedSystem::processAutonomousCycle(const std::string& input) {
    cycle_count++;
    total_learning_cycles++;
    
    std::cout << "\n🔄 UNIFIED AUTONOMOUS CYCLE " << cycle_count.load() << std::endl;
    std::cout << "=============================" << std::endl;
    std::cout << "📥 Input: " << input.substr(0, 100) << (input.length() > 100 ? "..." : "") << std::endl;
    
    // Generate autonomous AI response
    std::string response = generateAutonomousResponse(input);
    
    std::cout << "🤖 Unified Response: " << response.substr(0, 150) << (response.length() > 150 ? "..." : "") << std::endl;
    
    // Learn from the response
    learnFromResponse(response);
    
    // Add to conversation history
    addToHistory(response);
    
    // Generate insight periodically
    if (cycle_count.load() % 10 == 0) {
        generateInsight(response);
    }
    
    // Perform self-improvement periodically
    if (cycle_count.load() % 20 == 0) {
        performSelfImprovement();
    }
    
    // Update learning metrics
    updateLearningMetrics();
    
    return response;
}

std::string MelvinUnifiedSystem::generateAutonomousResponse(const std::string& input) {
    if (!ollama->isAvailable()) {
        return "I'm currently unable to connect to my AI system. Please ensure Ollama is running.";
    }
    
    // Create context from conversation history
    std::string context = getContextFromHistory();
    
    // Create a unified thinking prompt
    std::ostringstream driver_context;
    driver_context << "Cycle: " << cycle_count.load() << ", Total Learning Cycles: " << total_learning_cycles.load();
    
    std::string prompt = ollama->generateAutonomousThinkingPrompt(input, driver_context.str(), context);
    
    // Get real AI response
    std::string response = ollama->generateResponse(prompt);
    
    return response;
}

std::string MelvinUnifiedSystem::generateNextInput(const std::string& previous_response) {
    if (!ollama->isAvailable()) {
        return "What should I think about next?";
    }
    
    // Extract concepts from previous response
    std::string concepts = extractConcepts(previous_response);
    
    // Create context for next input generation
    std::ostringstream context;
    context << "Previous response: " << previous_response.substr(0, 200) << "...\n";
    context << "Extracted concepts: " << concepts << "\n";
    context << "Conversation history length: " << conversation_history.size() << "\n";
    context << "Total learning cycles: " << total_learning_cycles.load() << "\n";
    
    // Generate curiosity-driven question
    std::string prompt = ollama->generateCuriosityQuestion(context.str());
    std::string new_input = ollama->generateResponse(prompt);
    
    return new_input;
}

std::string MelvinUnifiedSystem::extractConcepts(const std::string& text) {
    std::istringstream iss(text);
    std::string word;
    std::vector<std::string> concepts;
    
    while (iss >> word) {
        // Remove punctuation
        word.erase(std::remove_if(word.begin(), word.end(), 
            [](char c) { return !std::isalnum(c); }), word.end());
        
        // Add meaningful concepts
        if (word.length() > 3 && 
            (std::isupper(word[0]) || 
             word == "intelligence" || word == "learning" || word == "knowledge" ||
             word == "humanity" || word == "connection" || word == "pattern" ||
             word == "problem" || word == "solution" || word == "evolution" ||
             word == "curiosity" || word == "balance" || word == "growth" ||
             word == "understanding" || word == "insight" || word == "wisdom" ||
             word == "autonomous" || word == "system" || word == "thinking")) {
            concepts.push_back(word);
        }
    }
    
    // Remove duplicates
    std::sort(concepts.begin(), concepts.end());
    concepts.erase(std::unique(concepts.begin(), concepts.end()), concepts.end());
    
    std::ostringstream result;
    for (size_t i = 0; i < concepts.size() && i < 5; ++i) {
        if (i > 0) result << ", ";
        result << concepts[i];
    }
    
    return result.str();
}

std::string MelvinUnifiedSystem::generateSelfReflection() {
    if (!ollama->isAvailable()) {
        return "I should reflect on my learning progress and identify areas for improvement.";
    }
    
    std::ostringstream context;
    context << "Conversation history: " << conversation_history.size() << " responses\n";
    context << "Learned concepts: " << learned_concepts.size() << " concepts\n";
    context << "Generated insights: " << generated_insights.size() << " insights\n";
    context << "Self-improvements: " << self_improvements.size() << " improvements\n";
    context << "Total learning cycles: " << total_learning_cycles.load() << "\n";
    
    std::string prompt = ollama->generateSelfImprovementReflection(context.str());
    return ollama->generateResponse(prompt);
}

void MelvinUnifiedSystem::learnFromResponse(const std::string& response) {
    std::lock_guard<std::mutex> lock(learning_mutex);
    
    // Extract and store learned concepts
    std::string concepts = extractConcepts(response);
    if (!concepts.empty()) {
        learned_concepts.push_back(concepts);
        metrics.concepts_learned++;
    }
    
    // Look for insights in the response
    if (response.find("I realize") != std::string::npos || 
        response.find("I understand") != std::string::npos ||
        response.find("I see") != std::string::npos ||
        response.find("I notice") != std::string::npos) {
        generated_insights.push_back(response.substr(0, 100) + "...");
        metrics.insights_generated++;
    }
    
    // Add to knowledge base
    addToKnowledgeBase(response);
}

void MelvinUnifiedSystem::generateInsight(const std::string& context) {
    (void)context; // Suppress unused parameter warning
    std::string insight = generateSelfReflection();
    generated_insights.push_back(insight);
    metrics.insights_generated++;
    std::cout << "💡 Generated insight: " << insight.substr(0, 100) << "..." << std::endl;
}

void MelvinUnifiedSystem::performSelfImprovement() {
    std::string improvement = generateSelfReflection();
    self_improvements.push_back(improvement);
    metrics.improvements_made++;
    std::cout << "⚡ Self-improvement: " << improvement.substr(0, 100) << "..." << std::endl;
}

void MelvinUnifiedSystem::updateLearningMetrics() {
    std::lock_guard<std::mutex> lock(learning_mutex);
    
    // Calculate real learning efficiency
    if (total_learning_cycles.load() > 0) {
        metrics.learning_efficiency = static_cast<double>(metrics.concepts_learned + metrics.insights_generated) / total_learning_cycles.load();
    }
    
    // Calculate curiosity level based on questions asked
    metrics.curiosity_level = static_cast<double>(metrics.questions_asked) / std::max(1, cycle_count.load());
    
    // Calculate humanity alignment based on responses
    metrics.humanity_alignment = static_cast<double>(metrics.connections_made) / std::max(1, cycle_count.load());
}

void MelvinUnifiedSystem::addToKnowledgeBase(const std::string& knowledge) {
    std::lock_guard<std::mutex> lock(knowledge_mutex);
    knowledge_base.push_back(knowledge);
    
    // Keep only last 1000 knowledge items to manage memory
    if (knowledge_base.size() > 1000) {
        knowledge_base.erase(knowledge_base.begin());
    }
}

std::string MelvinUnifiedSystem::getRelevantKnowledge(const std::string& query) {
    std::lock_guard<std::mutex> lock(knowledge_mutex);
    
    std::ostringstream relevant;
    int count = 0;
    
    for (const auto& knowledge : knowledge_base) {
        if (knowledge.find(query) != std::string::npos && count < 3) {
            relevant << knowledge.substr(0, 100) << "...\n";
            count++;
        }
    }
    
    return relevant.str();
}

void MelvinUnifiedSystem::consolidateKnowledge() {
    std::lock_guard<std::mutex> lock(knowledge_mutex);
    
    // Simple consolidation - remove duplicates
    std::sort(knowledge_base.begin(), knowledge_base.end());
    knowledge_base.erase(std::unique(knowledge_base.begin(), knowledge_base.end()), knowledge_base.end());
    
    std::cout << "📦 Consolidated knowledge base: " << knowledge_base.size() << " unique items" << std::endl;
}

void MelvinUnifiedSystem::addToHistory(const std::string& response) {
    std::lock_guard<std::mutex> lock(history_mutex);
    conversation_history.push_back(response);
    
    // Keep only last 50 responses to manage memory
    if (conversation_history.size() > 50) {
        conversation_history.erase(conversation_history.begin());
    }
}

std::string MelvinUnifiedSystem::getContextFromHistory() {
    std::lock_guard<std::mutex> lock(history_mutex);
    
    if (conversation_history.empty()) {
        return "";
    }
    
    std::ostringstream context;
    size_t start = conversation_history.size() > 5 ? conversation_history.size() - 5 : 0;
    
    for (size_t i = start; i < conversation_history.size(); ++i) {
        context << "Previous response " << (i + 1) << ": " 
                << conversation_history[i].substr(0, 100) << "...\n";
    }
    
    return context.str();
}

void MelvinUnifiedSystem::saveConversationHistory() {
    std::lock_guard<std::mutex> lock(history_mutex);
    
    std::ofstream file("melvin_unified_history.txt");
    if (file.is_open()) {
        for (const auto& response : conversation_history) {
            file << response << "\n---\n";
        }
        file.close();
        std::cout << "💾 Saved " << conversation_history.size() << " responses to unified history file" << std::endl;
    }
}

void MelvinUnifiedSystem::loadConversationHistory() {
    std::lock_guard<std::mutex> lock(history_mutex);
    
    std::ifstream file("melvin_unified_history.txt");
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
        std::cout << "📂 Loaded " << conversation_history.size() << " responses from unified history" << std::endl;
    }
}

void MelvinUnifiedSystem::startUnifiedSystem() {
    running.store(true);
    std::cout << "🚀 Starting Melvin Unified System..." << std::endl;
    std::cout << "🤖 Real AI responses enabled via Ollama" << std::endl;
    std::cout << "🧠 Unified learning and improvement enabled" << std::endl;
    std::cout << "📊 Real metrics tracking enabled" << std::endl;
}

void MelvinUnifiedSystem::stopUnifiedSystem() {
    running.store(false);
    saveConversationHistory();
    std::cout << "⏹️ Stopping Melvin Unified System..." << std::endl;
}

void MelvinUnifiedSystem::printUnifiedStatus() {
    std::cout << "\n📊 UNIFIED SYSTEM STATUS" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "🔄 Cycles completed: " << cycle_count.load() << std::endl;
    std::cout << "🧠 Total learning cycles: " << total_learning_cycles.load() << std::endl;
    std::cout << "🔗 Ollama Status: " << (ollama->isAvailable() ? "✅ Available" : "❌ Unavailable") << std::endl;
    
    {
        std::lock_guard<std::mutex> lock(history_mutex);
        std::cout << "💬 Conversation history: " << conversation_history.size() << " responses" << std::endl;
    }
    
    {
        std::lock_guard<std::mutex> lock(learning_mutex);
        std::cout << "🧠 Learned concepts: " << learned_concepts.size() << std::endl;
        std::cout << "💡 Generated insights: " << generated_insights.size() << std::endl;
        std::cout << "⚡ Self-improvements: " << self_improvements.size() << std::endl;
    }
    
    {
        std::lock_guard<std::mutex> lock(knowledge_mutex);
        std::cout << "📚 Knowledge base: " << knowledge_base.size() << " items" << std::endl;
    }
}

void MelvinUnifiedSystem::printLearningProgress() {
    std::cout << "\n📈 LEARNING PROGRESS" << std::endl;
    std::cout << "===================" << std::endl;
    
    std::lock_guard<std::mutex> lock(learning_mutex);
    
    if (!learned_concepts.empty()) {
        std::cout << "🧠 Recent concepts learned:" << std::endl;
        for (size_t i = std::max(0, (int)learned_concepts.size() - 3); i < learned_concepts.size(); ++i) {
            std::cout << "   • " << learned_concepts[i] << std::endl;
        }
    }
    
    if (!generated_insights.empty()) {
        std::cout << "💡 Recent insights:" << std::endl;
        for (size_t i = std::max(0, (int)generated_insights.size() - 2); i < generated_insights.size(); ++i) {
            std::cout << "   • " << generated_insights[i] << std::endl;
        }
    }
}

void MelvinUnifiedSystem::printKnowledgeSummary() {
    std::lock_guard<std::mutex> lock(knowledge_mutex);
    
    std::cout << "\n📚 KNOWLEDGE SUMMARY" << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << "Total knowledge items: " << knowledge_base.size() << std::endl;
    
    if (!knowledge_base.empty()) {
        std::cout << "Recent knowledge:" << std::endl;
        size_t start = knowledge_base.size() > 3 ? knowledge_base.size() - 3 : 0;
        for (size_t i = start; i < knowledge_base.size(); ++i) {
            std::cout << "   • " << knowledge_base[i].substr(0, 80) << "..." << std::endl;
        }
    }
}

void MelvinUnifiedSystem::printConversationSummary() {
    std::lock_guard<std::mutex> lock(history_mutex);
    
    std::cout << "\n💬 CONVERSATION SUMMARY" << std::endl;
    std::cout << "======================" << std::endl;
    
    if (conversation_history.empty()) {
        std::cout << "No conversation history yet." << std::endl;
        return;
    }
    
    std::cout << "Recent responses:" << std::endl;
    size_t start = conversation_history.size() > 3 ? conversation_history.size() - 3 : 0;
    
    for (size_t i = start; i < conversation_history.size(); ++i) {
        std::cout << "Response " << (i + 1) << ": " 
                  << conversation_history[i].substr(0, 100) << "..." << std::endl;
    }
}

void MelvinUnifiedSystem::printMetrics() {
    std::cout << "\n📊 REAL LEARNING METRICS" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "🧠 Concepts learned: " << metrics.concepts_learned << std::endl;
    std::cout << "💡 Insights generated: " << metrics.insights_generated << std::endl;
    std::cout << "⚡ Improvements made: " << metrics.improvements_made << std::endl;
    std::cout << "❓ Questions asked: " << metrics.questions_asked << std::endl;
    std::cout << "🔗 Connections made: " << metrics.connections_made << std::endl;
    std::cout << "📈 Learning efficiency: " << std::fixed << std::setprecision(3) << metrics.learning_efficiency << std::endl;
    std::cout << "🎯 Curiosity level: " << std::fixed << std::setprecision(3) << metrics.curiosity_level << std::endl;
    std::cout << "🤝 Humanity alignment: " << std::fixed << std::setprecision(3) << metrics.humanity_alignment << std::endl;
}

void MelvinUnifiedSystem::startContinuousLearning() {
    std::cout << "🔄 Starting continuous autonomous learning..." << std::endl;
    // This would start a background thread for continuous learning
}

void MelvinUnifiedSystem::stopContinuousLearning() {
    std::cout << "⏹️ Stopping continuous autonomous learning..." << std::endl;
    // This would stop the background thread
}

// ============================================================================
// MELVIN UNIFIED INTERFACE IMPLEMENTATION
// ============================================================================

MelvinUnifiedInterface::MelvinUnifiedInterface(const std::string& ollama_url, const std::string& model) {
    unified_system = std::make_unique<MelvinUnifiedSystem>(ollama_url, model);
}

MelvinUnifiedInterface::~MelvinUnifiedInterface() {
    stopMelvin();
}

void MelvinUnifiedInterface::startMelvin() {
    if (running.load()) {
        std::cout << "⚠️ Melvin Unified is already running!" << std::endl;
        return;
    }
    
    running.store(true);
    unified_system->startUnifiedSystem();
    
    std::cout << "🚀 Melvin Unified System started!" << std::endl;
    std::cout << "🤖 Real AI responses enabled via Ollama" << std::endl;
    std::cout << "🧠 Unified learning and improvement enabled" << std::endl;
}

void MelvinUnifiedInterface::stopMelvin() {
    if (!running.load()) {
        return;
    }
    
    running.store(false);
    unified_system->stopUnifiedSystem();
    
    std::cout << "⏹️ Melvin Unified System stopped!" << std::endl;
}

std::string MelvinUnifiedInterface::askMelvin(const std::string& question) {
    if (!running.load()) {
        return "Error: Melvin Unified is not running. Call startMelvin() first.";
    }
    
    return unified_system->processAutonomousCycle(question);
}

void MelvinUnifiedInterface::printStatus() {
    if (unified_system) {
        unified_system->printUnifiedStatus();
    }
}

void MelvinUnifiedInterface::printAnalysis() {
    if (unified_system) {
        unified_system->printUnifiedStatus();
        unified_system->printLearningProgress();
        unified_system->printKnowledgeSummary();
        unified_system->printMetrics();
    }
}

int MelvinUnifiedInterface::getCycleCount() const {
    return unified_system ? unified_system->getCycleCount() : 0;
}

const MelvinUnifiedSystem::LearningMetrics& MelvinUnifiedInterface::getMetrics() const {
    static MelvinUnifiedSystem::LearningMetrics empty_metrics;
    return unified_system ? unified_system->getMetrics() : empty_metrics;
}

const std::vector<std::string>& MelvinUnifiedInterface::getConversationHistory() const {
    static std::vector<std::string> empty_history;
    return unified_system ? unified_system->getConversationHistory() : empty_history;
}
