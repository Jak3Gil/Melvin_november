/*
 * Melvin Feedback Loop Engine
 * 
 * Implements the user's feedback loop prompt:
 * 1. Driver-guided questioning (curiosity, stability, reinforcement)
 * 2. Connection-driven refinement (use existing connections to generate new questions)
 * 3. Autonomous feedback loops (answers → connections → new questions)
 * 4. Dynamic driver evolution (drivers adapt based on experience)
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <set>
#include <sstream>
#include <fstream>
#include <curl/curl.h>
#include <chrono>
#include <thread>
#include <random>

// Enhanced Driver System with Feedback Loop Evolution
struct FeedbackDriverState {
    double curiosity;     // 0.0-1.0: Exploratory, open-ended questions
    double stability;     // 0.0-1.0: Clarifying, contradiction-checking questions  
    double reinforcement; // 0.0-1.0: Summarizing, practical "how to use" questions
    
    FeedbackDriverState() : curiosity(0.8), stability(0.6), reinforcement(0.5) {}
    
    void evolveBasedOnExperience(const std::string& experience_type, bool positive) {
        double adjustment = positive ? 0.05 : -0.05;
        
        if (experience_type == "novelty") {
            curiosity = std::clamp(curiosity + adjustment, 0.0, 1.0);
        } else if (experience_type == "contradiction_resolved") {
            stability = std::clamp(stability + adjustment, 0.0, 1.0);
        } else if (experience_type == "connection_confirmed") {
            reinforcement = std::clamp(reinforcement + adjustment, 0.0, 1.0);
        }
    }
    
    std::string getDominantDriver() const {
        if (curiosity >= stability && curiosity >= reinforcement) return "curiosity";
        if (stability >= reinforcement) return "stability";
        return "reinforcement";
    }
    
    double getDriverStrength(const std::string& driver) const {
        if (driver == "curiosity") return curiosity;
        if (driver == "stability") return stability;
        if (driver == "reinforcement") return reinforcement;
        return 0.0;
    }
};

// Enhanced Word Connection with Feedback Context
struct FeedbackWordConnection {
    std::string word1;
    std::string word2;
    int count;
    std::string context;
    std::string driver_when_created;
    std::vector<std::string> question_sources; // Which questions led to this connection
    double strength; // Calculated strength based on co-occurrence and context
    
    FeedbackWordConnection() : count(0), strength(0.0) {}
    
    FeedbackWordConnection(const std::string& w1, const std::string& w2, const std::string& ctx, 
                          const std::string& driver, const std::string& question) 
        : word1(w1), word2(w2), count(1), context(ctx), driver_when_created(driver), strength(1.0) {
        question_sources.push_back(question);
    }
    
    void updateStrength() {
        // Strength based on count, context length, and question diversity
        strength = count * 0.5 + (context.length() / 100.0) + (question_sources.size() * 0.1);
    }
};

// Melvin's Feedback Loop Engine
class MelvinFeedbackLoop {
private:
    std::map<std::string, std::map<std::string, FeedbackWordConnection>> connections;
    FeedbackDriverState drivers;
    std::mt19937 rng;
    std::string ollama_url = "http://localhost:11434/api/generate";
    
    std::vector<std::string> questions_asked;
    std::vector<std::string> answers_received;
    std::vector<std::string> concepts_explored;
    
    // Driver-guided question templates
    std::map<std::string, std::vector<std::string>> driver_questions = {
        {"curiosity", {
            "What is {}?",
            "How does {} work?",
            "Why is {} important?",
            "What makes {} unique?",
            "How is {} different from other things?",
            "What are the components of {}?",
            "Where does {} come from?",
            "When was {} invented?",
            "What can {} do?",
            "How is {} made?"
        }},
        {"stability", {
            "How does {} relate to what I already know?",
            "Does {} contradict anything I know?",
            "What patterns does {} follow?",
            "How is {} consistent with other concepts?",
            "What are the similarities between {} and other things?",
            "How does {} fit into my understanding?",
            "What are the rules that govern {}?",
            "How does {} connect to my existing knowledge?"
        }},
        {"reinforcement", {
            "How can {} be used practically?",
            "What are the applications of {}?",
            "How do people use {} in real life?",
            "What are the benefits of {}?",
            "How can {} be improved?",
            "What are the best practices for {}?",
            "How does {} solve problems?",
            "What are the limitations of {}?"
        }}
    };
    
    // Concepts to explore (expands over time)
    std::set<std::string> concepts = {
        "car", "bicycle", "motorcycle", "engine", "wheels", "transportation", 
        "vehicle", "fuel", "driver", "speed", "road", "traffic", "safety",
        "electricity", "battery", "motor", "brake", "steering", "acceleration"
    };

public:
    MelvinFeedbackLoop() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        std::cout << "🧠 Melvin Feedback Loop Engine Initialized" << std::endl;
        std::cout << "🔄 Continuous evolution through driver-guided questioning + connection-driven refinement!" << std::endl;
    }
    
    void runFeedbackLoop(int cycles = 10) {
        std::cout << "\n⏰ Starting " << cycles << " feedback loop cycles..." << std::endl;
        std::cout << "🔄 Each cycle: questions → answers → connections → new questions" << std::endl;
        
        for (int cycle = 1; cycle <= cycles; cycle++) {
            std::cout << "\n🔄 === FEEDBACK LOOP CYCLE " << cycle << " ===" << std::endl;
            
            // Step 1: Generate questions based on driver state + existing connections
            std::vector<std::string> questions = generateFeedbackQuestions();
            
            // Step 2: Ask questions to Ollama and collect answers
            std::vector<std::string> answers = processQuestions(questions);
            
            // Step 3: Extract words and build connections
            buildConnectionsFromAnswers(questions, answers);
            
            // Step 4: Evolve drivers based on experience
            evolveDriversFromCycle(questions, answers);
            
            // Step 5: Expand concept vocabulary
            expandConceptVocabulary(answers);
            
            std::cout << "✅ Cycle " << cycle << " completed. Total connections: " << getTotalConnections() << std::endl;
            
            // Brief pause between cycles
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        
        showFeedbackResults();
        saveFeedbackData();
    }
    
    void showFeedbackResults() {
        std::cout << "\n🧠 MELVIN FEEDBACK LOOP RESULTS" << std::endl;
        std::cout << "===============================" << std::endl;
        
        std::cout << "\n📊 Statistics:" << std::endl;
        std::cout << "  Questions asked: " << questions_asked.size() << std::endl;
        std::cout << "  Answers received: " << answers_received.size() << std::endl;
        std::cout << "  Concepts explored: " << concepts.size() << std::endl;
        std::cout << "  Total connections: " << getTotalConnections() << std::endl;
        
        std::cout << "\n🎭 Final Driver States:" << std::endl;
        std::cout << "  Curiosity: " << std::fixed << std::setprecision(2) << drivers.curiosity << std::endl;
        std::cout << "  Stability: " << drivers.stability << std::endl;
        std::cout << "  Reinforcement: " << drivers.reinforcement << std::endl;
        std::cout << "  Dominant: " << drivers.getDominantDriver() << std::endl;
        
        std::cout << "\n🔗 Strongest Word Connections:" << std::endl;
        showStrongestConnections();
        
        std::cout << "\n🔄 Feedback Loop Analysis:" << std::endl;
        analyzeFeedbackPatterns();
    }

private:
    std::vector<std::string> generateFeedbackQuestions() {
        std::vector<std::string> questions;
        std::string dominant_driver = drivers.getDominantDriver();
        
        std::cout << "🎭 Current dominant driver: " << dominant_driver 
                 << " (curiosity: " << std::fixed << std::setprecision(2) << drivers.curiosity 
                 << ", stability: " << drivers.stability 
                 << ", reinforcement: " << drivers.reinforcement << ")" << std::endl;
        
        // Generate 3-5 questions per cycle
        int num_questions = 3 + (rng() % 3);
        
        for (int i = 0; i < num_questions; i++) {
            std::string question;
            
            // 70% driver-guided questions, 30% connection-driven questions
            if (rng() % 10 < 7) {
                question = generateDriverGuidedQuestion(dominant_driver);
            } else {
                question = generateConnectionDrivenQuestion();
            }
            
            questions.push_back(question);
        }
        
        return questions;
    }
    
    std::string generateDriverGuidedQuestion(const std::string& driver) {
        // Pick a random concept
        std::vector<std::string> concept_list(concepts.begin(), concepts.end());
        std::uniform_int_distribution<int> concept_dist(0, concept_list.size() - 1);
        std::string target_concept = concept_list[concept_dist(rng)];
        
        // Get driver-specific questions
        const auto& questions = driver_questions[driver];
        std::uniform_int_distribution<int> question_dist(0, questions.size() - 1);
        std::string template_str = questions[question_dist(rng)];
        
        // Replace {} with concept
        size_t pos = template_str.find("{}");
        if (pos != std::string::npos) {
            template_str.replace(pos, 2, target_concept);
        }
        
        return template_str;
    }
    
    std::string generateConnectionDrivenQuestion() {
        // Find strong connections and generate questions about them
        std::vector<std::pair<std::string, double>> strong_connections;
        
        for (const auto& word1 : connections) {
            for (const auto& word2 : word1.second) {
                if (word2.second.count > 2) { // Only strong connections
                    std::string connection = word1.first + " ↔ " + word2.first;
                    strong_connections.push_back({connection, word2.second.strength});
                }
            }
        }
        
        if (strong_connections.empty()) {
            return generateDriverGuidedQuestion("curiosity");
        }
        
        // Sort by strength
        std::sort(strong_connections.begin(), strong_connections.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Pick a random strong connection
        std::uniform_int_distribution<int> conn_dist(0, std::min(4, (int)strong_connections.size() - 1));
        std::string connection = strong_connections[conn_dist(rng)].first;
        
        // Generate question about the connection
        std::vector<std::string> connection_questions = {
            "Why are " + connection + " always linked together?",
            "What is the relationship between " + connection + "?",
            "How do " + connection + " work together?",
            "What makes " + connection + " connected?",
            "Why do " + connection + " appear together so often?"
        };
        
        std::uniform_int_distribution<int> q_dist(0, connection_questions.size() - 1);
        return connection_questions[q_dist(rng)];
    }
    
    std::vector<std::string> processQuestions(const std::vector<std::string>& questions) {
        std::vector<std::string> answers;
        
        for (const auto& question : questions) {
            std::cout << "🤔 Melvin asks: " << question << std::endl;
            
            std::string answer = askOllama(question);
            std::cout << "💡 Ollama answers: " << answer << std::endl;
            
            questions_asked.push_back(question);
            answers_received.push_back(answer);
            answers.push_back(answer);
        }
        
        return answers;
    }
    
    void buildConnectionsFromAnswers(const std::vector<std::string>& questions, 
                                   const std::vector<std::string>& answers) {
        std::cout << "🔗 Building connections from answers..." << std::endl;
        
        for (size_t i = 0; i < questions.size() && i < answers.size(); i++) {
            const std::string& question = questions[i];
            const std::string& answer = answers[i];
            std::string driver = extractDriverFromQuestion(question);
            
            // Extract words from answer
            std::vector<std::string> words = extractWords(answer);
            
            // Connect each word with every other word in the same answer
            for (size_t j = 0; j < words.size(); j++) {
                for (size_t k = j + 1; k < words.size(); k++) {
                    std::string word1 = words[j];
                    std::string word2 = words[k];
                    
                    // Ensure consistent ordering
                    if (word1 > word2) {
                        std::swap(word1, word2);
                    }
                    
                    // Add or update connection
                    if (connections[word1].find(word2) == connections[word1].end()) {
                        connections[word1][word2] = FeedbackWordConnection(word1, word2, answer, driver, question);
                    } else {
                        connections[word1][word2].count++;
                        connections[word1][word2].question_sources.push_back(question);
                        connections[word1][word2].updateStrength();
                    }
                    
                    std::cout << "  📎 " << word1 << " ↔ " << word2 << " (" << driver << ")" << std::endl;
                }
            }
            
            std::cout << "  ✅ Found " << words.size() << " words, created " << (words.size() * (words.size() - 1) / 2) << " connections" << std::endl;
        }
    }
    
    void evolveDriversFromCycle(const std::vector<std::string>& questions, 
                              const std::vector<std::string>& answers) {
        std::cout << "🎭 Evolving drivers based on experience..." << std::endl;
        
        for (size_t i = 0; i < questions.size() && i < answers.size(); i++) {
            const std::string& question = questions[i];
            const std::string& answer = answers[i];
            
            // Analyze question type and answer content
            if (question.find("What is") != std::string::npos || 
                question.find("How does") != std::string::npos ||
                question.find("Why is") != std::string::npos) {
                drivers.evolveBasedOnExperience("novelty", true);
            }
            
            if (question.find("contradict") != std::string::npos ||
                question.find("consistent") != std::string::npos) {
                drivers.evolveBasedOnExperience("contradiction_resolved", true);
            }
            
            if (question.find("use") != std::string::npos ||
                question.find("practical") != std::string::npos ||
                question.find("application") != std::string::npos) {
                drivers.evolveBasedOnExperience("connection_confirmed", true);
            }
        }
        
        std::cout << "  📈 Driver evolution: curiosity=" << std::fixed << std::setprecision(2) << drivers.curiosity 
                 << ", stability=" << drivers.stability 
                 << ", reinforcement=" << drivers.reinforcement << std::endl;
    }
    
    void expandConceptVocabulary(const std::vector<std::string>& answers) {
        for (const auto& answer : answers) {
            std::vector<std::string> words = extractWords(answer);
            for (const auto& word : words) {
                if (word.length() > 3 && !isCommonWord(word)) {
                    concepts.insert(word);
                }
            }
        }
    }
    
    std::string extractDriverFromQuestion(const std::string& question) {
        if (question.find("What is") != std::string::npos || 
            question.find("How does") != std::string::npos ||
            question.find("Why is") != std::string::npos) {
            return "curiosity";
        }
        
        if (question.find("contradict") != std::string::npos ||
            question.find("consistent") != std::string::npos ||
            question.find("relate") != std::string::npos) {
            return "stability";
        }
        
        if (question.find("use") != std::string::npos ||
            question.find("practical") != std::string::npos ||
            question.find("application") != std::string::npos) {
            return "reinforcement";
        }
        
        return "curiosity"; // Default
    }
    
    void showStrongestConnections() {
        std::vector<std::pair<std::string, double>> connection_strengths;
        
        for (const auto& word1 : connections) {
            for (const auto& word2 : word1.second) {
                std::string connection = word1.first + " ↔ " + word2.first;
                connection_strengths.push_back({connection, word2.second.strength});
            }
        }
        
        // Sort by strength
        std::sort(connection_strengths.begin(), connection_strengths.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Show top 10
        for (int i = 0; i < std::min(10, (int)connection_strengths.size()); i++) {
            std::cout << "  " << connection_strengths[i].first 
                     << " (strength: " << std::fixed << std::setprecision(2) 
                     << connection_strengths[i].second << ")" << std::endl;
        }
    }
    
    void analyzeFeedbackPatterns() {
        std::cout << "  🔄 Feedback Loop Analysis:" << std::endl;
        std::cout << "    - Questions generated: " << questions_asked.size() << std::endl;
        std::cout << "    - Answers processed: " << answers_received.size() << std::endl;
        std::cout << "    - Concepts discovered: " << concepts.size() << std::endl;
        std::cout << "    - Connections built: " << getTotalConnections() << std::endl;
        std::cout << "    - Driver evolution: " << drivers.getDominantDriver() << " dominant" << std::endl;
    }
    
    // Ollama integration methods (same as before)
    std::string askOllama(const std::string& question) {
        CURL* curl;
        CURLcode res;
        std::string response;
        
        curl = curl_easy_init();
        if (curl) {
            std::string json_request = "{\"model\":\"llama3.2:latest\",\"prompt\":\"" + question + "\",\"stream\":false}";
            
            curl_easy_setopt(curl, CURLOPT_URL, ollama_url.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_request.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_request.length());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_slist_append(nullptr, "Content-Type: application/json"));
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            
            res = curl_easy_perform(curl);
            curl_easy_cleanup(curl);
        }
        
        return parseOllamaResponse(response);
    }
    
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
        size_t total_size = size * nmemb;
        response->append((char*)contents, total_size);
        return total_size;
    }
    
    std::string parseOllamaResponse(const std::string& response) {
        size_t response_start = response.find("\"response\":\"");
        if (response_start != std::string::npos) {
            response_start += 12;
            size_t response_end = response.find("\"", response_start);
            if (response_end != std::string::npos) {
                return response.substr(response_start, response_end - response_start);
            }
        }
        return "Error: Could not parse Ollama response";
    }
    
    // Word processing methods (same as before)
    std::vector<std::string> extractWords(const std::string& text) {
        std::vector<std::string> words;
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            word = cleanWord(word);
            if (word.length() > 2 && !isCommonWord(word)) {
                words.push_back(word);
            }
        }
        
        return words;
    }
    
    std::string cleanWord(const std::string& word) {
        std::string cleaned = word;
        cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(), 
            [](char c) { return !std::isalnum(c); }), cleaned.end());
        std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);
        return cleaned;
    }
    
    bool isCommonWord(const std::string& word) {
        std::set<std::string> common_words = {
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "can", "this", "that", "these", "those", "a", "an", "as", "if", "when",
            "where", "why", "how", "what", "who", "which", "it", "its", "they", "them"
        };
        return common_words.find(word) != common_words.end();
    }
    
    int getTotalConnections() {
        int total = 0;
        for (const auto& word1 : connections) {
            total += word1.second.size();
        }
        return total;
    }
    
    void saveFeedbackData() {
        std::ofstream file("melvin_feedback_loop.json");
        if (file.is_open()) {
            file << "{\n";
            file << "  \"connections\": [\n";
            
            bool first = true;
            for (const auto& word1 : connections) {
                for (const auto& word2 : word1.second) {
                    if (!first) file << ",\n";
                    file << "    {\n";
                    file << "      \"word1\": \"" << word2.second.word1 << "\",\n";
                    file << "      \"word2\": \"" << word2.second.word2 << "\",\n";
                    file << "      \"count\": " << word2.second.count << ",\n";
                    file << "      \"strength\": " << word2.second.strength << ",\n";
                    file << "      \"driver\": \"" << word2.second.driver_when_created << "\",\n";
                    file << "      \"context\": \"" << word2.second.context << "\"\n";
                    file << "    }";
                    first = false;
                }
            }
            
            file << "\n  ],\n";
            file << "  \"metadata\": {\n";
            file << "    \"total_connections\": " << getTotalConnections() << ",\n";
            file << "    \"questions_asked\": " << questions_asked.size() << ",\n";
            file << "    \"answers_received\": " << answers_received.size() << ",\n";
            file << "    \"concepts_explored\": " << concepts.size() << ",\n";
            file << "    \"final_drivers\": {\n";
            file << "      \"curiosity\": " << drivers.curiosity << ",\n";
            file << "      \"stability\": " << drivers.stability << ",\n";
            file << "      \"reinforcement\": " << drivers.reinforcement << "\n";
            file << "    }\n";
            file << "  }\n";
            file << "}\n";
            
            file.close();
            std::cout << "\n💾 Feedback loop data saved to melvin_feedback_loop.json" << std::endl;
        }
    }
};

int main() {
    std::cout << "🧠 MELVIN FEEDBACK LOOP ENGINE" << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << "🔄 Continuous evolution through autonomous feedback loops!" << std::endl;
    
    MelvinFeedbackLoop melvin;
    melvin.runFeedbackLoop(8); // 8 cycles for demo
    
    std::cout << "\n✅ Feedback loop exploration completed!" << std::endl;
    std::cout << "🧠 Melvin has evolved through autonomous questioning and connection building!" << std::endl;
    
    return 0;
}
