/*
 * Melvin Driver-Guided Word Connection System
 * 
 * Combines:
 * 1. Driver-guided questioning (varied question types based on drivers)
 * 2. Real Ollama answers (not simulated)
 * 3. Word connection analysis (connect words that appear together)
 * 4. Knowledge persistence (save everything)
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

// Driver System - Melvin's Internal Motivations
struct DriverState {
    double survival;      // 0.0-1.0: Avoid harm, seek safety
    double curiosity;     // 0.0-1.0: Explore new knowledge
    double efficiency;    // 0.0-1.0: Optimize, avoid waste
    double social;        // 0.0-1.0: Connect with others, be helpful
    double consistency;   // 0.0-1.0: Maintain stability, avoid contradictions
    
    DriverState() : survival(0.7), curiosity(0.8), efficiency(0.6), social(0.5), consistency(0.7) {}
    
    void updateBasedOnExperience(const std::string& experience_type, bool positive) {
        double adjustment = positive ? 0.1 : -0.1;
        
        if (experience_type == "danger") {
            survival = std::clamp(survival + adjustment, 0.0, 1.0);
        } else if (experience_type == "discovery") {
            curiosity = std::clamp(curiosity + adjustment, 0.0, 1.0);
        } else if (experience_type == "waste") {
            efficiency = std::clamp(efficiency + adjustment, 0.0, 1.0);
        } else if (experience_type == "connection") {
            social = std::clamp(social + adjustment, 0.0, 1.0);
        } else if (experience_type == "contradiction") {
            consistency = std::clamp(consistency + adjustment, 0.0, 1.0);
        }
    }
    
    std::string getDominantDriver() const {
        std::map<std::string, double> drivers = {
            {"survival", survival},
            {"curiosity", curiosity},
            {"efficiency", efficiency},
            {"social", social},
            {"consistency", consistency}
        };
        
        return std::max_element(drivers.begin(), drivers.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    }
};

// Word connection storage
struct WordConnection {
    std::string word1;
    std::string word2;
    int count;
    std::string context; // The answer where they appeared together
    std::string driver_when_created; // Which driver was active when this connection was made
    
    WordConnection() : count(0) {}
    
    WordConnection(const std::string& w1, const std::string& w2, const std::string& ctx, const std::string& driver) 
        : word1(w1), word2(w2), count(1), context(ctx), driver_when_created(driver) {}
};

// Melvin's Driver-Guided Word Connection Brain
class MelvinDriverWordBrain {
private:
    std::map<std::string, std::map<std::string, WordConnection>> connections;
    DriverState drivers;
    std::mt19937 rng;
    std::string ollama_url = "http://localhost:11434/api/generate";
    std::vector<std::string> questions_asked;
    std::vector<std::string> answers_received;
    
    // Driver-guided question templates
    std::map<std::string, std::vector<std::string>> driver_questions = {
        {"survival", {
            "What dangers does {} pose?",
            "How can {} be used safely?",
            "What safety measures are needed for {}?",
            "What could go wrong with {}?",
            "How do I protect myself from {}?"
        }},
        {"curiosity", {
            "What is {}?",
            "How does {} work?",
            "What are the components of {}?",
            "What is similar to {}?",
            "What is different about {}?",
            "What can {} do?",
            "Where does {} come from?",
            "When was {} invented?",
            "Why is {} important?",
            "How is {} made?"
        }},
        {"efficiency", {
            "How can {} be optimized?",
            "What is the most efficient way to use {}?",
            "How does {} save time or resources?",
            "What are the costs of {}?",
            "How can {} be improved?"
        }},
        {"social", {
            "How do people use {}?",
            "What do others think about {}?",
            "How does {} help people?",
            "Who benefits from {}?",
            "How does {} connect people?"
        }},
        {"consistency", {
            "How does {} relate to what I already know?",
            "Does {} contradict anything I know?",
            "How does {} fit into my understanding?",
            "What patterns does {} follow?",
            "How is {} consistent with other concepts?"
        }}
    };
    
    // Concepts to explore
    std::vector<std::string> concepts = {
        "car", "bicycle", "motorcycle", "engine", "wheels", "transportation", 
        "vehicle", "fuel", "driver", "speed", "road", "traffic", "safety",
        "electricity", "battery", "motor", "brake", "steering", "acceleration"
    };

public:
    MelvinDriverWordBrain() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        std::cout << "🧠 Melvin Driver-Guided Word Connection Brain Initialized" << std::endl;
        std::cout << "🎯 Drivers guide questions + Real Ollama answers + Word connections!" << std::endl;
    }
    
    void runDriverWordExploration(int minutes = 5) {
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + std::chrono::minutes(minutes);
        
        std::cout << "\n⏰ Starting " << minutes << "-minute driver-guided word exploration..." << std::endl;
        std::cout << "🎯 Melvin will ask driver-guided questions, get Ollama answers, and connect words!" << std::endl;
        
        int question_count = 0;
        
        while (std::chrono::steady_clock::now() < end_time) {
            // Step 1: Check current driver state
            std::string dominant_driver = drivers.getDominantDriver();
            std::cout << "\n🎭 Current dominant driver: " << dominant_driver 
                     << " (curiosity: " << std::fixed << std::setprecision(2) << drivers.curiosity 
                     << ", efficiency: " << drivers.efficiency << ")" << std::endl;
            
            // Step 2: Generate driver-guided question
            std::string question = generateDriverGuidedQuestion(dominant_driver);
            std::cout << "🤔 Melvin's " << dominant_driver << "-driven question: " << question << std::endl;
            
            // Step 3: Get real answer from Ollama
            std::string answer = askOllama(question);
            std::cout << "💡 Ollama answers: " << answer << std::endl;
            
            // Step 4: Store question and answer
            this->questions_asked.push_back(question);
            this->answers_received.push_back(answer);
            
            // Step 5: Connect words from the answer
            connectWordsFromAnswer(answer, dominant_driver);
            
            // Step 6: Update drivers based on experience
            updateDriversFromExperience(question, answer);
            
            question_count++;
            
            // Show progress
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            std::cout << "⏱️ Time: " << elapsed_seconds << "s | Questions: " << question_count << std::endl;
            
            // Driver-guided thinking pause
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        }
        
        showResults();
        saveConnections();
    }
    
    void showResults() {
        std::cout << "\n🧠 MELVIN'S DRIVER-GUIDED WORD CONNECTION RESULTS" << std::endl;
        std::cout << "================================================" << std::endl;
        
        std::cout << "\n📊 Statistics:" << std::endl;
        std::cout << "  Questions asked: " << questions_asked.size() << std::endl;
        std::cout << "  Answers received: " << answers_received.size() << std::endl;
        
        // Count total connections
        int total_connections = 0;
        for (const auto& word1 : connections) {
            total_connections += word1.second.size();
        }
        std::cout << "  Total word connections: " << total_connections << std::endl;
        
        std::cout << "\n🎭 Final Driver States:" << std::endl;
        std::cout << "  Survival: " << std::fixed << std::setprecision(2) << drivers.survival << std::endl;
        std::cout << "  Curiosity: " << drivers.curiosity << std::endl;
        std::cout << "  Efficiency: " << drivers.efficiency << std::endl;
        std::cout << "  Social: " << drivers.social << std::endl;
        std::cout << "  Consistency: " << drivers.consistency << std::endl;
        
        std::cout << "\n🔗 Top Word Connections:" << std::endl;
        showTopConnections();
        
        std::cout << "\n🎯 Driver-Guided Learning Patterns:" << std::endl;
        analyzeDriverPatterns();
    }

private:
    std::string generateDriverGuidedQuestion(const std::string& dominant_driver) {
        // Pick a random concept
        std::uniform_int_distribution<int> concept_dist(0, concepts.size() - 1);
        std::string target_concept = concepts[concept_dist(rng)];
        
        // Get driver-specific questions
        const auto& questions = driver_questions[dominant_driver];
        std::uniform_int_distribution<int> question_dist(0, questions.size() - 1);
        std::string template_str = questions[question_dist(rng)];
        
        // Replace {} with concept
        size_t pos = template_str.find("{}");
        if (pos != std::string::npos) {
            template_str.replace(pos, 2, target_concept);
        }
        
        return template_str;
    }
    
    std::string askOllama(const std::string& question) {
        CURL* curl;
        CURLcode res;
        std::string response;
        
        curl = curl_easy_init();
        if (curl) {
            // Prepare simple JSON request
            std::string json_request = "{\"model\":\"llama3.2:latest\",\"prompt\":\"" + question + "\",\"stream\":false}";
            
            // Set up curl
            curl_easy_setopt(curl, CURLOPT_URL, ollama_url.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_request.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_request.length());
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_slist_append(nullptr, "Content-Type: application/json"));
            
            // Response callback
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
            
            // Perform request
            res = curl_easy_perform(curl);
            
            if (res != CURLE_OK) {
                std::cout << "❌ Ollama request failed: " << curl_easy_strerror(res) << std::endl;
                return "Error: Could not get answer from Ollama";
            }
            
            curl_easy_cleanup(curl);
        }
        
        // Parse response
        return parseOllamaResponse(response);
    }
    
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* response) {
        size_t total_size = size * nmemb;
        response->append((char*)contents, total_size);
        return total_size;
    }
    
    std::string parseOllamaResponse(const std::string& response) {
        // Simple JSON parsing - find "response" field
        size_t response_start = response.find("\"response\":\"");
        if (response_start != std::string::npos) {
            response_start += 12; // Skip "response":"
            size_t response_end = response.find("\"", response_start);
            if (response_end != std::string::npos) {
                return response.substr(response_start, response_end - response_start);
            }
        }
        
        return "Error: Could not parse Ollama response";
    }
    
    void connectWordsFromAnswer(const std::string& answer, const std::string& driver) {
        // Extract words from answer
        std::vector<std::string> words = extractWords(answer);
        
        std::cout << "🔗 Melvin connects words from " << driver << "-driven answer..." << std::endl;
        
        // Connect each word with every other word in the same answer
        for (size_t i = 0; i < words.size(); i++) {
            for (size_t j = i + 1; j < words.size(); j++) {
                std::string word1 = words[i];
                std::string word2 = words[j];
                
                // Ensure consistent ordering (alphabetical)
                if (word1 > word2) {
                    std::swap(word1, word2);
                }
                
                // Add or update connection
                if (connections[word1].find(word2) == connections[word1].end()) {
                    connections[word1][word2] = WordConnection(word1, word2, answer, driver);
                } else {
                    connections[word1][word2].count++;
                }
                
                std::cout << "  📎 " << word1 << " ↔ " << word2 << " (" << driver << ")" << std::endl;
            }
        }
        
        std::cout << "  ✅ Found " << words.size() << " words, created " << (words.size() * (words.size() - 1) / 2) << " connections" << std::endl;
    }
    
    void updateDriversFromExperience(const std::string& question, const std::string& answer) {
        // Update drivers based on the type of question and answer
        
        if (question.find("dangers") != std::string::npos || question.find("safety") != std::string::npos) {
            drivers.updateBasedOnExperience("danger", true);
        } else if (question.find("optimize") != std::string::npos || question.find("efficient") != std::string::npos) {
            drivers.updateBasedOnExperience("waste", false); // Learning efficiency is good
        } else if (question.find("people") != std::string::npos || question.find("social") != std::string::npos) {
            drivers.updateBasedOnExperience("connection", true);
        } else if (question.find("contradict") != std::string::npos) {
            drivers.updateBasedOnExperience("contradiction", false);
        } else {
            drivers.updateBasedOnExperience("discovery", true);
        }
    }
    
    void showTopConnections() {
        // Collect all connections with their counts
        std::vector<std::pair<std::string, int>> connection_counts;
        
        for (const auto& word1 : connections) {
            for (const auto& word2 : word1.second) {
                std::string connection = word1.first + " ↔ " + word2.first;
                connection_counts.push_back({connection, word2.second.count});
            }
        }
        
        // Sort by count (highest first)
        std::sort(connection_counts.begin(), connection_counts.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Show top 10
        int count = 0;
        for (const auto& conn : connection_counts) {
            if (count >= 10) break;
            std::cout << "  " << conn.first << " (appeared together " << conn.second << " times)" << std::endl;
            count++;
        }
    }
    
    void analyzeDriverPatterns() {
        std::cout << "  🎭 Driver Evolution:" << std::endl;
        std::cout << "    - Melvin's drivers adapt based on experience" << std::endl;
        std::cout << "    - Current dominant: " << drivers.getDominantDriver() << std::endl;
        std::cout << "    - Questions are guided by driver state" << std::endl;
        std::cout << "    - Word connections show driver influence" << std::endl;
    }
    
    std::vector<std::string> extractWords(const std::string& text) {
        std::vector<std::string> words;
        std::istringstream iss(text);
        std::string word;
        
        while (iss >> word) {
            // Clean the word
            word = cleanWord(word);
            
            // Skip short words and common words
            if (word.length() > 2 && !isCommonWord(word)) {
                words.push_back(word);
            }
        }
        
        return words;
    }
    
    std::string cleanWord(const std::string& word) {
        std::string cleaned = word;
        
        // Remove punctuation
        cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(), 
            [](char c) { return !std::isalnum(c); }), cleaned.end());
        
        // Convert to lowercase
        std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);
        
        return cleaned;
    }
    
    bool isCommonWord(const std::string& word) {
        std::set<std::string> common_words = {
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "can", "this", "that", "these", "those", "a", "an", "as", "if", "when",
            "where", "why", "how", "what", "who", "which", "it", "its", "they", "them",
            "their", "there", "here", "very", "much", "more", "most", "some", "any",
            "all", "each", "every", "other", "another", "such", "so", "too", "also"
        };
        
        return common_words.find(word) != common_words.end();
    }
    
    void saveConnections() {
        std::ofstream file("melvin_driver_word_connections.json");
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
            file << "    \"final_drivers\": {\n";
            file << "      \"survival\": " << drivers.survival << ",\n";
            file << "      \"curiosity\": " << drivers.curiosity << ",\n";
            file << "      \"efficiency\": " << drivers.efficiency << ",\n";
            file << "      \"social\": " << drivers.social << ",\n";
            file << "      \"consistency\": " << drivers.consistency << "\n";
            file << "    }\n";
            file << "  }\n";
            file << "}\n";
            
            file.close();
            std::cout << "\n💾 Driver-guided word connections saved to melvin_driver_word_connections.json" << std::endl;
        }
    }
    
    int getTotalConnections() {
        int total = 0;
        for (const auto& word1 : connections) {
            total += word1.second.size();
        }
        return total;
    }
};

int main() {
    std::cout << "🧠 MELVIN DRIVER-GUIDED WORD CONNECTION BRAIN" << std::endl;
    std::cout << "=============================================" << std::endl;
    std::cout << "🎯 Drivers guide questions + Real Ollama answers + Word connections!" << std::endl;
    
    MelvinDriverWordBrain melvin;
    melvin.runDriverWordExploration(3); // 3 minutes for demo
    
    std::cout << "\n✅ Driver-guided word exploration completed!" << std::endl;
    std::cout << "🧠 Melvin's drivers evolved and his word connections are saved!" << std::endl;
    
    return 0;
}
