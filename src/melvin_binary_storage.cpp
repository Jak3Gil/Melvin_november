/*
 * Melvin Binary Storage System
 * 
 * Converts JSON storage to efficient binary format:
 * - Compact binary representation
 * - Fast read/write operations
 * - Memory efficient storage
 * - No JSON parsing overhead
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
#include <cstring>

// Binary storage structures
struct BinaryWordConnection {
    char word1[32];        // Fixed size for efficiency
    char word2[32];        // Fixed size for efficiency
    uint32_t count;        // 4 bytes
    float strength;        // 4 bytes
    char driver[16];       // Fixed size for driver name
    uint32_t context_length; // Length of context string
    // Context string follows this struct
    
    BinaryWordConnection() : count(0), strength(0.0f), context_length(0) {
        memset(word1, 0, sizeof(word1));
        memset(word2, 0, sizeof(word2));
        memset(driver, 0, sizeof(driver));
    }
    
    BinaryWordConnection(const std::string& w1, const std::string& w2, 
                        uint32_t c, float s, const std::string& d, const std::string& ctx)
        : count(c), strength(s), context_length(ctx.length()) {
        
        // Copy strings with bounds checking
        strncpy(word1, w1.c_str(), sizeof(word1) - 1);
        strncpy(word2, w2.c_str(), sizeof(word2) - 1);
        strncpy(driver, d.c_str(), sizeof(driver) - 1);
        
        // Ensure null termination
        word1[sizeof(word1) - 1] = '\0';
        word2[sizeof(word2) - 1] = '\0';
        driver[sizeof(driver) - 1] = '\0';
    }
};

struct BinaryMetadata {
    uint32_t total_connections;
    uint32_t questions_asked;
    uint32_t answers_received;
    uint32_t concepts_explored;
    float curiosity_driver;
    float stability_driver;
    float reinforcement_driver;
    uint64_t timestamp;
    
    BinaryMetadata() : total_connections(0), questions_asked(0), answers_received(0),
                      concepts_explored(0), curiosity_driver(0.0f), stability_driver(0.0f),
                      reinforcement_driver(0.0f), timestamp(0) {}
};

// Melvin's Binary Storage Brain
class MelvinBinaryStorage {
private:
    std::map<std::string, std::map<std::string, std::pair<uint32_t, float>>> connections;
    std::vector<std::string> questions_asked;
    std::vector<std::string> answers_received;
    std::set<std::string> concepts;
    
    float curiosity_driver = 0.8f;
    float stability_driver = 0.6f;
    float reinforcement_driver = 0.5f;
    
    std::mt19937 rng;
    std::string ollama_url = "http://localhost:11434/api/generate";
    
    std::string binary_file = "melvin_binary.bin";
    std::string context_file = "melvin_contexts.bin";

public:
    MelvinBinaryStorage() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        std::cout << "🧠 Melvin Binary Storage System Initialized" << std::endl;
        std::cout << "💾 Using efficient binary format instead of JSON!" << std::endl;
        
        loadBinaryData();
    }
    
    ~MelvinBinaryStorage() {
        saveBinaryData();
    }
    
    void runBinaryLearning(int cycles = 5) {
        std::cout << "\n⏰ Starting " << cycles << " binary learning cycles..." << std::endl;
        std::cout << "💾 All data will be stored in efficient binary format!" << std::endl;
        
        for (int cycle = 1; cycle <= cycles; cycle++) {
            std::cout << "\n🔄 === BINARY LEARNING CYCLE " << cycle << " ===" << std::endl;
            
            // Generate questions
            std::vector<std::string> questions = generateQuestions();
            
            // Process questions
            for (const auto& question : questions) {
                std::cout << "🤔 Melvin asks: " << question << std::endl;
                
                std::string answer = askOllama(question);
                std::cout << "💡 Ollama answers: " << answer.substr(0, 100) << "..." << std::endl;
                
                questions_asked.push_back(question);
                answers_received.push_back(answer);
                
                // Build binary connections
                buildBinaryConnections(question, answer);
            }
            
            std::cout << "✅ Cycle " << cycle << " completed. Total connections: " << getTotalConnections() << std::endl;
            
            // Save incrementally
            saveBinaryData();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        
        showBinaryResults();
    }
    
    void showBinaryResults() {
        std::cout << "\n🧠 MELVIN BINARY STORAGE RESULTS" << std::endl;
        std::cout << "================================" << std::endl;
        
        std::cout << "\n📊 Statistics:" << std::endl;
        std::cout << "  Questions asked: " << questions_asked.size() << std::endl;
        std::cout << "  Answers received: " << answers_received.size() << std::endl;
        std::cout << "  Concepts explored: " << concepts.size() << std::endl;
        std::cout << "  Total connections: " << getTotalConnections() << std::endl;
        
        std::cout << "\n💾 Storage Analysis:" << std::endl;
        analyzeStorage();
        
        std::cout << "\n🔗 Top Connections:" << std::endl;
        showTopConnections();
    }

private:
    std::vector<std::string> generateQuestions() {
        std::vector<std::string> questions;
        std::vector<std::string> concepts_list = {
            "car", "bicycle", "motorcycle", "engine", "wheels", "transportation", 
            "vehicle", "fuel", "driver", "speed", "road", "traffic", "safety",
            "electricity", "battery", "motor", "brake", "steering", "acceleration"
        };
        
        // Generate 2-3 questions per cycle
        int num_questions = 2 + (rng() % 2);
        
        for (int i = 0; i < num_questions; i++) {
            std::uniform_int_distribution<int> concept_dist(0, concepts_list.size() - 1);
            std::string concept = concepts_list[concept_dist(rng)];
            
            std::vector<std::string> question_templates = {
                "What is " + concept + "?",
                "How does " + concept + " work?",
                "Why is " + concept + " important?",
                "What are the components of " + concept + "?",
                "How is " + concept + " used?"
            };
            
            std::uniform_int_distribution<int> template_dist(0, question_templates.size() - 1);
            questions.push_back(question_templates[template_dist(rng)]);
        }
        
        return questions;
    }
    
    void buildBinaryConnections(const std::string& question, const std::string& answer) {
        std::cout << "🔗 Building binary connections..." << std::endl;
        
        // Extract words from answer
        std::vector<std::string> words = extractWords(answer);
        
        // Connect each word with every other word
        for (size_t i = 0; i < words.size(); i++) {
            for (size_t j = i + 1; j < words.size(); j++) {
                std::string word1 = words[i];
                std::string word2 = words[j];
                
                // Ensure consistent ordering
                if (word1 > word2) {
                    std::swap(word1, word2);
                }
                
                // Update connection count and strength
                if (connections[word1].find(word2) == connections[word1].end()) {
                    connections[word1][word2] = {1, 1.0f};
                } else {
                    connections[word1][word2].first++;
                    connections[word1][word2].second += 0.1f; // Increase strength
                }
                
                std::cout << "  📎 " << word1 << " ↔ " << word2 << " (count: " << connections[word1][word2].first << ")" << std::endl;
            }
        }
        
        // Add new concepts
        for (const auto& word : words) {
            if (word.length() > 3) {
                concepts.insert(word);
            }
        }
        
        std::cout << "  ✅ Found " << words.size() << " words, created " << (words.size() * (words.size() - 1) / 2) << " connections" << std::endl;
    }
    
    void saveBinaryData() {
        std::cout << "💾 Saving binary data..." << std::endl;
        
        // Save connections to binary file
        std::ofstream bin_file(binary_file, std::ios::binary);
        if (bin_file.is_open()) {
            // Write metadata
            BinaryMetadata metadata;
            metadata.total_connections = getTotalConnections();
            metadata.questions_asked = questions_asked.size();
            metadata.answers_received = answers_received.size();
            metadata.concepts_explored = concepts.size();
            metadata.curiosity_driver = curiosity_driver;
            metadata.stability_driver = stability_driver;
            metadata.reinforcement_driver = reinforcement_driver;
            metadata.timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            bin_file.write(reinterpret_cast<const char*>(&metadata), sizeof(BinaryMetadata));
            
            // Write connections
            for (const auto& word1 : connections) {
                for (const auto& word2 : word1.second) {
                    BinaryWordConnection conn(word1.first, word2.first, 
                                            word2.second.first, word2.second.second,
                                            "curiosity", ""); // Empty context for now
                    
                    bin_file.write(reinterpret_cast<const char*>(&conn), sizeof(BinaryWordConnection));
                }
            }
            
            bin_file.close();
        }
        
        // Save contexts separately (for space efficiency)
        std::ofstream ctx_file(context_file, std::ios::binary);
        if (ctx_file.is_open()) {
            for (size_t i = 0; i < questions_asked.size() && i < answers_received.size(); i++) {
                // Write question length and question
                uint32_t q_len = questions_asked[i].length();
                ctx_file.write(reinterpret_cast<const char*>(&q_len), sizeof(uint32_t));
                ctx_file.write(questions_asked[i].c_str(), q_len);
                
                // Write answer length and answer
                uint32_t a_len = answers_received[i].length();
                ctx_file.write(reinterpret_cast<const char*>(&a_len), sizeof(uint32_t));
                ctx_file.write(answers_received[i].c_str(), a_len);
            }
            ctx_file.close();
        }
        
        std::cout << "  ✅ Binary data saved to " << binary_file << " and " << context_file << std::endl;
    }
    
    void loadBinaryData() {
        std::cout << "📂 Loading binary data..." << std::endl;
        
        std::ifstream bin_file(binary_file, std::ios::binary);
        if (bin_file.is_open()) {
            // Read metadata
            BinaryMetadata metadata;
            bin_file.read(reinterpret_cast<char*>(&metadata), sizeof(BinaryMetadata));
            
            std::cout << "  📊 Loaded: " << metadata.total_connections << " connections, "
                     << metadata.questions_asked << " questions, "
                     << metadata.concepts_explored << " concepts" << std::endl;
            
            // Read connections
            for (uint32_t i = 0; i < metadata.total_connections; i++) {
                BinaryWordConnection conn;
                bin_file.read(reinterpret_cast<char*>(&conn), sizeof(BinaryWordConnection));
                
                connections[conn.word1][conn.word2] = {conn.count, conn.strength};
            }
            
            bin_file.close();
        } else {
            std::cout << "  🆕 No existing binary data found, starting fresh" << std::endl;
        }
    }
    
    void analyzeStorage() {
        // Calculate file sizes
        std::ifstream bin_file(binary_file, std::ios::binary | std::ios::ate);
        std::ifstream ctx_file(context_file, std::ios::binary | std::ios::ate);
        
        size_t bin_size = 0;
        size_t ctx_size = 0;
        
        if (bin_file.is_open()) {
            bin_size = bin_file.tellg();
            bin_file.close();
        }
        
        if (ctx_file.is_open()) {
            ctx_size = ctx_file.tellg();
            ctx_file.close();
        }
        
        std::cout << "  📁 Binary connections file: " << (bin_size / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  📁 Context file: " << (ctx_size / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  📁 Total storage: " << ((bin_size + ctx_size) / 1024.0 / 1024.0) << " MB" << std::endl;
        
        // Calculate efficiency
        int total_connections = getTotalConnections();
        if (total_connections > 0) {
            double bytes_per_connection = (double)(bin_size + ctx_size) / total_connections;
            std::cout << "  ⚡ Storage efficiency: " << bytes_per_connection << " bytes per connection" << std::endl;
        }
    }
    
    void showTopConnections() {
        std::vector<std::pair<std::string, uint32_t>> connection_counts;
        
        for (const auto& word1 : connections) {
            for (const auto& word2 : word1.second) {
                std::string connection = word1.first + " ↔ " + word2.first;
                connection_counts.push_back({connection, word2.second.first});
            }
        }
        
        // Sort by count
        std::sort(connection_counts.begin(), connection_counts.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Show top 10
        for (int i = 0; i < std::min(10, (int)connection_counts.size()); i++) {
            std::cout << "  " << connection_counts[i].first 
                     << " (count: " << connection_counts[i].second << ")" << std::endl;
        }
    }
    
    // Ollama integration (same as before)
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
    
    // Word processing (same as before)
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
            "can", "this", "that", "these", "those", "a", "an", "as", "if", "when"
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
};

int main() {
    std::cout << "🧠 MELVIN BINARY STORAGE SYSTEM" << std::endl;
    std::cout << "===============================" << std::endl;
    std::cout << "💾 Efficient binary storage instead of JSON!" << std::endl;
    
    MelvinBinaryStorage melvin;
    melvin.runBinaryLearning(5); // 5 cycles for demo
    
    std::cout << "\n✅ Binary learning completed!" << std::endl;
    std::cout << "💾 All data stored in efficient binary format!" << std::endl;
    
    return 0;
}
