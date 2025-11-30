/*
 * Melvin Word Connection System
 * 
 * Simple approach:
 * 1. Melvin asks Ollama questions
 * 2. Ollama gives answers
 * 3. Melvin extracts words from answers
 * 4. Melvin connects words that appear together
 * 5. Save everything
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

// Simple word connection storage
struct WordConnection {
    std::string word1;
    std::string word2;
    int count;
    std::string context; // The answer where they appeared together
    
    WordConnection() : count(0) {}
    
    WordConnection(const std::string& w1, const std::string& w2, const std::string& ctx) 
        : word1(w1), word2(w2), count(1), context(ctx) {}
};

// Melvin's Word Connection Brain
class MelvinWordBrain {
private:
    std::map<std::string, std::map<std::string, WordConnection>> connections;
    std::vector<std::string> questions;
    std::vector<std::string> answers;
    std::string ollama_url = "http://localhost:11434/api/generate";
    
public:
    MelvinWordBrain() {
        std::cout << "🧠 Melvin Word Connection Brain Initialized" << std::endl;
        std::cout << "🔗 Will connect words that appear together in Ollama answers" << std::endl;
        
        // Initialize some questions
        questions = {
            "What is a car?",
            "What is a bicycle?", 
            "What is a motorcycle?",
            "What is an engine?",
            "What are wheels?",
            "What is transportation?",
            "What is a vehicle?",
            "What is fuel?",
            "What is a driver?",
            "What is speed?"
        };
    }
    
    void runWordConnections(int num_questions = 10) {
        std::cout << "\n⏰ Starting word connection analysis..." << std::endl;
        std::cout << "🎯 Melvin will ask " << num_questions << " questions and connect words" << std::endl;
        
        for (int i = 0; i < num_questions && i < questions.size(); i++) {
            std::string question = questions[i];
            std::cout << "\n🤔 Melvin asks: " << question << std::endl;
            
            // Get answer from Ollama
            std::string answer = askOllama(question);
            std::cout << "💡 Ollama answers: " << answer << std::endl;
            
            // Store the answer
            answers.push_back(answer);
            
            // Extract and connect words
            connectWordsFromAnswer(answer);
            
            std::cout << "📊 Progress: " << (i+1) << "/" << num_questions << std::endl;
        }
        
        showResults();
        saveConnections();
    }
    
    void showResults() {
        std::cout << "\n🧠 MELVIN'S WORD CONNECTION RESULTS" << std::endl;
        std::cout << "===================================" << std::endl;
        
        std::cout << "\n📊 Statistics:" << std::endl;
        std::cout << "  Questions asked: " << questions.size() << std::endl;
        std::cout << "  Answers received: " << answers.size() << std::endl;
        
        // Count total connections
        int total_connections = 0;
        for (const auto& word1 : connections) {
            total_connections += word1.second.size();
        }
        std::cout << "  Total word connections: " << total_connections << std::endl;
        
        std::cout << "\n🔗 Top Word Connections:" << std::endl;
        showTopConnections();
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

private:
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
    
    void connectWordsFromAnswer(const std::string& answer) {
        // Extract words from answer
        std::vector<std::string> words = extractWords(answer);
        
        std::cout << "🔗 Melvin connects words from answer..." << std::endl;
        
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
                    connections[word1][word2] = WordConnection(word1, word2, answer);
                } else {
                    connections[word1][word2].count++;
                }
                
                std::cout << "  📎 " << word1 << " ↔ " << word2 << std::endl;
            }
        }
        
        std::cout << "  ✅ Found " << words.size() << " words, created " << (words.size() * (words.size() - 1) / 2) << " connections" << std::endl;
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
        std::ofstream file("melvin_word_connections.json");
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
                    file << "      \"context\": \"" << word2.second.context << "\"\n";
                    file << "    }";
                    first = false;
                }
            }
            
            file << "\n  ],\n";
            file << "  \"metadata\": {\n";
            file << "    \"total_connections\": " << getTotalConnections() << ",\n";
            file << "    \"questions_asked\": " << questions.size() << ",\n";
            file << "    \"answers_received\": " << answers.size() << "\n";
            file << "  }\n";
            file << "}\n";
            
            file.close();
            std::cout << "\n💾 Word connections saved to melvin_word_connections.json" << std::endl;
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
    std::cout << "🧠 MELVIN WORD CONNECTION BRAIN" << std::endl;
    std::cout << "===============================" << std::endl;
    std::cout << "🔗 Melvin connects words that appear together in Ollama answers!" << std::endl;
    
    MelvinWordBrain melvin;
    melvin.runWordConnections(10);
    
    std::cout << "\n✅ Word connection analysis completed!" << std::endl;
    std::cout << "🧠 Melvin has connected words based on Ollama answers!" << std::endl;
    
    return 0;
}
