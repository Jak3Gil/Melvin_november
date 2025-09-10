#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <chrono>
#include <random>
#include <cmath>

// 🧠 MELVIN SIMPLE WORKING DEMONSTRATION
// ======================================
// A simplified but functional version of Melvin's core reasoning system

class SimpleMelvin {
private:
    std::map<std::string, std::vector<std::string>> memory;
    std::vector<std::string> recent_inputs;
    int processing_count = 0;
    
public:
    SimpleMelvin() {
        std::cout << "🧠 Melvin Simple Brain initialized\n";
        std::cout << "===================================\n\n";
    }
    
    // Core reasoning function
    std::string process_input(const std::string& input) {
        processing_count++;
        recent_inputs.push_back(input);
        
        // Keep only last 10 inputs
        if (recent_inputs.size() > 10) {
            recent_inputs.erase(recent_inputs.begin());
        }
        
        std::cout << "[Recall Track]\n";
        std::string recall_result = attempt_recall(input);
        
        std::cout << "[Exploration Track]\n";
        std::string exploration_result = generate_exploration(input);
        
        std::cout << "[Integration Phase]\n";
        std::string integration = integrate_reasoning(recall_result, exploration_result, input);
        
        return integration;
    }
    
private:
    std::string attempt_recall(const std::string& input) {
        std::cout << "  🔍 Searching memory for relevant information...\n";
        
        // Simple keyword matching for recall
        std::vector<std::string> keywords = extract_keywords(input);
        std::vector<std::string> relevant_memories;
        
        for (const auto& keyword : keywords) {
            if (memory.find(keyword) != memory.end()) {
                for (const auto& memory_item : memory[keyword]) {
                    relevant_memories.push_back(memory_item);
                }
            }
        }
        
        if (!relevant_memories.empty()) {
            std::string result = "Found " + std::to_string(relevant_memories.size()) + " relevant memories:\n";
            for (size_t i = 0; i < std::min(relevant_memories.size(), size_t(3)); i++) {
                result += "  - " + relevant_memories[i] + "\n";
            }
            std::cout << "  ✅ " << result;
            return result;
        } else {
            std::cout << "  ❌ No relevant memories found\n";
            return "No relevant memories found";
        }
    }
    
    std::string generate_exploration(const std::string& input) {
        std::cout << "  🚀 Generating new insights and connections...\n";
        
        // Simple exploration based on input analysis
        std::vector<std::string> keywords = extract_keywords(input);
        std::string exploration = "Exploration insights:\n";
        
        for (const auto& keyword : keywords) {
            exploration += "  - Analyzing concept: " + keyword + "\n";
            exploration += "  - Potential connections to explore\n";
        }
        
        exploration += "  - Generating curiosity questions\n";
        exploration += "  - Building new knowledge connections\n";
        
        std::cout << "  ✅ " << exploration;
        return exploration;
    }
    
    std::string integrate_reasoning(const std::string& recall, const std::string& exploration, const std::string& input) {
        std::cout << "  🔗 Integrating recall and exploration tracks...\n";
        
        // Calculate confidence based on recall success
        float confidence = (recall.find("No relevant memories") != std::string::npos) ? 0.3f : 0.7f;
        
        std::string integration = "INTEGRATED RESPONSE:\n";
        integration += "===================\n";
        integration += "Input: " + input + "\n\n";
        
        if (confidence > 0.5f) {
            integration += "High confidence - emphasizing recall track:\n";
            integration += recall + "\n";
            integration += "Building on existing knowledge.\n";
        } else {
            integration += "Low confidence - emphasizing exploration track:\n";
            integration += exploration + "\n";
            integration += "Generating new understanding.\n";
        }
        
        integration += "\nConfidence Score: " + std::to_string(confidence) + "\n";
        integration += "Processing Count: " + std::to_string(processing_count) + "\n";
        
        // Store the processed information
        store_memory(input, integration);
        
        std::cout << "  ✅ Integration complete\n\n";
        return integration;
    }
    
    std::vector<std::string> extract_keywords(const std::string& input) {
        std::vector<std::string> keywords;
        std::string current_word;
        
        for (char c : input) {
            if (std::isalpha(c) || std::isdigit(c)) {
                current_word += std::tolower(c);
            } else if (!current_word.empty()) {
                if (current_word.length() > 2) { // Only words longer than 2 chars
                    keywords.push_back(current_word);
                }
                current_word.clear();
            }
        }
        
        if (!current_word.empty() && current_word.length() > 2) {
            keywords.push_back(current_word);
        }
        
        return keywords;
    }
    
    void store_memory(const std::string& input, const std::string& processed) {
        std::vector<std::string> keywords = extract_keywords(input);
        
        for (const auto& keyword : keywords) {
            memory[keyword].push_back(processed);
            
            // Keep only last 5 memories per keyword
            if (memory[keyword].size() > 5) {
                memory[keyword].erase(memory[keyword].begin());
            }
        }
    }
    
public:
    void show_memory_status() {
        std::cout << "🧠 MEMORY STATUS\n";
        std::cout << "================\n";
        std::cout << "Total keywords stored: " << memory.size() << "\n";
        std::cout << "Recent inputs: " << recent_inputs.size() << "\n";
        std::cout << "Processing count: " << processing_count << "\n\n";
    }
    
    void demonstrate_curiosity() {
        std::cout << "🤔 CURIOSITY DEMONSTRATION\n";
        std::cout << "==========================\n";
        
        std::vector<std::string> curiosity_questions = {
            "What is the relationship between memory and learning?",
            "How do neural networks process information?",
            "What makes something memorable?",
            "How can we improve reasoning capabilities?"
        };
        
        for (const auto& question : curiosity_questions) {
            std::cout << "\nQuestion: " << question << "\n";
            std::cout << "----------------------------------------\n";
            process_input(question);
        }
    }
};

int main() {
    std::cout << "🧠 MELVIN SIMPLE WORKING DEMONSTRATION\n";
    std::cout << "======================================\n";
    std::cout << "Testing core reasoning with clearer responses\n";
    std::cout << "No complex dependencies - Pure C++ implementation\n\n";
    
    SimpleMelvin melvin;
    
    // Test basic processing
    std::cout << "📝 TESTING BASIC PROCESSING\n";
    std::cout << "============================\n";
    
    std::vector<std::string> test_inputs = {
        "I want to learn about artificial intelligence",
        "How does memory work in the brain?",
        "What is machine learning?",
        "Can you explain neural networks?"
    };
    
    for (const auto& input : test_inputs) {
        std::cout << "\nProcessing: \"" << input << "\"\n";
        std::cout << "----------------------------------------\n";
        std::string result = melvin.process_input(input);
        std::cout << result << "\n";
    }
    
    // Show memory status
    melvin.show_memory_status();
    
    // Demonstrate curiosity
    melvin.demonstrate_curiosity();
    
    std::cout << "\n🎉 DEMONSTRATION COMPLETE!\n";
    std::cout << "==========================\n";
    std::cout << "Melvin successfully demonstrated:\n";
    std::cout << "- Recall Track processing\n";
    std::cout << "- Exploration Track generation\n";
    std::cout << "- Integration Phase reasoning\n";
    std::cout << "- Memory storage and retrieval\n";
    std::cout << "- Curiosity-driven learning\n\n";
    
    return 0;
}
