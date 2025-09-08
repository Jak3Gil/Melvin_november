#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <curl/curl.h>
#include "logging.h"
#include "websearch_fallback.cpp"

// Forward declaration
std::string perform_web_search(const std::string& query);

class MelvinUnifiedBrain {
private:
    std::string storage_path;
    bool initialized;
    
public:
    MelvinUnifiedBrain(const std::string& path = "melvin_binary_memory") 
        : storage_path(path), initialized(false) {
        melvin::Logger::log_info("Initializing Melvin Unified Brain with storage path: %s", path.c_str());
        initialized = true;
    }
    
    ~MelvinUnifiedBrain() {
        if (initialized) {
            melvin::Logger::log_info("Shutting down Melvin Unified Brain");
        }
    }
    
    std::string process_input(const std::string& user_input) {
        melvin::Logger::log_info("Processing input: %s", user_input.c_str());
        
        // Simple processing for demo
        if (user_input.find("search") != std::string::npos || user_input.find("find") != std::string::npos) {
            melvin::Logger::log_info("Triggering web search for query: %s", user_input.c_str());
            std::string search_result = perform_web_search(user_input);
            if (!search_result.empty()) {
                melvin::Logger::log_info("Web search successful, result length: %zu", search_result.length());
                return "Based on research: " + search_result;
            } else {
                melvin::Logger::log_warn("Web search returned empty result");
                return "I couldn't find information about that topic. Please try rephrasing your question.";
            }
        }
        
        return "I understand you said: \"" + user_input + "\". How can I help you further?";
    }
    
    void show_diagnostics() {
        melvin::Logger::log_info("Running diagnostic mode");
        
        std::cout << "🧠 MELVIN UNIFIED BRAIN DIAGNOSTICS" << std::endl;
        std::cout << "===================================" << std::endl;
        
        // Check libcurl version
        curl_version_info_data* curl_info = curl_version_info(CURLVERSION_NOW);
        if (curl_info) {
            std::cout << "📡 libcurl version: " << curl_info->version << std::endl;
            std::cout << "🔧 SSL support: " << (curl_info->features & CURL_VERSION_SSL ? "Yes" : "No") << std::endl;
        }
        
        // Check environment variables
        const char* bing_key = std::getenv("BING_API_KEY");
        std::cout << "🔑 BING_API_KEY: " << (bing_key ? "Set" : "Not set") << std::endl;
        
        if (!bing_key) {
            std::cout << "⚠️  Warning: BING_API_KEY not set, will use DuckDuckGo fallback" << std::endl;
        }
        
        // Test web search
        std::cout << "🔍 Testing web search..." << std::endl;
        std::string test_result = perform_web_search("test");
        std::cout << "✅ Web search test: " << (test_result.empty() ? "Failed" : "Success") << std::endl;
        
        std::cout << "===================================" << std::endl;
        std::cout << "Diagnostic complete. Check melvin_debug.log for detailed logs." << std::endl;
    }
    
    void run_interactive_session() {
        melvin::Logger::log_info("Starting interactive session");
        
        std::cout << "🧠 MELVIN UNIFIED BRAIN" << std::endl;
        std::cout << "=======================" << std::endl;
        std::cout << "Welcome! I'm Melvin, your unified brain AI companion." << std::endl;
        std::cout << "Type 'quit' to exit, 'diag' for diagnostics." << std::endl;
        std::cout << "=======================" << std::endl;
        
        std::string user_input;
        
        while (true) {
            std::cout << "\nYou: ";
            std::getline(std::cin, user_input);
            
            if (user_input.empty()) {
                continue;
            }
            
            std::string lower_input = user_input;
            std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
            
            if (lower_input == "quit" || lower_input == "exit") {
                melvin::Logger::log_info("User requested exit");
                std::cout << "\nMelvin: Thank you for the conversation! Goodbye! 🧠✨" << std::endl;
                break;
            } else if (lower_input == "diag") {
                show_diagnostics();
                continue;
            }
            
            // Process input through unified brain system
            std::cout << "\nMelvin: ";
            try {
                std::string response = process_input(user_input);
                std::cout << response << std::endl;
            } catch (const std::exception& e) {
                melvin::Logger::log_error("Error processing input: %s", e.what());
                std::cout << "I encountered an error processing your input. Please try again." << std::endl;
            }
        }
    }
};

int main(int argc, char* argv[]) {
    // Initialize logging first
    melvin::Logger::init_logging("melvin_debug.log", true);
    melvin::Logger::log_info("Melvin Unified Brain starting up");
    
    // Check for diagnostic mode
    bool diagnostic_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--diag") {
            diagnostic_mode = true;
            break;
        }
    }
    
    try {
        // Initialize libcurl globally
        curl_global_init(CURL_GLOBAL_DEFAULT);
        melvin::Logger::log_info("libcurl initialized");
        
        // Create Melvin brain instance
        MelvinUnifiedBrain melvin;
        
        if (diagnostic_mode) {
            melvin.show_diagnostics();
        } else {
            melvin.run_interactive_session();
        }
        
        // Cleanup libcurl
        curl_global_cleanup();
        melvin::Logger::log_info("libcurl cleanup completed");
        
    } catch (const std::exception& e) {
        melvin::Logger::log_error("Fatal error: %s", e.what());
        std::cerr << "\n❌ Fatal Error: " << e.what() << std::endl;
        std::cerr << "Please check melvin_debug.log for more details." << std::endl;
        curl_global_cleanup();
        return 1;
    } catch (...) {
        melvin::Logger::log_error("Unknown fatal error occurred");
        std::cerr << "\n❌ Unknown Fatal Error occurred" << std::endl;
        std::cerr << "Please check melvin_debug.log for more details." << std::endl;
        curl_global_cleanup();
        return 1;
    }
    
    melvin::Logger::log_info("Melvin Unified Brain shutdown complete");
    return 0;
}