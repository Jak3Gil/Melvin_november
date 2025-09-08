#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <curl/curl.h>
#include "logging.h"
#include "websearch_nodes.h"
#include "memory_nodes.h"

// Forward declarations
std::string perform_web_search(const std::string& query);

class MelvinUnifiedBrain {
private:
    melvin::GlobalMemoryManager* memory_manager;
    std::string storage_path;
    bool initialized;
    
public:
    MelvinUnifiedBrain(const std::string& path = "melvin_global_memory") 
        : storage_path(path), initialized(false) {
        melvin::Logger::log_info("Initializing Melvin Unified Brain with storage path: %s", path.c_str());
        
        // Initialize memory manager
        memory_manager = new melvin::GlobalMemoryManager(path);
        initialized = true;
        
        melvin::Logger::log_info("Memory manager initialized with %zu nodes and %zu connections", 
                                memory_manager->get_node_count(), memory_manager->get_connection_count());
    }
    
    ~MelvinUnifiedBrain() {
        if (initialized && memory_manager) {
            melvin::Logger::log_info("Shutting down Melvin Unified Brain");
            delete memory_manager;
        }
    }
    
    std::string process_input(const std::string& user_input) {
        melvin::Logger::log_info("Processing input: %s", user_input.c_str());
        
        // Convert input to nodes and connections
        std::vector<uint64_t> activated_nodes = memory_manager->process_input_to_nodes(user_input);
        melvin::Logger::log_info("Created %zu nodes from input", activated_nodes.size());
        
        // Activate nodes (strengthen their importance)
        for (uint64_t node_id : activated_nodes) {
            memory_manager->activate_node(node_id);
        }
        
        // Check if this is a search query
        bool is_search_query = (user_input.find("search") != std::string::npos || 
                               user_input.find("find") != std::string::npos ||
                               user_input.find("what is") != std::string::npos ||
                               user_input.find("who is") != std::string::npos ||
                               user_input.find("how") != std::string::npos);
        
        std::string response;
        
        if (is_search_query) {
            melvin::Logger::log_info("Detected search query, triggering web search");
            
            // Perform web search
            std::string search_result = perform_web_search(user_input);
            
            if (!search_result.empty()) {
                melvin::Logger::log_info("Web search successful, result length: %zu", search_result.length());
                
                // Create search result node
                uint64_t search_node_id = memory_manager->create_node(search_result, melvin::NodeType::SEARCH_RESULT, 200);
                
                // Connect search result to input concept
                if (!activated_nodes.empty()) {
                    uint64_t concept_id = activated_nodes.back(); // Last node is the concept
                    memory_manager->create_connection(concept_id, search_node_id, melvin::ConnectionType::SEMANTIC, 0.9f);
                }
                
                response = "Based on research: " + search_result;
            } else {
                melvin::Logger::log_warn("Web search returned empty result");
                response = memory_manager->generate_response_from_nodes(activated_nodes);
                response += " I couldn't find additional information about this topic.";
            }
        } else {
            // Generate response from existing memory
            response = memory_manager->generate_response_from_nodes(activated_nodes);
            
            // If response is too generic, try to enhance it
            if (response.find("I don't have enough information") != std::string::npos) {
                response = "I understand you said: \"" + user_input + "\". ";
                response += "I've stored this in my memory. Would you like me to search for more information about this topic?";
            }
        }
        
        // Save memory to disk
        memory_manager->save_to_disk();
        
        return response;
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
        
        // Show memory statistics
        std::cout << memory_manager->get_memory_stats() << std::endl;
        
        // Test web search
        std::cout << "🔍 Testing web search..." << std::endl;
        std::string test_result = perform_web_search("test");
        std::cout << "✅ Web search test: " << (test_result.empty() ? "Failed" : "Success") << std::endl;
        
        if (!test_result.empty()) {
            std::cout << "📄 Sample result: " << test_result.substr(0, 100) << "..." << std::endl;
        }
        
        std::cout << "===================================" << std::endl;
        std::cout << "Diagnostic complete. Check melvin_debug.log for detailed logs." << std::endl;
    }
    
    void show_memory_status() {
        std::cout << "\n📊 MEMORY STATUS" << std::endl;
        std::cout << "================" << std::endl;
        std::cout << memory_manager->get_memory_stats() << std::endl;
        
        // Show recent nodes
        auto recent_nodes = memory_manager->find_nodes_by_type(melvin::NodeType::CONCEPT);
        std::cout << "Recent concepts:" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), recent_nodes.size()); ++i) {
            std::cout << "  - " << recent_nodes[i]->content.substr(0, 50) << "..." << std::endl;
        }
    }
    
    void run_interactive_session() {
        melvin::Logger::log_info("Starting interactive session");
        
        std::cout << "🧠 MELVIN UNIFIED BRAIN" << std::endl;
        std::cout << "=======================" << std::endl;
        std::cout << "Welcome! I'm Melvin, your unified brain AI companion." << std::endl;
        std::cout << "I convert all inputs into nodes and connections, storing them in global memory." << std::endl;
        std::cout << "Type 'quit' to exit, 'diag' for diagnostics, 'memory' for memory status." << std::endl;
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
                std::cout << "\nMelvin: Thank you for the conversation! ";
                std::cout << "I've stored everything in my global memory. Goodbye! 🧠✨" << std::endl;
                break;
            } else if (lower_input == "diag") {
                show_diagnostics();
                continue;
            } else if (lower_input == "memory") {
                show_memory_status();
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
