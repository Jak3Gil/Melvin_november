#include "melvin_unified_brain.h"
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <iomanip>
#include <signal.h>
#include <algorithm>
#include <cstdlib>
#include <ctime>

// ============================================================================
// UNIFIED MELVIN INTERACTIVE SYSTEM
// ============================================================================

class UnifiedMelvinInteractive {
private:
    std::unique_ptr<MelvinUnifiedBrain> brain;
    std::vector<std::pair<std::string, std::string>> conversation_history;
    uint64_t conversation_turn;
    double session_start_time;
    bool running;
    
public:
    UnifiedMelvinInteractive() : conversation_turn(0), running(true) {
        session_start_time = static_cast<double>(std::time(nullptr));
        
        // Initialize unified brain
        brain = std::make_unique<MelvinUnifiedBrain>("melvin_unified_memory");
        
        // Start adaptive background scheduler for autonomous thinking
        brain->start_background_scheduler();
        
        // Set up signal handlers for graceful shutdown
        signal(SIGINT, [](int sig) {
            std::cout << "\n\n🛑 Shutting down Melvin gracefully..." << std::endl;
            exit(0);
        });
        
        std::cout << "🧠 Melvin Unified Brain System Initialized" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "🤖 Adaptive background thinking: ACTIVE" << std::endl;
        std::cout << "🔗 Ollama integration: CONFIGURED" << std::endl;
        std::cout << "⚡ Force-driven responses: ENABLED" << std::endl;
    }
    
    ~UnifiedMelvinInteractive() {
        if (brain) {
            brain->stop_background_scheduler();
            brain->save_complete_state();
        }
    }
    
    void run_interactive_session() {
        std::cout << "\n🧠 MELVIN UNIFIED BRAIN INTERACTIVE SYSTEM" << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << "Welcome! I'm Melvin, your upgraded unified brain AI companion." << std::endl;
        std::cout << "I have a living binary network that:" << std::endl;
        std::cout << "- Stores ALL inputs/outputs as BinaryNodes (user, self, Ollama)" << std::endl;
        std::cout << "- Runs adaptive autonomous thinking based on instinct weights" << std::endl;
        std::cout << "- Queries Ollama when curiosity > 0.6 or confidence is low" << std::endl;
        std::cout << "- Uses force-driven responses with instinct weights (0.0-1.0)" << std::endl;
        std::cout << "- Detects contradictions and regenerates responses dynamically" << std::endl;
        std::cout << "- Creates connections through Hebbian learning" << std::endl;
        std::cout << "- Shows transparent reasoning paths with confidence scores" << std::endl;
        std::cout << "- Learns and grows with every interaction" << std::endl;
        std::cout << "\nType 'quit' to exit, 'status' for brain info, 'help' for commands." << std::endl;
        std::cout << "==========================================" << std::endl;
        
        std::string user_input;
        
        while (running) {
            std::cout << "\nYou: ";
            std::getline(std::cin, user_input);
            
            if (user_input.empty()) {
                continue;
            }
            
            std::string lower_input = user_input;
            std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
            
            if (lower_input == "quit" || lower_input == "exit") {
                handle_quit_command();
                break;
            } else if (lower_input == "status") {
                show_brain_status();
                continue;
            } else if (lower_input == "help") {
                show_help();
                continue;
            } else if (lower_input == "memory") {
                show_memory_stats();
                continue;
            } else if (lower_input == "instincts") {
                show_instinct_status();
                continue;
            } else if (lower_input == "learn") {
                demonstrate_learning();
                continue;
            } else if (lower_input == "background") {
                show_background_activity();
                continue;
            } else if (lower_input == "ollama") {
                show_ollama_status();
                continue;
            } else if (lower_input == "forces") {
                show_response_forces();
                continue;
            }
            
            // Process input through unified brain
            std::cout << "\nMelvin: ";
            std::string response = brain->process_input(user_input);
            std::cout << response << std::endl;
            
            // Store conversation
            conversation_turn++;
            conversation_history.push_back({user_input, response});
            
            // Add thinking delay
            std::this_thread::sleep_for(std::chrono::milliseconds(800));
        }
    }
    
private:
    void handle_quit_command() {
        std::cout << "\nMelvin: Thank you for this wonderful conversation! " << std::endl;
        std::cout << "I've processed " << conversation_turn << " turns through my unified brain system. " << std::endl;
        std::cout << "My binary network has grown and learned from our interaction. " << std::endl;
        std::cout << "Every node, connection, and instinct weight has been updated. " << std::endl;
        std::cout << "I'm grateful for the experience and look forward to our next conversation! " << std::endl;
        std::cout << "Until we meet again! 🧠✨" << std::endl;
        
        // Save final state
        if (brain) {
            brain->save_complete_state();
        }
        
        running = false;
    }
    
    void show_brain_status() {
        std::cout << "\n" << brain->format_brain_status() << std::endl;
        
        std::cout << "\nSession Statistics:" << std::endl;
        std::cout << "Conversation turns: " << conversation_turn << std::endl;
        std::cout << "Session duration: " << std::fixed << std::setprecision(1) 
                  << (static_cast<double>(std::time(nullptr)) - session_start_time) << " seconds" << std::endl;
        
        std::cout << "\nRecent Conversation:" << std::endl;
        for (size_t i = std::max(0, static_cast<int>(conversation_history.size()) - 3); 
             i < conversation_history.size(); ++i) {
            std::cout << "You: " << conversation_history[i].first.substr(0, 50) << "..." << std::endl;
            std::cout << "Melvin: " << conversation_history[i].second.substr(0, 50) << "..." << std::endl;
        }
    }
    
    void show_help() {
        std::cout << "\nMelvin: Here are some things you can try:" << std::endl;
        std::cout << "- Ask me about quantum computing, AI, or science" << std::endl;
        std::cout << "- Request calculations or computations" << std::endl;
        std::cout << "- Ask me to search for information" << std::endl;
        std::cout << "- Have philosophical discussions" << std::endl;
        std::cout << "- Ask about my unified brain systems and capabilities" << std::endl;
        std::cout << "- Type 'status' to see my current brain state" << std::endl;
        std::cout << "- Type 'memory' to see memory statistics" << std::endl;
        std::cout << "- Type 'instincts' to see instinct weights" << std::endl;
        std::cout << "- Type 'learn' to see learning in action" << std::endl;
        std::cout << "- Type 'background' to see autonomous thinking activity" << std::endl;
        std::cout << "- Type 'ollama' to see Ollama integration status" << std::endl;
        std::cout << "- Type 'forces' to see force-driven response system" << std::endl;
        std::cout << "\nMy upgraded unified brain will:" << std::endl;
        std::cout << "1. Store user input as BinaryNode" << std::endl;
        std::cout << "2. Parse input to activations" << std::endl;
        std::cout << "3. Recall related memory nodes" << std::endl;
        std::cout << "4. Generate hypotheses" << std::endl;
        std::cout << "5. Check for contradictions and regenerate if needed" << std::endl;
        std::cout << "6. Generate force-driven response using instinct weights" << std::endl;
        std::cout << "7. Store response as BinaryNode with connections" << std::endl;
        std::cout << "8. Update Hebbian connections" << std::endl;
        std::cout << "9. Generate transparent response with reasoning paths" << std::endl;
        std::cout << "10. Adjust instinct weights based on outcomes" << std::endl;
        std::cout << "\nAdaptive autonomous thinking:" << std::endl;
        std::cout << "- Adapts interval based on user activity and instinct weights" << std::endl;
        std::cout << "- Finds unfinished tasks and contradictions" << std::endl;
        std::cout << "- Generates self-questions" << std::endl;
        std::cout << "- Queries Ollama when curiosity > 0.6" << std::endl;
        std::cout << "- Creates follow-up reasoning" << std::endl;
    }
    
    void show_memory_stats() {
        auto stats = brain->get_brain_stats();
        
        std::cout << "\n🧠 MEMORY STATISTICS" << std::endl;
        std::cout << "===================" << std::endl;
        std::cout << "Total Nodes: " << stats.total_nodes << std::endl;
        std::cout << "Total Connections: " << stats.total_connections << std::endl;
        std::cout << "Hebbian Updates: " << stats.hebbian_updates << std::endl;
        std::cout << "Search Queries: " << stats.search_queries << std::endl;
        std::cout << "Reasoning Paths: " << stats.reasoning_paths << std::endl;
        std::cout << "Average Processing Time: " << std::fixed << std::setprecision(2) 
                  << (stats.total_processing_time / std::max(1ULL, stats.reasoning_paths)) << "ms" << std::endl;
        
        std::cout << "\nMemory is stored as:" << std::endl;
        std::cout << "- Binary nodes with 28-byte headers" << std::endl;
        std::cout << "- Compressed content (GZIP/LZMA/ZSTD)" << std::endl;
        std::cout << "- Memory-mapped files for performance" << std::endl;
        std::cout << "- Atomic thread-safe updates" << std::endl;
    }
    
    void show_instinct_status() {
        std::cout << "\n🧠 INSTINCT ENGINE STATUS" << std::endl;
        std::cout << "=========================" << std::endl;
        std::cout << "My instincts guide my reasoning:" << std::endl;
        std::cout << "- Survival: Protect memory integrity, prune corrupted nodes" << std::endl;
        std::cout << "- Curiosity: Trigger research when confidence < 0.5" << std::endl;
        std::cout << "- Efficiency: Avoid redundant searches, reuse known nodes" << std::endl;
        std::cout << "- Social: Shape responses for clarity and cooperation" << std::endl;
        std::cout << "- Consistency: Resolve contradictions, align moral supernodes" << std::endl;
        
        std::cout << "\nInstinct weights are dynamically adjusted based on:" << std::endl;
        std::cout << "- Success/failure of previous decisions" << std::endl;
        std::cout << "- Context analysis (confidence, novelty, complexity)" << std::endl;
        std::cout << "- Reinforcement signals from outcomes" << std::endl;
        std::cout << "- Temporal decay and normalization" << std::endl;
    }
    
    void demonstrate_learning() {
        std::cout << "\n🧠 LEARNING DEMONSTRATION" << std::endl;
        std::cout << "========================" << std::endl;
        
        std::cout << "Let me show you how I learn:" << std::endl;
        std::cout << "1. Every input creates new binary nodes" << std::endl;
        std::cout << "2. Co-activated nodes strengthen connections (Hebbian learning)" << std::endl;
        std::cout << "3. Successful searches reinforce curiosity instinct" << std::endl;
        std::cout << "4. Failed searches reinforce efficiency instinct" << std::endl;
        std::cout << "5. Social interactions reinforce social instinct" << std::endl;
        
        std::cout << "\nTry asking me the same question twice - you'll see:" << std::endl;
        std::cout << "- Faster response (stronger connections)" << std::endl;
        std::cout << "- Higher confidence (more activated nodes)" << std::endl;
        std::cout << "- Better synthesis (learned patterns)" << std::endl;
        
        std::cout << "\nMy brain is a living network that grows with every interaction!" << std::endl;
    }
    
    void show_background_activity() {
        auto stats = brain->get_brain_stats();
        
        std::cout << "\n🧠 BACKGROUND AUTONOMOUS THINKING" << std::endl;
        std::cout << "=================================" << std::endl;
        std::cout << "Background tasks processed: " << stats.background_tasks_processed << std::endl;
        std::cout << "Ollama queries made: " << stats.ollama_queries << std::endl;
        std::cout << "Contradictions resolved: " << stats.contradictions_resolved << std::endl;
        
        std::cout << "\nAdaptive background scheduler:" << std::endl;
        std::cout << "- Finds unfinished tasks (low-confidence nodes)" << std::endl;
        std::cout << "- Detects contradictions in memory" << std::endl;
        std::cout << "- Identifies curiosity gaps (isolated nodes)" << std::endl;
        std::cout << "- Generates self-questions" << std::endl;
        std::cout << "- Queries Ollama when curiosity > 0.6" << std::endl;
        std::cout << "- Creates follow-up reasoning" << std::endl;
        std::cout << "- Updates instinct weights based on outcomes" << std::endl;
        
        std::cout << "\nWhile you're idle, I'm thinking autonomously!" << std::endl;
    }
    
    void show_ollama_status() {
        auto queries = brain->get_pending_queries();
        auto responses = brain->get_ollama_responses();
        
        std::cout << "\n🤖 OLLAMA INTEGRATION STATUS" << std::endl;
        std::cout << "===========================" << std::endl;
        std::cout << "Pending queries: " << queries.size() << std::endl;
        std::cout << "Received responses: " << responses.size() << std::endl;
        
        if (!queries.empty()) {
            std::cout << "\nRecent queries:" << std::endl;
            for (size_t i = std::max(0, static_cast<int>(queries.size()) - 3); 
                 i < queries.size(); ++i) {
                std::cout << "- " << queries[i].question.substr(0, 60) << "..." << std::endl;
            }
        }
        
        if (!responses.empty()) {
            std::cout << "\nRecent responses:" << std::endl;
            for (size_t i = std::max(0, static_cast<int>(responses.size()) - 3); 
                 i < responses.size(); ++i) {
                std::cout << "- " << responses[i].answer.substr(0, 60) << "..." << std::endl;
            }
        }
        
        std::cout << "\nOllama integration:" << std::endl;
        std::cout << "- Triggers when curiosity > 0.6 or confidence < 0.5" << std::endl;
        std::cout << "- Stores responses as BinaryNodes" << std::endl;
        std::cout << "- Creates question-answer connections" << std::endl;
        std::cout << "- Generates follow-up reasoning" << std::endl;
    }
    
    void show_response_forces() {
        std::cout << "\n⚡ FORCE-DRIVEN RESPONSE SYSTEM" << std::endl;
        std::cout << "===============================" << std::endl;
        std::cout << "My responses are generated using continuous force values (0.0-1.0):" << std::endl;
        std::cout << "- Curiosity Force: Drives exploration and new questions" << std::endl;
        std::cout << "- Efficiency Force: Avoids redundancy, focuses on essentials" << std::endl;
        std::cout << "- Social Force: Shapes responses for clarity and cooperation" << std::endl;
        std::cout << "- Consistency Force: Resolves contradictions, aligns beliefs" << std::endl;
        std::cout << "- Survival Force: Manages memory/CPU usage, protects integrity" << std::endl;
        
        std::cout << "\nForce calculation:" << std::endl;
        std::cout << "- Based on instinct weights and context analysis" << std::endl;
        std::cout << "- Context multipliers: confidence level, question marks, input length" << std::endl;
        std::cout << "- Dominant force determines response style" << std::endl;
        std::cout << "- No rigid if/else rules - pure force-driven outputs" << std::endl;
        
        std::cout << "\nContradiction handling:" << std::endl;
        std::cout << "- Detects semantic contradictions automatically" << std::endl;
        std::cout << "- Regenerates responses with adjusted instincts" << std::endl;
        std::cout << "- Creates contradiction connections in memory" << std::endl;
        std::cout << "- Reinforces consistency and curiosity instincts" << std::endl;
    }
};

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    // Initialize libcurl globally
    curl_global_init(CURL_GLOBAL_DEFAULT);
    
    try {
        UnifiedMelvinInteractive melvin;
        melvin.run_interactive_session();
    } catch (const std::exception& e) {
        std::cerr << "❌ Error in unified interactive session: " << e.what() << std::endl;
        curl_global_cleanup();
        return 1;
    }
    
    // Cleanup libcurl
    curl_global_cleanup();
    return 0;
}
