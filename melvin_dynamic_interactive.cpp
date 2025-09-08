#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <set>

// 🧠 MELVIN DYNAMIC INTERACTIVE SYSTEM
// ====================================
// Phase 9.1: Dynamic Response Generator with Context Weaving & Curiosity Injection

class DynamicMelvin {
private:
    std::map<std::string, std::vector<std::string>> knowledge_base;
    std::vector<std::pair<std::string, std::string>> conversation_pairs;
    std::vector<std::string> recent_responses;
    std::random_device rd;
    std::mt19937_64 gen;
    uint64_t conversation_turn;
    double session_start_time;
    
    // Dynamic personality system
    std::string current_personality;
    std::map<std::string, std::vector<std::string>> personality_banks;
    std::map<std::string, std::vector<std::string>> curiosity_questions;
    std::map<std::string, std::vector<std::string>> context_connectors;
    
    // Memory for context weaving
    std::vector<std::string> user_topics;
    std::vector<std::string> user_interests;
    std::map<std::string, int> topic_frequency;
    
    // Tool simulation
    std::map<std::string, bool> tool_availability;
    std::map<std::string, std::string> tool_responses;
    
public:
    DynamicMelvin() : gen(rd()), conversation_turn(0), 
                      session_start_time(static_cast<double>(std::time(nullptr))),
                      current_personality("curious") {
        initialize_knowledge_base();
        initialize_personality_banks();
        initialize_curiosity_system();
        initialize_context_system();
        initialize_tools();
    }
    
    void initialize_knowledge_base() {
        // Medical knowledge with context
        knowledge_base["cancer"] = {
            "Cancer immunotherapy is revolutionizing treatment approaches",
            "CAR-T cell therapy shows remarkable success in blood cancers",
            "Early detection through screening programs saves countless lives",
            "Precision medicine is tailoring treatments to individual patients",
            "Cancer research funding has accelerated breakthrough discoveries"
        };
        
        knowledge_base["health"] = {
            "Mental health awareness is transforming healthcare approaches",
            "Preventive medicine focuses on lifestyle and early intervention",
            "Telemedicine is expanding access to healthcare worldwide",
            "Personalized nutrition is optimizing individual health outcomes"
        };
        
        // Science with dynamic connections
        knowledge_base["space"] = {
            "James Webb Telescope is revealing the universe's earliest galaxies",
            "Mars exploration missions are preparing for human colonization",
            "Exoplanet discoveries are expanding our understanding of life",
            "Space debris management is becoming a critical challenge",
            "Private space companies are democratizing space access"
        };
        
        knowledge_base["ai"] = {
            "Large language models are transforming human-computer interaction",
            "AI ethics frameworks are shaping responsible development",
            "Machine learning is accelerating drug discovery processes",
            "Neural networks are mimicking biological brain structures",
            "AI safety research is ensuring beneficial outcomes"
        };
        
        knowledge_base["technology"] = {
            "Quantum computing promises to solve previously impossible problems",
            "5G networks are enabling real-time augmented reality experiences",
            "Blockchain technology is revolutionizing digital trust systems",
            "Edge computing is bringing intelligence closer to data sources"
        };
    }
    
    void initialize_personality_banks() {
        // Curious personality responses
        personality_banks["curious"] = {
            "That's fascinating! I'm connecting this to something I've been thinking about...",
            "Oh, this sparks so many questions in my mind!",
            "I find myself deeply curious about this topic!",
            "This reminds me of something intriguing I've learned...",
            "What an interesting perspective! It makes me wonder...",
            "I'm genuinely excited to explore this with you!",
            "This touches on something I find endlessly fascinating...",
            "I can't help but be curious about the deeper implications...",
            "This opens up such interesting possibilities!",
            "I'm drawn to explore this topic further..."
        };
        
        // Empathetic personality responses
        personality_banks["empathetic"] = {
            "I can sense this is important to you, and I want to help...",
            "I understand how this might feel significant...",
            "This sounds like it could be meaningful for you...",
            "I'm here to support you in exploring this...",
            "I can relate to the importance of this topic...",
            "This seems to touch on something deeply personal...",
            "I want to approach this with care and understanding...",
            "I'm listening and I want to help however I can...",
            "This feels like something that matters to you...",
            "I'm here to provide whatever support you need..."
        };
        
        // Technical personality responses
        personality_banks["technical"] = {
            "From a technical perspective, this involves several fascinating aspects...",
            "The underlying mechanisms here are quite sophisticated...",
            "This presents an interesting engineering challenge...",
            "The data suggests some compelling patterns...",
            "From a systems perspective, this is quite complex...",
            "The technical implementation here is noteworthy...",
            "This involves some intriguing algorithmic considerations...",
            "The technical specifications are quite impressive...",
            "From an analytical standpoint, this is fascinating...",
            "The technical architecture here is quite elegant..."
        };
        
        // Casual personality responses
        personality_banks["casual"] = {
            "Oh, that's cool! I was just thinking about something similar...",
            "Nice! This reminds me of something I heard recently...",
            "That's pretty interesting! I wonder if...",
            "Oh wow, that's actually really neat!",
            "That's awesome! I'm totally curious about...",
            "Sweet! This is right up my alley...",
            "That's really cool! I've been wondering about...",
            "Oh, that's neat! I was just reading about...",
            "That's pretty awesome! I'm thinking...",
            "Cool! This is something I find really interesting..."
        };
    }
    
    void initialize_curiosity_system() {
        // Curiosity questions for different contexts
        curiosity_questions["medical"] = {
            "Should I search for the latest research developments?",
            "Would you like me to find recent breakthrough studies?",
            "I'm curious about the newest treatment approaches - should I look them up?",
            "Would you be interested in the latest clinical trial results?",
            "Should I pull up the most recent medical guidelines?",
            "I'm wondering about the newest diagnostic techniques - want me to search?",
            "Would you like to see the latest patient outcome data?",
            "Should I find the most recent expert recommendations?"
        };
        
        curiosity_questions["science"] = {
            "Should I search for the newest discoveries in this field?",
            "Would you like me to find the latest research papers?",
            "I'm curious about recent breakthroughs - should I look them up?",
            "Would you be interested in the newest experimental results?",
            "Should I pull up the latest scientific consensus?",
            "I'm wondering about recent developments - want me to search?",
            "Would you like to see the newest findings?",
            "Should I find the most recent expert analysis?"
        };
        
        curiosity_questions["technology"] = {
            "Should I search for the latest tech developments?",
            "Would you like me to find recent innovation updates?",
            "I'm curious about the newest implementations - should I look them up?",
            "Would you be interested in the latest technical breakthroughs?",
            "Should I pull up the most recent industry trends?",
            "I'm wondering about new applications - want me to search?",
            "Would you like to see the newest technical solutions?",
            "Should I find the most recent expert insights?"
        };
        
        curiosity_questions["general"] = {
            "What aspect of this interests you most?",
            "Should I explore this topic further?",
            "What would you like to know more about?",
            "I'm curious - what sparked your interest in this?",
            "Would you like me to dive deeper into this?",
            "What angle would you like to explore?",
            "I'm wondering - what's your take on this?",
            "Should I search for more information about this?"
        };
    }
    
    void initialize_context_system() {
        // Context connectors for weaving previous conversations
        context_connectors["medical"] = {
            "Building on what we discussed about health earlier...",
            "This connects to the medical topics we've been exploring...",
            "Following up on our health conversation...",
            "This relates to the medical research we touched on...",
            "Expanding on our discussion about healthcare..."
        };
        
        context_connectors["science"] = {
            "This builds on the scientific concepts we've been discussing...",
            "Connecting to our exploration of science...",
            "This relates to the scientific topics we've covered...",
            "Following up on our science conversation...",
            "This expands on our scientific discussion..."
        };
        
        context_connectors["technology"] = {
            "This connects to the tech topics we've been exploring...",
            "Building on our technology discussion...",
            "This relates to the tech concepts we've covered...",
            "Following up on our technology conversation...",
            "This expands on our tech discussion..."
        };
    }
    
    void initialize_tools() {
        tool_availability["WebSearchTool"] = true;
        tool_availability["KnowledgeBase"] = true;
        tool_availability["CuriosityEngine"] = true;
        
        tool_responses["WebSearchTool"] = "I can search for the latest information on this topic.";
        tool_responses["KnowledgeBase"] = "I have relevant knowledge I can share about this.";
        tool_responses["CuriosityEngine"] = "I can generate some interesting questions about this.";
    }
    
    std::string process_input(const std::string& user_input) {
        conversation_turn++;
        
        // Analyze input and context
        std::string input_type = analyze_input_type(user_input);
        std::string intent = analyze_intent(user_input);
        std::string emotion = detect_emotion(user_input);
        
        // Update context memory
        update_context_memory(user_input, input_type);
        
        // Select dynamic personality
        select_dynamic_personality(input_type, emotion);
        
        // Generate dynamic response
        std::string response = generate_dynamic_response(user_input, input_type, intent, emotion);
        
        // Check for repetition and regenerate if needed
        if (is_repetitive(response)) {
            response = regenerate_with_different_style(user_input, input_type, intent, emotion);
        }
        
        // Store conversation
        conversation_pairs.push_back({user_input, response});
        recent_responses.push_back(response);
        
        // Keep only last 5 responses for repetition check
        if (recent_responses.size() > 5) {
            recent_responses.erase(recent_responses.begin());
        }
        
        return response;
    }
    
    std::string analyze_input_type(const std::string& input) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        if (lower_input.find("cancer") != std::string::npos || 
            lower_input.find("health") != std::string::npos ||
            lower_input.find("medical") != std::string::npos ||
            lower_input.find("doctor") != std::string::npos ||
            lower_input.find("treatment") != std::string::npos) {
            return "medical";
        }
        
        if (lower_input.find("space") != std::string::npos ||
            lower_input.find("science") != std::string::npos ||
            lower_input.find("physics") != std::string::npos ||
            lower_input.find("research") != std::string::npos) {
            return "science";
        }
        
        if (lower_input.find("ai") != std::string::npos ||
            lower_input.find("artificial") != std::string::npos ||
            lower_input.find("technology") != std::string::npos ||
            lower_input.find("computer") != std::string::npos ||
            lower_input.find("tech") != std::string::npos) {
            return "technology";
        }
        
        return "general";
    }
    
    std::string analyze_intent(const std::string& input) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        for (const auto& topic : knowledge_base) {
            if (lower_input.find(topic.first) != std::string::npos) {
                return topic.first;
            }
        }
        
        return "general_inquiry";
    }
    
    std::string detect_emotion(const std::string& input) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        if (lower_input.find("!") != std::string::npos) {
            return "excited";
        }
        
        if (lower_input.find("?") != std::string::npos) {
            return "curious";
        }
        
        if (lower_input.find("cancer") != std::string::npos ||
            lower_input.find("worried") != std::string::npos ||
            lower_input.find("concerned") != std::string::npos) {
            return "concerned";
        }
        
        return "neutral";
    }
    
    void update_context_memory(const std::string& input, const std::string& input_type) {
        // Track user topics
        if (std::find(user_topics.begin(), user_topics.end(), input_type) == user_topics.end()) {
            user_topics.push_back(input_type);
        }
        
        // Track topic frequency
        topic_frequency[input_type]++;
        
        // Extract interests from input
        std::vector<std::string> words = tokenize(input);
        for (const auto& word : words) {
            if (word.length() > 3 && 
                std::find(user_interests.begin(), user_interests.end(), word) == user_interests.end()) {
                user_interests.push_back(word);
            }
        }
    }
    
    void select_dynamic_personality(const std::string& input_type, const std::string& emotion) {
        // Dynamic personality selection based on context
        if (emotion == "concerned" || input_type == "medical") {
            current_personality = "empathetic";
        } else if (input_type == "technology" || input_type == "science") {
            current_personality = "technical";
        } else if (emotion == "excited") {
            current_personality = "casual";
        } else {
            // Random selection for variety
            std::vector<std::string> personalities = {"curious", "empathetic", "technical", "casual"};
            std::uniform_int_distribution<> dis(0, personalities.size() - 1);
            current_personality = personalities[dis(gen)];
        }
    }
    
    std::string generate_dynamic_response(const std::string& input, const std::string& input_type, 
                                        const std::string& intent, const std::string& emotion) {
        
        std::ostringstream response;
        
        // Context weaving - connect to previous conversations
        if (conversation_turn > 1 && !user_topics.empty()) {
            std::string context_connector = get_context_connector(input_type);
            if (!context_connector.empty()) {
                response << context_connector << " ";
            }
        }
        
        // Dynamic personality response
        std::string personality_response = get_personality_response(current_personality);
        response << personality_response << " ";
        
        // Knowledge injection with context
        std::string knowledge = get_contextual_knowledge(intent, input_type);
        if (!knowledge.empty()) {
            response << knowledge << " ";
        }
        
        // Tool-backed response if knowledge gap detected
        if (knowledge.empty() && tool_availability["WebSearchTool"]) {
            response << tool_responses["WebSearchTool"] << " ";
        }
        
        // Curiosity injection
        std::string curiosity_question = get_curiosity_question(input_type);
        if (!curiosity_question.empty()) {
            response << curiosity_question;
        }
        
        return response.str();
    }
    
    std::string get_context_connector(const std::string& input_type) {
        if (context_connectors.find(input_type) != context_connectors.end()) {
            const auto& connectors = context_connectors[input_type];
            std::uniform_int_distribution<> dis(0, connectors.size() - 1);
            return connectors[dis(gen)];
        }
        return "";
    }
    
    std::string get_personality_response(const std::string& personality) {
        if (personality_banks.find(personality) != personality_banks.end()) {
            const auto& responses = personality_banks[personality];
            std::uniform_int_distribution<> dis(0, responses.size() - 1);
            return responses[dis(gen)];
        }
        return "That's interesting! ";
    }
    
    std::string get_contextual_knowledge(const std::string& intent, const std::string& input_type) {
        if (knowledge_base.find(intent) != knowledge_base.end()) {
            const auto& knowledge = knowledge_base[intent];
            std::uniform_int_distribution<> dis(0, knowledge.size() - 1);
            return knowledge[dis(gen)];
        }
        return "";
    }
    
    std::string get_curiosity_question(const std::string& input_type) {
        if (curiosity_questions.find(input_type) != curiosity_questions.end()) {
            const auto& questions = curiosity_questions[input_type];
            std::uniform_int_distribution<> dis(0, questions.size() - 1);
            return questions[dis(gen)];
        }
        return curiosity_questions["general"][0];
    }
    
    bool is_repetitive(const std::string& response) {
        if (recent_responses.size() < 2) return false;
        
        // Check similarity with recent responses
        for (const auto& recent : recent_responses) {
            if (calculate_similarity(response, recent) > 0.7) {
                return true;
            }
        }
        return false;
    }
    
    double calculate_similarity(const std::string& str1, const std::string& str2) {
        // Simple similarity calculation based on common words
        std::vector<std::string> words1 = tokenize(str1);
        std::vector<std::string> words2 = tokenize(str2);
        
        std::set<std::string> set1(words1.begin(), words1.end());
        std::set<std::string> set2(words2.begin(), words2.end());
        
        std::set<std::string> intersection;
        std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                            std::inserter(intersection, intersection.begin()));
        
        return static_cast<double>(intersection.size()) / 
               static_cast<double>(std::max(set1.size(), set2.size()));
    }
    
    std::string regenerate_with_different_style(const std::string& input, const std::string& input_type, 
                                             const std::string& intent, const std::string& emotion) {
        // Force different personality
        std::vector<std::string> personalities = {"curious", "empathetic", "technical", "casual"};
        personalities.erase(std::remove(personalities.begin(), personalities.end(), current_personality), personalities.end());
        
        if (!personalities.empty()) {
            std::uniform_int_distribution<> dis(0, personalities.size() - 1);
            current_personality = personalities[dis(gen)];
        }
        
        return generate_dynamic_response(input, input_type, intent, emotion);
    }
    
    std::vector<std::string> tokenize(const std::string& input) {
        std::vector<std::string> tokens;
        std::string current_token;
        
        for (char c : input) {
            if (std::isalpha(c) || std::isdigit(c)) {
                current_token += std::tolower(c);
            } else if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        }
        
        if (!current_token.empty()) {
            tokens.push_back(current_token);
        }
        
        return tokens;
    }
    
    void show_system_status() {
        std::cout << "\n📊 MELVIN DYNAMIC SYSTEM STATUS" << std::endl;
        std::cout << "===============================" << std::endl;
        std::cout << "Conversation turns: " << conversation_turn << std::endl;
        std::cout << "Current personality: " << current_personality << std::endl;
        std::cout << "User topics: ";
        for (const auto& topic : user_topics) {
            std::cout << topic << " ";
        }
        std::cout << std::endl;
        std::cout << "Topic frequency: ";
        for (const auto& pair : topic_frequency) {
            std::cout << pair.first << "(" << pair.second << ") ";
        }
        std::cout << std::endl;
    }
    
    void run_interactive_session() {
        std::cout << "🧠 MELVIN DYNAMIC INTERACTIVE SYSTEM" << std::endl;
        std::cout << "====================================" << std::endl;
        std::cout << "Hello! I'm Melvin with dynamic personality and context awareness!" << std::endl;
        std::cout << "I adapt my responses based on:" << std::endl;
        std::cout << "- Your interests and conversation history" << std::endl;
        std::cout << "- Emotional context and topic type" << std::endl;
        std::cout << "- Curiosity-driven questions" << std::endl;
        std::cout << "- Anti-repetition algorithms" << std::endl;
        std::cout << "\nType 'quit' to exit, 'status' for system info." << std::endl;
        std::cout << "====================================" << std::endl;
        
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
                std::cout << "\nMelvin: Thank you for this dynamic conversation! ";
                std::cout << "I've learned so much from our " << conversation_turn << " turns together. ";
                std::cout << "My personality adapted " << topic_frequency.size() << " times based on our topics. ";
                std::cout << "Until next time! 🧠✨" << std::endl;
                break;
            } else if (lower_input == "status") {
                show_system_status();
                continue;
            }
            
            // Process input through dynamic system
            std::cout << "\nMelvin: ";
            std::string response = process_input(user_input);
            std::cout << response << std::endl;
            
            // Add thinking delay
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
        }
    }
};

int main() {
    try {
        DynamicMelvin melvin;
        melvin.run_interactive_session();
    } catch (const std::exception& e) {
        std::cerr << "❌ Error in dynamic interactive session: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
