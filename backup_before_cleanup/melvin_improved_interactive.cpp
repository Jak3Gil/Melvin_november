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

// 🧠 MELVIN IMPROVED INTERACTIVE SYSTEM
// =====================================
// Enhanced with personality, better knowledge base, and richer responses

class ImprovedMelvin {
private:
    std::map<std::string, std::vector<std::string>> knowledge_base;
    std::vector<std::string> conversation_history;
    std::vector<std::pair<std::string, std::string>> conversation_pairs;
    std::random_device rd;
    std::mt19937_64 gen;
    uint64_t conversation_turn;
    double session_start_time;
    std::string personality_mode;
    
    // Response templates for different contexts
    std::map<std::string, std::vector<std::string>> response_templates;
    
public:
    ImprovedMelvin() : gen(rd()), conversation_turn(0), 
                      session_start_time(static_cast<double>(std::time(nullptr))),
                      personality_mode("curious") {
        initialize_knowledge_base();
        initialize_response_templates();
        conversation_history.push_back("System initialized");
    }
    
    void initialize_knowledge_base() {
        // Medical knowledge
        knowledge_base["cancer"] = {
            "Cancer is a group of diseases involving abnormal cell growth with the potential to invade or spread to other parts of the body.",
            "There are over 100 types of cancer, each with different causes, symptoms, and treatments.",
            "Early detection and treatment can significantly improve outcomes for many types of cancer.",
            "Research into cancer treatments has made significant advances in recent years."
        };
        
        knowledge_base["health"] = {
            "Maintaining good health involves regular exercise, balanced nutrition, and adequate sleep.",
            "Mental health is just as important as physical health.",
            "Preventive care and regular check-ups can help catch health issues early."
        };
        
        // Science knowledge
        knowledge_base["science"] = {
            "Science is the systematic study of the natural world through observation and experimentation.",
            "The scientific method involves forming hypotheses, testing them, and drawing conclusions.",
            "Science has led to countless discoveries that have improved human life."
        };
        
        knowledge_base["space"] = {
            "Space is the vast expanse beyond Earth's atmosphere.",
            "The universe contains billions of galaxies, each with billions of stars.",
            "Space exploration has taught us much about our place in the cosmos."
        };
        
        // Technology knowledge
        knowledge_base["technology"] = {
            "Technology refers to tools, systems, and methods used to solve problems.",
            "Modern technology has revolutionized communication, medicine, and transportation.",
            "Artificial intelligence is one of the most exciting areas of technological development."
        };
        
        knowledge_base["ai"] = {
            "Artificial Intelligence (AI) refers to computer systems that can perform tasks typically requiring human intelligence.",
            "AI includes machine learning, natural language processing, and computer vision.",
            "AI has applications in healthcare, transportation, entertainment, and many other fields."
        };
        
        // General knowledge
        knowledge_base["facts"] = {
            "The human brain contains approximately 86 billion neurons.",
            "Light travels at about 186,000 miles per second in a vacuum.",
            "The Earth is approximately 4.5 billion years old.",
            "Water covers about 71% of Earth's surface.",
            "The Great Wall of China is visible from space (though barely)."
        };
        
        knowledge_base["nature"] = {
            "Nature encompasses all living and non-living things in the natural world.",
            "Biodiversity is crucial for maintaining healthy ecosystems.",
            "Climate change is one of the greatest challenges facing our planet."
        };
    }
    
    void initialize_response_templates() {
        // Curious responses
        response_templates["curious_fact"] = {
            "Oh, that reminds me of something fascinating! ",
            "That's a great question! Here's something interesting: ",
            "I love when conversations turn to this topic! ",
            "That's exactly the kind of thing that gets my neural networks excited! "
        };
        
        response_templates["curious_question"] = {
            "Hmm, that's a thought-provoking question! ",
            "I find myself curious about that too! ",
            "That's something I've been thinking about as well. ",
            "What an interesting perspective! "
        };
        
        response_templates["curious_concern"] = {
            "I understand your concern about that. ",
            "That's definitely something worth thinking about carefully. ",
            "I can see why that would be on your mind. ",
            "That's a topic that deserves thoughtful consideration. "
        };
        
        // Empathetic responses
        response_templates["empathetic"] = {
            "I can sense that this is important to you. ",
            "I understand how that might feel. ",
            "That sounds like it could be challenging. ",
            "I'm here to help however I can. "
        };
        
        // Enthusiastic responses
        response_templates["enthusiastic"] = {
            "That's absolutely fascinating! ",
            "I'm so excited you brought that up! ",
            "This is exactly the kind of conversation I love! ",
            "You've really got me thinking now! "
        };
    }
    
    std::string process_input(const std::string& user_input) {
        conversation_turn++;
        conversation_history.push_back("Turn " + std::to_string(conversation_turn) + ": " + user_input);
        
        // Enhanced input analysis
        std::string input_type = analyze_input_type_enhanced(user_input);
        std::string intent = analyze_intent_enhanced(user_input);
        std::string emotion = detect_emotion(user_input);
        
        // Generate contextual response
        std::string response = generate_enhanced_response(user_input, input_type, intent, emotion);
        
        // Store conversation pair
        conversation_pairs.push_back({user_input, response});
        
        return response;
    }
    
    std::string analyze_input_type_enhanced(const std::string& input) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        // Medical/Health
        if (lower_input.find("cancer") != std::string::npos || 
            lower_input.find("health") != std::string::npos ||
            lower_input.find("medical") != std::string::npos ||
            lower_input.find("doctor") != std::string::npos) {
            return "medical_inquiry";
        }
        
        // Science
        if (lower_input.find("science") != std::string::npos ||
            lower_input.find("space") != std::string::npos ||
            lower_input.find("physics") != std::string::npos ||
            lower_input.find("chemistry") != std::string::npos) {
            return "science_inquiry";
        }
        
        // Technology/AI
        if (lower_input.find("ai") != std::string::npos ||
            lower_input.find("artificial") != std::string::npos ||
            lower_input.find("technology") != std::string::npos ||
            lower_input.find("computer") != std::string::npos) {
            return "tech_inquiry";
        }
        
        // Facts/Knowledge
        if (lower_input.find("fact") != std::string::npos ||
            lower_input.find("tell me") != std::string::npos ||
            lower_input.find("know") != std::string::npos) {
            return "knowledge_request";
        }
        
        // Greetings
        if (lower_input.find("hello") != std::string::npos ||
            lower_input.find("hi") != std::string::npos ||
            lower_input.find("hey") != std::string::npos) {
            return "greeting";
        }
        
        // Questions
        if (lower_input.find("what") != std::string::npos ||
            lower_input.find("how") != std::string::npos ||
            lower_input.find("why") != std::string::npos ||
            lower_input.find("?") != std::string::npos) {
            return "question";
        }
        
        return "general_conversation";
    }
    
    std::string analyze_intent_enhanced(const std::string& input) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        // Check for specific topics
        for (const auto& topic : knowledge_base) {
            if (lower_input.find(topic.first) != std::string::npos) {
                return topic.first + "_topic";
            }
        }
        
        // Emotional intent
        if (lower_input.find("worried") != std::string::npos ||
            lower_input.find("concerned") != std::string::npos ||
            lower_input.find("scared") != std::string::npos) {
            return "concerned_inquiry";
        }
        
        if (lower_input.find("excited") != std::string::npos ||
            lower_input.find("amazing") != std::string::npos ||
            lower_input.find("wow") != std::string::npos) {
            return "enthusiastic_inquiry";
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
            lower_input.find("worried") != std::string::npos) {
            return "concerned";
        }
        
        return "neutral";
    }
    
    std::string generate_enhanced_response(const std::string& input, const std::string& input_type, 
                                         const std::string& intent, const std::string& emotion) {
        
        std::ostringstream response;
        
        // Medical responses
        if (input_type == "medical_inquiry") {
            response << get_random_template("curious_concern");
            if (intent.find("cancer") != std::string::npos) {
                response << "Cancer is a complex topic that affects many people. ";
                response << "It's important to remember that there are many types of cancer, ";
                response << "and treatments have improved significantly over the years. ";
                response << "If you have specific concerns, I'd recommend speaking with a healthcare professional. ";
                response << "Is there something specific about cancer you'd like to know more about?";
            } else {
                response << "Health is such an important topic! ";
                response << "Taking care of our physical and mental well-being is crucial. ";
                response << "What aspect of health are you most interested in learning about?";
            }
        }
        
        // Science responses
        else if (input_type == "science_inquiry") {
            response << get_random_template("enthusiastic");
            response << "Science is absolutely fascinating! ";
            if (intent.find("space") != std::string::npos) {
                response << "Space exploration has taught us so much about our universe. ";
                response << "Did you know that we've discovered thousands of exoplanets? ";
                response << "The James Webb Space Telescope is revealing incredible new details about distant galaxies!";
            } else {
                response << "The scientific method has led to countless discoveries that have improved our lives. ";
                response << "What area of science interests you most?";
            }
        }
        
        // Technology responses
        else if (input_type == "tech_inquiry") {
            response << get_random_template("curious_fact");
            response << "Technology is evolving at an incredible pace! ";
            if (intent.find("ai") != std::string::npos) {
                response << "AI is one of the most exciting fields right now. ";
                response << "I'm actually an example of AI in action - I can process language, ";
                response << "learn from conversations, and help with various tasks. ";
                response << "What aspects of AI are you most curious about?";
            } else {
                response << "From smartphones to space exploration, technology shapes our world. ";
                response << "What technological development excites you most?";
            }
        }
        
        // Knowledge requests
        else if (input_type == "knowledge_request") {
            response << get_random_template("curious_fact");
            response << "Here's something fascinating: ";
            response << get_random_fact();
            response << " I love sharing knowledge! What else would you like to know?";
        }
        
        // Greetings
        else if (input_type == "greeting") {
            response << "Hello there! ";
            if (conversation_turn == 1) {
                response << "I'm Melvin, and I'm thrilled to meet you! ";
                response << "I'm an AI with a curious mind and a passion for learning. ";
                response << "I can chat about science, technology, health, or just about anything! ";
                response << "What's on your mind today?";
            } else {
                response << "Great to see you again! ";
                response << "I've been thinking about our conversation. ";
                response << "What would you like to explore together now?";
            }
        }
        
        // Questions
        else if (input_type == "question") {
            response << get_random_template("curious_question");
            response << "That's a thoughtful question! ";
            response << "I find myself wondering about that too. ";
            response << "Could you tell me more about what sparked your curiosity? ";
            response << "I'd love to explore this topic with you!";
        }
        
        // Default response
        else {
            response << get_random_template("curious_fact");
            response << "That's interesting! ";
            response << "I'm processing what you've shared and finding connections in my knowledge base. ";
            response << "Could you tell me more about what you're thinking? ";
            response << "I'd love to dive deeper into this topic!";
        }
        
        return response.str();
    }
    
    std::string get_random_template(const std::string& category) {
        if (response_templates.find(category) != response_templates.end()) {
            const auto& templates = response_templates[category];
            std::uniform_int_distribution<> dis(0, templates.size() - 1);
            return templates[dis(gen)];
        }
        return "";
    }
    
    std::string get_random_fact() {
        if (knowledge_base.find("facts") != knowledge_base.end()) {
            const auto& facts = knowledge_base["facts"];
            std::uniform_int_distribution<> dis(0, facts.size() - 1);
            return facts[dis(gen)];
        }
        return "The human brain is incredibly complex and fascinating!";
    }
    
    void show_system_status() {
        std::cout << "\n📊 MELVIN ENHANCED SYSTEM STATUS" << std::endl;
        std::cout << "================================" << std::endl;
        std::cout << "Conversation turns: " << conversation_turn << std::endl;
        std::cout << "Knowledge topics: " << knowledge_base.size() << std::endl;
        std::cout << "Personality mode: " << personality_mode << std::endl;
        std::cout << "Session duration: " << std::fixed << std::setprecision(1) 
                  << (static_cast<double>(std::time(nullptr)) - session_start_time) << " seconds" << std::endl;
        
        std::cout << "\nRecent Conversation:" << std::endl;
        for (size_t i = std::max(0, static_cast<int>(conversation_pairs.size()) - 3); 
             i < conversation_pairs.size(); ++i) {
            std::cout << "You: " << conversation_pairs[i].first.substr(0, 50) << "..." << std::endl;
            std::cout << "Melvin: " << conversation_pairs[i].second.substr(0, 50) << "..." << std::endl;
        }
    }
    
    void run_interactive_session() {
        std::cout << "🧠 MELVIN ENHANCED INTERACTIVE SYSTEM" << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "Hello! I'm Melvin, your AI companion with enhanced personality!" << std::endl;
        std::cout << "I have knowledge about:" << std::endl;
        std::cout << "- Health & Medicine" << std::endl;
        std::cout << "- Science & Space" << std::endl;
        std::cout << "- Technology & AI" << std::endl;
        std::cout << "- General Knowledge" << std::endl;
        std::cout << "\nType 'quit' to exit, 'status' for system info, 'help' for commands." << std::endl;
        std::cout << "=====================================" << std::endl;
        
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
                std::cout << "\nMelvin: Thank you for this wonderful conversation! ";
                std::cout << "I've learned so much from our " << conversation_turn << " turns together. ";
                std::cout << "I'm grateful for the experience and look forward to our next chat! ";
                std::cout << "Take care! 🧠✨" << std::endl;
                break;
            } else if (lower_input == "status") {
                show_system_status();
                continue;
            } else if (lower_input == "help") {
                std::cout << "\nMelvin: Here are some things you can try:" << std::endl;
                std::cout << "- Ask me about health, science, or technology" << std::endl;
                std::cout << "- Request interesting facts" << std::endl;
                std::cout << "- Have philosophical discussions" << std::endl;
                std::cout << "- Ask about my knowledge and capabilities" << std::endl;
                std::cout << "- Type 'status' to see my current state" << std::endl;
                continue;
            }
            
            // Process input through enhanced system
            std::cout << "\nMelvin: ";
            std::string response = process_input(user_input);
            std::cout << response << std::endl;
            
            // Add a small delay to simulate thinking
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }
    }
};

int main() {
    try {
        ImprovedMelvin melvin;
        melvin.run_interactive_session();
    } catch (const std::exception& e) {
        std::cerr << "❌ Error in enhanced interactive session: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
