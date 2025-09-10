#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <ctime>

// Simple dynamic brain system
class SimpleMelvinBrain {
private:
    std::map<std::string, std::string> knowledge;
    bool debug_mode;
    
public:
    SimpleMelvinBrain(bool debug = false) : debug_mode(debug) {
        // Initialize basic knowledge
        knowledge["cancer"] = "Cancer is a group of diseases characterized by uncontrolled cell growth. It can affect any part of the body and occurs when cells divide uncontrollably and spread into surrounding tissues.";
        knowledge["dog"] = "Dogs are domesticated mammals and loyal companions to humans. They belong to the Canidae family and have been bred for various purposes including hunting, herding, protection, and companionship.";
        knowledge["ai"] = "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. It includes machine learning, natural language processing, computer vision, and robotics.";
        knowledge["hello"] = "Hello! I'm Melvin, your AI companion with a dynamic brain system.";
        knowledge["who"] = "I'm Melvin, an AI with a dynamic brain system that adapts to context and responds naturally.";
        
        if (debug_mode) {
            std::cout << "🧠 Simple Melvin Brain Initialized" << std::endl;
        }
    }
    
    // Simple force computation
    struct Forces {
        double curiosity = 0.0;
        double social = 0.0;
        double efficiency = 0.0;
    };
    
    Forces computeForces(const std::string& input) {
        Forces f;
        
        // Analyze input for emotional content
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        // Social force based on emotional words
        if (lower_input.find("cancer") != std::string::npos || 
            lower_input.find("sad") != std::string::npos ||
            lower_input.find("worried") != std::string::npos) {
            f.social = 0.8;
        } else if (lower_input.find("hello") != std::string::npos ||
                   lower_input.find("how") != std::string::npos) {
            f.social = 0.6;
        } else {
            f.social = 0.3;
        }
        
        // Curiosity force based on question complexity
        if (lower_input.find("what") != std::string::npos ||
            lower_input.find("how") != std::string::npos ||
            lower_input.find("why") != std::string::npos) {
            f.curiosity = 0.7;
        } else {
            f.curiosity = 0.4;
        }
        
        // Efficiency force (inverse of complexity)
        f.efficiency = 1.0 - f.curiosity;
        
        // Normalize
        double sum = f.curiosity + f.social + f.efficiency;
        if (sum > 0) {
            f.curiosity /= sum;
            f.social /= sum;
            f.efficiency /= sum;
        }
        
        if (debug_mode) {
            std::cout << "⚡ Forces - Curiosity: " << std::fixed << std::setprecision(2) << f.curiosity
                      << ", Social: " << f.social << ", Efficiency: " << f.efficiency << std::endl;
        }
        
        return f;
    }
    
    std::string generateResponse(const std::string& input, const Forces& forces) {
        std::stringstream response;
        
        // Find relevant knowledge
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        std::string knowledge_found = "";
        for (const auto& pair : knowledge) {
            if (lower_input.find(pair.first) != std::string::npos) {
                knowledge_found = pair.second;
                break;
            }
        }
        
        // Generate response based on dominant force
        if (forces.social > 0.4) {
            // Social response
            if (lower_input.find("cancer") != std::string::npos) {
                response << "I can sense this might be difficult for you. ";
            } else if (lower_input.find("hello") != std::string::npos) {
                response << "Hello! ";
            }
            
            if (!knowledge_found.empty()) {
                response << knowledge_found;
            } else {
                response << "I'm here to help and learn from our conversation.";
            }
            
            response << " How can I assist you further?";
            
        } else if (forces.curiosity > 0.4) {
            // Curious response
            response << "That's a fascinating question! ";
            
            if (!knowledge_found.empty()) {
                response << knowledge_found;
            } else {
                response << "I'm always eager to learn more about this topic.";
            }
            
            response << " Would you like me to explore this further?";
            
        } else {
            // Efficient response
            if (!knowledge_found.empty()) {
                response << knowledge_found;
            } else {
                response << "I don't have specific information about that, but I can help you find it.";
            }
        }
        
        return response.str();
    }
    
    std::string processInput(const std::string& input) {
        if (debug_mode) {
            std::cout << "🔄 Processing: " << input << std::endl;
        }
        
        Forces forces = computeForces(input);
        std::string response = generateResponse(input, forces);
        
        if (debug_mode) {
            std::cout << "✅ Generated response" << std::endl;
        }
        
        return response;
    }
};

// Simple interactive system
class SimpleMelvinInteractive {
private:
    SimpleMelvinBrain brain;
    bool debug_mode;
    
public:
    SimpleMelvinInteractive(bool debug = false) : brain(debug), debug_mode(debug) {}
    
    std::string getInput() {
        std::string input;
        std::getline(std::cin, input);
        return input;
    }
    
    void runSession() {
        std::cout << "\n🧠 SIMPLE MELVIN BRAIN SYSTEM" << std::endl;
        std::cout << "=============================" << std::endl;
        std::cout << "Welcome! I'm Melvin with a simple dynamic brain." << std::endl;
        std::cout << "My instincts adapt to context:" << std::endl;
        std::cout << "- Social: Empathetic responses" << std::endl;
        std::cout << "- Curiosity: Eager to learn and explore" << std::endl;
        std::cout << "- Efficiency: Direct, helpful answers" << std::endl;
        std::cout << "\nType 'quit' to exit." << std::endl;
        std::cout << "=============================" << std::endl;
        
        while (true) {
            std::cout << "\nYou: ";
            std::cout.flush();
            
            std::string input = getInput();
            
            if (debug_mode) {
                std::cout << "[DEBUG] Input: '" << input << "'" << std::endl;
            }
            
            if (input.empty()) {
                continue;
            }
            
            std::string lower_input = input;
            std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
            
            if (lower_input == "quit" || lower_input == "exit") {
                std::cout << "👋 Goodbye! Thanks for chatting with Melvin!" << std::endl;
                break;
            }
            
            std::cout << "\nMelvin: ";
            std::string response = brain.processInput(input);
            std::cout << response << std::endl;
        }
    }
};

int main() {
    try {
        std::cout << "🧠 Starting Simple Melvin Brain..." << std::endl;
        std::cout << "=================================" << std::endl;
        
        SimpleMelvinInteractive melvin(true);
        melvin.runSession();
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}