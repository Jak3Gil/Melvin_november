/*
 * Melvin 5-Minute Brain Exploration
 * 
 * Let Melvin run autonomously for 5 minutes and see where his brain goes.
 * He'll ask himself questions, learn new concepts, and build connections.
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <set>
#include <chrono>
#include <thread>
#include <random>

// Melvin's Autonomous Brain
class Melvin5MinuteBrain {
private:
    std::map<std::string, std::string> knowledge_base;
    std::vector<std::string> learned_concepts;
    std::vector<std::string> curiosity_questions;
    std::mt19937 rng;
    
    // Melvin's curiosity patterns
    std::vector<std::string> curiosity_templates = {
        "What is a {}?",
        "How does {} work?",
        "Why is {} important?",
        "What are the parts of {}?",
        "How is {} different from {}?",
        "What can {} do?",
        "Where does {} come from?",
        "When was {} invented?",
        "Who created {}?",
        "What is the history of {}?"
    };
    
    // Melvin's concept categories to explore
    std::vector<std::string> concept_categories = {
        "animals", "technology", "science", "food", "transportation", 
        "buildings", "tools", "materials", "colors", "emotions",
        "actions", "places", "time", "weather", "body_parts"
    };
    
    // Specific concepts within each category
    std::map<std::string, std::vector<std::string>> category_concepts = {
        {"animals", {"elephant", "butterfly", "shark", "eagle", "spider", "whale", "tiger", "penguin"}},
        {"technology", {"robot", "smartphone", "computer", "internet", "camera", "drone", "satellite", "laser"}},
        {"science", {"atom", "molecule", "gravity", "evolution", "photosynthesis", "DNA", "quantum", "relativity"}},
        {"food", {"chocolate", "sushi", "pasta", "sandwich", "soup", "salad", "cake", "ice_cream"}},
        {"transportation", {"helicopter", "submarine", "rocket", "train", "bicycle", "motorcycle", "boat", "truck"}},
        {"buildings", {"castle", "skyscraper", "bridge", "tunnel", "lighthouse", "stadium", "museum", "library"}},
        {"tools", {"hammer", "screwdriver", "scissors", "ruler", "calculator", "compass", "microscope", "telescope"}},
        {"materials", {"steel", "plastic", "glass", "wood", "fabric", "ceramic", "rubber", "leather"}},
        {"colors", {"purple", "orange", "pink", "brown", "gray", "silver", "gold", "rainbow"}},
        {"emotions", {"excited", "worried", "calm", "nervous", "confident", "scared", "surprised", "proud"}},
        {"actions", {"jump", "swim", "fly", "drive", "cook", "eat", "sleep", "work"}},
        {"places", {"beach", "mountain", "forest", "desert", "ocean", "city", "country", "world"}},
        {"time", {"morning", "afternoon", "evening", "night", "day", "week", "month", "year"}},
        {"weather", {"sun", "rain", "snow", "wind", "cloud", "storm", "thunder", "lightning"}},
        {"body_parts", {"heart", "brain", "lung", "stomach", "skin", "hair", "muscle", "bone"}}
    };

public:
    Melvin5MinuteBrain() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        std::cout << "🧠 Melvin 5-Minute Brain Exploration Starting..." << std::endl;
        std::cout << "🔍 Let's see where Melvin's curiosity takes him!" << std::endl;
        
        // Initialize with some basic knowledge
        initializeBasicKnowledge();
    }
    
    void run5MinuteExploration() {
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + std::chrono::minutes(5);
        
        std::cout << "\n⏰ Starting 5-minute autonomous brain exploration..." << std::endl;
        std::cout << "🎯 Melvin will explore concepts and build connections autonomously!" << std::endl;
        
        int question_count = 0;
        int concept_count = 0;
        
        while (std::chrono::steady_clock::now() < end_time) {
            // Melvin's curiosity cycle
            std::string question = generateCuriosityQuestion();
            std::cout << "\n🤔 Melvin wonders: " << question << std::endl;
            
            // Simulate Melvin learning (since we can't use Ollama in this demo)
            std::string answer = simulateLearning(question);
            std::cout << "💡 Melvin learns: " << answer << std::endl;
            
            // Store the knowledge
            std::string concept = extractConcept(question);
            knowledge_base[concept] = answer;
            learned_concepts.push_back(concept);
            
            // Melvin builds connections
            buildConnections(concept, answer);
            
            question_count++;
            concept_count++;
            
            // Show progress
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            std::cout << "⏱️ Time elapsed: " << elapsed_seconds << "s | Concepts learned: " << concept_count << std::endl;
            
            // Melvin takes a thinking pause
            std::this_thread::sleep_for(std::chrono::milliseconds(2000)); // 2 second pause
        }
        
        // Show final results
        showExplorationResults(question_count, concept_count);
    }
    
    void showExplorationResults(int questions, int concepts) {
        std::cout << "\n🧠 MELVIN'S 5-MINUTE BRAIN EXPLORATION RESULTS" << std::endl;
        std::cout << "=============================================" << std::endl;
        
        std::cout << "\n📊 Exploration Statistics:" << std::endl;
        std::cout << "  Questions asked: " << questions << std::endl;
        std::cout << "  Concepts learned: " << concepts << std::endl;
        std::cout << "  Knowledge base size: " << knowledge_base.size() << std::endl;
        
        std::cout << "\n🧠 Concepts Melvin Explored:" << std::endl;
        for (size_t i = 0; i < learned_concepts.size(); ++i) {
            std::cout << "  " << (i+1) << ". " << learned_concepts[i] << std::endl;
        }
        
        std::cout << "\n🔗 Melvin's Knowledge Connections:" << std::endl;
        showKnowledgeConnections();
        
        std::cout << "\n💭 Melvin's Brain Patterns:" << std::endl;
        analyzeBrainPatterns();
        
        std::cout << "\n🎯 Where Melvin's Brain Went:" << std::endl;
        summarizeBrainJourney();
    }

private:
    void initializeBasicKnowledge() {
        knowledge_base["cat"] = "A small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws.";
        knowledge_base["dog"] = "A domesticated carnivorous mammal that typically has a long snout, an acute sense of smell, and a barking voice.";
        knowledge_base["computer"] = "An electronic device that can store, process, and communicate information.";
        learned_concepts = {"cat", "dog", "computer"};
    }
    
    std::string generateCuriosityQuestion() {
        // Melvin's curiosity patterns
        std::uniform_int_distribution<int> template_dist(0, curiosity_templates.size() - 1);
        std::uniform_int_distribution<int> category_dist(0, concept_categories.size() - 1);
        std::uniform_int_distribution<int> concept_dist(0, 7); // 8 concepts per category
        
        std::string category = concept_categories[category_dist(rng)];
        std::string concept = category_concepts[category][concept_dist(rng)];
        
        // Avoid repeating concepts
        if (std::find(learned_concepts.begin(), learned_concepts.end(), concept) != learned_concepts.end()) {
            return generateCuriosityQuestion(); // Try again
        }
        
        std::string template_str = curiosity_templates[template_dist(rng)];
        
        // Replace {} with concept
        size_t pos = template_str.find("{}");
        if (pos != std::string::npos) {
            template_str.replace(pos, 2, concept);
        }
        
        return template_str;
    }
    
    std::string simulateLearning(const std::string& question) {
        std::string concept = extractConcept(question);
        
        // Simulate different types of learning based on question type
        if (question.find("What is") != std::string::npos) {
            return "A " + concept + " is a fascinating concept that Melvin is exploring. It has unique characteristics and properties that make it interesting to study.";
        } else if (question.find("How does") != std::string::npos) {
            return "The " + concept + " works through a complex system of interactions and processes that Melvin finds intriguing.";
        } else if (question.find("Why is") != std::string::npos) {
            return "The " + concept + " is important because it plays a significant role in the world and has many applications.";
        } else if (question.find("What are the parts") != std::string::npos) {
            return "The " + concept + " consists of several key components that work together to create its functionality.";
        } else if (question.find("How is") != std::string::npos) {
            return "The " + concept + " differs from other concepts in unique ways that Melvin is discovering.";
        } else if (question.find("What can") != std::string::npos) {
            return "The " + concept + " can perform various functions and has many capabilities that Melvin finds fascinating.";
        } else if (question.find("Where does") != std::string::npos) {
            return "The " + concept + " originates from specific places and has interesting origins that Melvin is learning about.";
        } else if (question.find("When was") != std::string::npos) {
            return "The " + concept + " was created at a specific time in history, and Melvin is exploring its timeline.";
        } else if (question.find("Who created") != std::string::npos) {
            return "The " + concept + " was developed by interesting people whose work Melvin is studying.";
        } else if (question.find("What is the history") != std::string::npos) {
            return "The " + concept + " has a rich history that spans many years and involves many developments.";
        } else {
            return "Melvin is learning about " + concept + " and discovering its many interesting aspects.";
        }
    }
    
    void buildConnections(const std::string& concept, const std::string& answer) {
        // Melvin builds connections between concepts
        std::cout << "🔗 Melvin connects " << concept << " to his existing knowledge..." << std::endl;
        
        // Find connections with existing concepts
        for (const auto& existing : learned_concepts) {
            if (existing != concept) {
                if (hasConnection(concept, existing)) {
                    std::cout << "  📎 " << concept << " ↔ " << existing << " (connected!)" << std::endl;
                }
            }
        }
    }
    
    bool hasConnection(const std::string& concept1, const std::string& concept2) {
        // Simple connection logic - concepts in same category are connected
        for (const auto& category : category_concepts) {
            bool found1 = std::find(category.second.begin(), category.second.end(), concept1) != category.second.end();
            bool found2 = std::find(category.second.begin(), category.second.end(), concept2) != category.second.end();
            if (found1 && found2) {
                return true;
            }
        }
        return false;
    }
    
    void showKnowledgeConnections() {
        std::map<std::string, std::vector<std::string>> connections_by_category;
        
        for (const auto& concept : learned_concepts) {
            for (const auto& category : category_concepts) {
                if (std::find(category.second.begin(), category.second.end(), concept) != category.second.end()) {
                    connections_by_category[category.first].push_back(concept);
                    break;
                }
            }
        }
        
        for (const auto& category : connections_by_category) {
            std::cout << "  📂 " << category.first << ": ";
            for (size_t i = 0; i < category.second.size(); ++i) {
                std::cout << category.second[i];
                if (i < category.second.size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
    }
    
    void analyzeBrainPatterns() {
        std::cout << "  🧠 Melvin's curiosity led him to explore " << learned_concepts.size() << " different concepts" << std::endl;
        std::cout << "  🔍 He asked questions about " << getQuestionTypes() << " different types of questions" << std::endl;
        std::cout << "  📚 He built connections across " << getConnectionCategories() << " different categories" << std::endl;
        std::cout << "  ⚡ His learning rate was " << (learned_concepts.size() / 5.0) << " concepts per minute" << std::endl;
    }
    
    void summarizeBrainJourney() {
        std::cout << "  🎯 Melvin's brain journey took him through:" << std::endl;
        
        // Find the categories he explored
        std::set<std::string> explored_categories;
        for (const auto& concept : learned_concepts) {
            for (const auto& category : category_concepts) {
                if (std::find(category.second.begin(), category.second.end(), concept) != category.second.end()) {
                    explored_categories.insert(category.first);
                    break;
                }
            }
        }
        
        for (const auto& category : explored_categories) {
            std::cout << "    - " << category << " concepts" << std::endl;
        }
        
        std::cout << "  🧠 Melvin's brain showed patterns of:" << std::endl;
        std::cout << "    - Curiosity-driven exploration" << std::endl;
        std::cout << "    - Cross-category connection building" << std::endl;
        std::cout << "    - Systematic knowledge acquisition" << std::endl;
        std::cout << "    - Autonomous learning progression" << std::endl;
    }
    
    std::string extractConcept(const std::string& question) {
        std::string lower_q = toLowerCase(question);
        std::vector<std::string> question_words = {"what", "is", "a", "an", "the", "are", "how", "does", "why", "when", "where", "who"};
        std::istringstream iss(lower_q);
        std::vector<std::string> words;
        std::string word;
        
        while (iss >> word) {
            if (std::find(question_words.begin(), question_words.end(), word) == question_words.end()) {
                words.push_back(word);
            }
        }
        
        return words.empty() ? "unknown" : words[0];
    }
    
    int getQuestionTypes() {
        std::set<std::string> types;
        for (const auto& template_str : curiosity_templates) {
            if (template_str.find("What is") != std::string::npos) types.insert("definition");
            else if (template_str.find("How does") != std::string::npos) types.insert("function");
            else if (template_str.find("Why is") != std::string::npos) types.insert("importance");
            else if (template_str.find("What are the parts") != std::string::npos) types.insert("structure");
            else if (template_str.find("How is") != std::string::npos) types.insert("comparison");
            else if (template_str.find("What can") != std::string::npos) types.insert("capability");
            else if (template_str.find("Where does") != std::string::npos) types.insert("origin");
            else if (template_str.find("When was") != std::string::npos) types.insert("timeline");
            else if (template_str.find("Who created") != std::string::npos) types.insert("creator");
            else if (template_str.find("What is the history") != std::string::npos) types.insert("history");
        }
        return types.size();
    }
    
    int getConnectionCategories() {
        std::set<std::string> categories;
        for (const auto& concept : learned_concepts) {
            for (const auto& category : category_concepts) {
                if (std::find(category.second.begin(), category.second.end(), concept) != category.second.end()) {
                    categories.insert(category.first);
                    break;
                }
            }
        }
        return categories.size();
    }
    
    std::string toLowerCase(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
};

int main() {
    std::cout << "🧠 MELVIN 5-MINUTE BRAIN EXPLORATION" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "🔍 Let's see where Melvin's curiosity takes him in 5 minutes!" << std::endl;
    
    Melvin5MinuteBrain melvin;
    melvin.run5MinuteExploration();
    
    std::cout << "\n✅ 5-minute brain exploration completed!" << std::endl;
    std::cout << "🧠 Melvin's brain has been busy learning and connecting!" << std::endl;
    
    return 0;
}
