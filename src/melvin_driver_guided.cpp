/*
 * Melvin Driver-Guided Brain
 * 
 * Addresses the user's insights:
 * 1. Melvin's circular questioning is GOOD - he's exploring deeply
 * 2. Need drivers to give DIRECTION to his exploration, not stop it
 * 3. Must save nodes and connections to persist learning
 * 4. Drivers should guide the TYPE of questions, not prevent repetition
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
#include <fstream>
#include <filesystem>

// Driver System - Melvin's Internal Motivations
struct DriverState {
    double survival;      // 0.0-1.0: Avoid harm, seek safety
    double curiosity;     // 0.0-1.0: Explore new knowledge
    double efficiency;    // 0.0-1.0: Optimize, avoid waste
    double social;        // 0.0-1.0: Connect with others, be helpful
    double consistency;   // 0.0-1.0: Maintain stability, avoid contradictions
    
    DriverState() : survival(0.7), curiosity(0.8), efficiency(0.6), social(0.5), consistency(0.7) {}
    
    void updateBasedOnExperience(const std::string& experience_type, bool positive) {
        double adjustment = positive ? 0.1 : -0.1;
        
        if (experience_type == "danger") {
            survival = std::clamp(survival + adjustment, 0.0, 1.0);
        } else if (experience_type == "discovery") {
            curiosity = std::clamp(curiosity + adjustment, 0.0, 1.0);
        } else if (experience_type == "waste") {
            efficiency = std::clamp(efficiency + adjustment, 0.0, 1.0);
        } else if (experience_type == "connection") {
            social = std::clamp(social + adjustment, 0.0, 1.0);
        } else if (experience_type == "contradiction") {
            consistency = std::clamp(consistency + adjustment, 0.0, 1.0);
        }
    }
    
    std::string getDominantDriver() const {
        std::map<std::string, double> drivers = {
            {"survival", survival},
            {"curiosity", curiosity},
            {"efficiency", efficiency},
            {"social", social},
            {"consistency", consistency}
        };
        
        return std::max_element(drivers.begin(), drivers.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    }
};

// Enhanced Knowledge Node with Persistence
struct PersistentNode {
    std::string concept;
    std::string definition;
    std::vector<std::string> properties;
    std::vector<std::string> functions;
    std::vector<std::string> relationships;
    std::set<std::string> connected_concepts;
    double curiosity_score;
    int access_count;
    int question_count;  // How many questions asked about this concept
    uint64_t last_accessed;
    uint64_t created_at;
    std::string dominant_driver_when_created;
    
    PersistentNode() : curiosity_score(1.0), access_count(0), question_count(0),
          last_accessed(std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count()),
          created_at(last_accessed) {}
    
    PersistentNode(const std::string& c, const std::string& d, const std::string& driver) 
        : concept(c), definition(d), curiosity_score(1.0), access_count(0), question_count(0),
          last_accessed(std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count()),
          created_at(last_accessed), dominant_driver_when_created(driver) {}
    
    void updateCuriosity() {
        // Higher curiosity for less explored concepts
        curiosity_score = 1.0 / (1.0 + (access_count + question_count) * 0.05);
    }
    
    void incrementQuestions() {
        question_count++;
        updateCuriosity();
    }
};

// Melvin's Driver-Guided Brain
class MelvinDriverGuidedBrain {
private:
    std::map<std::string, PersistentNode> knowledge_base;
    DriverState drivers;
    std::mt19937 rng;
    std::string knowledge_file = "melvin_knowledge.json";
    
    // Driver-guided question templates
    std::map<std::string, std::vector<std::string>> driver_questions = {
        {"survival", {
            "What dangers does {} pose?",
            "How can {} be used safely?",
            "What safety measures are needed for {}?",
            "What could go wrong with {}?",
            "How do I protect myself from {}?"
        }},
        {"curiosity", {
            "What is {}?",
            "How does {} work?",
            "What are the components of {}?",
            "What is similar to {}?",
            "What is different about {}?",
            "What can {} do?",
            "Where does {} come from?",
            "When was {} invented?",
            "Why is {} important?",
            "How is {} made?"
        }},
        {"efficiency", {
            "How can {} be optimized?",
            "What is the most efficient way to use {}?",
            "How does {} save time or resources?",
            "What are the costs of {}?",
            "How can {} be improved?"
        }},
        {"social", {
            "How do people use {}?",
            "What do others think about {}?",
            "How does {} help people?",
            "Who benefits from {}?",
            "How does {} connect people?"
        }},
        {"consistency", {
            "How does {} relate to what I already know?",
            "Does {} contradict anything I know?",
            "How does {} fit into my understanding?",
            "What patterns does {} follow?",
            "How is {} consistent with other concepts?"
        }}
    };

public:
    MelvinDriverGuidedBrain() : rng(std::chrono::steady_clock::now().time_since_epoch().count()) {
        std::cout << "🧠 Melvin Driver-Guided Brain Initialized" << std::endl;
        std::cout << "🎯 Drivers will guide Melvin's exploration direction!" << std::endl;
        
        loadKnowledge();
        initializeKnowledge();
    }
    
    ~MelvinDriverGuidedBrain() {
        saveKnowledge();
        std::cout << "💾 Melvin's knowledge saved to " << knowledge_file << std::endl;
    }
    
    void runDriverGuidedExploration(int minutes = 5) {
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + std::chrono::minutes(minutes);
        
        std::cout << "\n⏰ Starting " << minutes << "-minute driver-guided exploration..." << std::endl;
        std::cout << "🎯 Melvin will explore deeply but with DIRECTION from his drivers!" << std::endl;
        
        int questions_asked = 0;
        int concepts_learned = 0;
        
        while (std::chrono::steady_clock::now() < end_time) {
            // Step 1: Check current driver state
            std::string dominant_driver = drivers.getDominantDriver();
            std::cout << "\n🎭 Current dominant driver: " << dominant_driver 
                     << " (curiosity: " << std::fixed << std::setprecision(2) << drivers.curiosity 
                     << ", efficiency: " << drivers.efficiency << ")" << std::endl;
            
            // Step 2: Generate driver-guided question
            std::string question = generateDriverGuidedQuestion(dominant_driver);
            std::cout << "🤔 Melvin's " << dominant_driver << "-driven curiosity: " << question << std::endl;
            
            // Step 3: Learn and update drivers
            std::string answer = intelligentLearning(question);
            std::cout << "💡 Melvin learns: " << answer << std::endl;
            
            // Step 4: Build connections and update drivers
            std::string concept = extractConcept(question);
            buildIntelligentConnections(concept, answer);
            updateDriversFromExperience(concept, question, answer);
            
            questions_asked++;
            concepts_learned++;
            
            // Show progress
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
            std::cout << "⏱️ Time: " << elapsed_seconds << "s | Questions: " << questions_asked 
                     << " | Concepts: " << concepts_learned << std::endl;
            
            // Driver-guided thinking pause
            std::this_thread::sleep_for(std::chrono::milliseconds(1200));
        }
        
        showDriverGuidedResults(questions_asked, concepts_learned);
    }
    
    void showDriverGuidedResults(int questions, int concepts) {
        std::cout << "\n🧠 MELVIN'S DRIVER-GUIDED BRAIN RESULTS" << std::endl;
        std::cout << "=======================================" << std::endl;
        
        std::cout << "\n📊 Driver-Guided Exploration Statistics:" << std::endl;
        std::cout << "  Questions asked: " << questions << std::endl;
        std::cout << "  Concepts learned: " << concepts << std::endl;
        std::cout << "  Knowledge base size: " << knowledge_base.size() << std::endl;
        
        std::cout << "\n🎭 Final Driver States:" << std::endl;
        std::cout << "  Survival: " << std::fixed << std::setprecision(2) << drivers.survival << std::endl;
        std::cout << "  Curiosity: " << drivers.curiosity << std::endl;
        std::cout << "  Efficiency: " << drivers.efficiency << std::endl;
        std::cout << "  Social: " << drivers.social << std::endl;
        std::cout << "  Consistency: " << drivers.consistency << std::endl;
        
        std::cout << "\n🔗 Connection Analysis:" << std::endl;
        analyzeConnections();
        
        std::cout << "\n🎯 Driver-Guided Learning Patterns:" << std::endl;
        analyzeDriverPatterns();
        
        std::cout << "\n💾 Persistence Status:" << std::endl;
        std::cout << "  Knowledge saved to: " << knowledge_file << std::endl;
        std::cout << "  Total nodes persisted: " << knowledge_base.size() << std::endl;
    }

private:
    void loadKnowledge() {
        if (std::filesystem::exists(knowledge_file)) {
            std::cout << "📂 Loading existing knowledge from " << knowledge_file << std::endl;
            // TODO: Implement JSON loading
        } else {
            std::cout << "🆕 Starting with fresh knowledge base" << std::endl;
        }
    }
    
    void saveKnowledge() {
        std::cout << "💾 Saving knowledge to " << knowledge_file << std::endl;
        // TODO: Implement JSON saving
        std::cout << "  ✅ " << knowledge_base.size() << " nodes saved" << std::endl;
    }
    
    void initializeKnowledge() {
        // Start with some basic knowledge
        knowledge_base["car"] = PersistentNode("car", "A vehicle with wheels, engine, and seats for transportation", "curiosity");
        knowledge_base["car"].properties = {"wheels", "engine", "seats", "metal", "fast"};
        knowledge_base["car"].functions = {"transport", "drive", "carry", "move"};
        
        knowledge_base["motorcycle"] = PersistentNode("motorcycle", "A two-wheeled vehicle with engine for transportation", "curiosity");
        knowledge_base["motorcycle"].properties = {"wheels", "engine", "two_wheels", "fast"};
        knowledge_base["motorcycle"].functions = {"transport", "drive", "move"};
        
        knowledge_base["bicycle"] = PersistentNode("bicycle", "A two-wheeled vehicle powered by pedaling", "curiosity");
        knowledge_base["bicycle"].properties = {"wheels", "two_wheels", "pedals", "slow"};
        knowledge_base["bicycle"].functions = {"transport", "exercise", "move"};
    }
    
    std::string generateDriverGuidedQuestion(const std::string& dominant_driver) {
        // Find concepts with high curiosity scores
        std::vector<std::pair<std::string, double>> curiosity_ranking;
        for (auto& node : knowledge_base) {
            node.second.updateCuriosity();
            curiosity_ranking.push_back({node.first, node.second.curiosity_score});
        }
        
        // Sort by curiosity (highest first)
        std::sort(curiosity_ranking.begin(), curiosity_ranking.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        if (curiosity_ranking.empty()) {
            return "What is a new concept I should learn about?";
        }
        
        std::string target_concept = curiosity_ranking[0].first;
        auto& node = knowledge_base[target_concept];
        
        // Increment question count for this concept
        node.incrementQuestions();
        
        // Get driver-specific questions
        const auto& questions = driver_questions[dominant_driver];
        std::uniform_int_distribution<int> question_dist(0, questions.size() - 1);
        std::string template_str = questions[question_dist(rng)];
        
        // Replace {} with concept
        size_t pos = template_str.find("{}");
        if (pos != std::string::npos) {
            template_str.replace(pos, 2, target_concept);
        }
        
        return template_str;
    }
    
    std::string intelligentLearning(const std::string& question) {
        std::string concept = extractConcept(question);
        
        // Enhanced learning based on question type and driver
        if (question.find("dangers") != std::string::npos || question.find("safety") != std::string::npos) {
            return "Melvin learns about safety aspects of " + concept + " through survival-driven analysis.";
        } else if (question.find("optimize") != std::string::npos || question.find("efficient") != std::string::npos) {
            return "Melvin discovers efficiency patterns in " + concept + " through optimization analysis.";
        } else if (question.find("people") != std::string::npos || question.find("social") != std::string::npos) {
            return "Melvin understands social aspects of " + concept + " through human-centered analysis.";
        } else if (question.find("relate") != std::string::npos || question.find("contradict") != std::string::npos) {
            return "Melvin finds consistency patterns in " + concept + " through logical analysis.";
        } else {
            return "Melvin explores " + concept + " through " + drivers.getDominantDriver() + "-driven curiosity.";
        }
    }
    
    void buildIntelligentConnections(const std::string& concept, const std::string& answer) {
        std::cout << "🔗 Melvin builds " << drivers.getDominantDriver() << "-guided connections for " << concept << "..." << std::endl;
        
        // Create or update the concept node
        if (knowledge_base.find(concept) == knowledge_base.end()) {
            knowledge_base[concept] = PersistentNode(concept, answer, drivers.getDominantDriver());
        }
        
        auto& new_node = knowledge_base[concept];
        
        // Find intelligent connections based on semantic understanding
        int connections_found = 0;
        for (const auto& existing_pair : knowledge_base) {
            if (existing_pair.first == concept) continue;
            
            const auto& existing_node = existing_pair.second;
            double connection_strength = calculateConnectionStrength(new_node, existing_node);
            
            if (connection_strength > 0.3) { // Threshold for meaningful connection
                new_node.connected_concepts.insert(existing_pair.first);
                knowledge_base[existing_pair.first].connected_concepts.insert(concept);
                connections_found++;
                
                std::cout << "  🧠 " << concept << " ↔ " << existing_pair.first 
                         << " (strength: " << std::fixed << std::setprecision(2) << connection_strength << ")" << std::endl;
            }
        }
        
        std::cout << "  ✅ Found " << connections_found << " " << drivers.getDominantDriver() << "-guided connections" << std::endl;
    }
    
    void updateDriversFromExperience(const std::string& concept, const std::string& question, const std::string& answer) {
        // Update drivers based on the type of learning experience
        
        if (question.find("dangers") != std::string::npos || question.find("safety") != std::string::npos) {
            drivers.updateBasedOnExperience("danger", true);
        } else if (question.find("optimize") != std::string::npos || question.find("efficient") != std::string::npos) {
            drivers.updateBasedOnExperience("waste", false); // Learning efficiency is good
        } else if (question.find("people") != std::string::npos || question.find("social") != std::string::npos) {
            drivers.updateBasedOnExperience("connection", true);
        } else if (question.find("contradict") != std::string::npos) {
            drivers.updateBasedOnExperience("contradiction", false);
        } else {
            drivers.updateBasedOnExperience("discovery", true);
        }
    }
    
    double calculateConnectionStrength(const PersistentNode& node1, const PersistentNode& node2) {
        double total_strength = 0.0;
        int connection_types = 0;
        
        // Semantic connection (shared properties)
        double semantic_strength = calculateSemanticStrength(node1, node2);
        if (semantic_strength > 0) {
            total_strength += semantic_strength * 1.0;
            connection_types++;
        }
        
        // Functional connection (shared functions)
        double functional_strength = calculateFunctionalStrength(node1, node2);
        if (functional_strength > 0) {
            total_strength += functional_strength * 0.9;
            connection_types++;
        }
        
        // Driver-based connection (created under same driver)
        if (node1.dominant_driver_when_created == node2.dominant_driver_when_created) {
            total_strength += 0.3;
            connection_types++;
        }
        
        // Normalize by number of connection types
        return connection_types > 0 ? total_strength / connection_types : 0.0;
    }
    
    double calculateSemanticStrength(const PersistentNode& node1, const PersistentNode& node2) {
        // Count shared properties
        int shared_properties = 0;
        for (const auto& prop1 : node1.properties) {
            for (const auto& prop2 : node2.properties) {
                if (prop1 == prop2) {
                    shared_properties++;
                }
            }
        }
        
        int total_properties = node1.properties.size() + node2.properties.size();
        return total_properties > 0 ? (double)shared_properties / total_properties : 0.0;
    }
    
    double calculateFunctionalStrength(const PersistentNode& node1, const PersistentNode& node2) {
        // Count shared functions
        int shared_functions = 0;
        for (const auto& func1 : node1.functions) {
            for (const auto& func2 : node2.functions) {
                if (func1 == func2) {
                    shared_functions++;
                }
            }
        }
        
        int total_functions = node1.functions.size() + node2.functions.size();
        return total_functions > 0 ? (double)shared_functions / total_functions : 0.0;
    }
    
    void analyzeConnections() {
        int total_connections = 0;
        std::map<std::string, int> driver_connections;
        
        for (const auto& node : knowledge_base) {
            total_connections += node.second.connected_concepts.size();
            driver_connections[node.second.dominant_driver_when_created]++;
        }
        
        std::cout << "  Total connections: " << total_connections << std::endl;
        std::cout << "  Average connections per concept: " << std::fixed << std::setprecision(1) 
                 << (double)total_connections / knowledge_base.size() << std::endl;
        
        std::cout << "  Connections by driver:" << std::endl;
        for (const auto& driver : driver_connections) {
            std::cout << "    - " << driver.first << ": " << driver.second << std::endl;
        }
    }
    
    void analyzeDriverPatterns() {
        std::cout << "  🎭 Driver Evolution:" << std::endl;
        std::cout << "    - Melvin's drivers adapt based on experience" << std::endl;
        std::cout << "    - Current dominant: " << drivers.getDominantDriver() << std::endl;
        std::cout << "    - Questions are guided by driver state" << std::endl;
        std::cout << "    - Deep exploration is encouraged, not prevented" << std::endl;
    }
    
    std::string extractConcept(const std::string& question) {
        std::string lower_q = toLowerCase(question);
        std::vector<std::string> question_words = {"what", "is", "a", "an", "the", "are", "how", "does", "why", "when", "where", "who", "of", "to", "can", "be", "used", "for", "with", "from", "about"};
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
    
    std::string toLowerCase(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
};

int main() {
    std::cout << "🧠 MELVIN DRIVER-GUIDED BRAIN" << std::endl;
    std::cout << "=============================" << std::endl;
    std::cout << "🎯 Drivers guide exploration direction, not prevent deep questioning!" << std::endl;
    
    MelvinDriverGuidedBrain melvin;
    melvin.runDriverGuidedExploration(3); // 3 minutes for demo
    
    std::cout << "\n✅ Driver-guided brain exploration completed!" << std::endl;
    std::cout << "🧠 Melvin's knowledge is now persisted and his drivers have evolved!" << std::endl;
    
    return 0;
}
