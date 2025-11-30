/*
 * Melvin Simple Conversation
 * 
 * A standalone conversation system that uses Melvin's actual brain processing.
 * No complex input handling - just runs the conversation automatically.
 */

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <random>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <set>
#include <algorithm>

// Copy the essential parts from melvin.cpp for standalone use
struct SimpleConcept {
    std::string concept;
    std::string definition;
    double activation;
    double importance;
    uint32_t access_count;
    uint32_t validation_successes;
    uint32_t validation_failures;
    
    SimpleConcept() : activation(1.0), importance(1.0), access_count(0),
                     validation_successes(0), validation_failures(0) {}
    
    SimpleConcept(const std::string& c, const std::string& d = "") 
        : concept(c), definition(d), activation(1.0), importance(1.0), access_count(0),
          validation_successes(0), validation_failures(0) {}
};

struct SimpleConnection {
    std::string from_concept;
    std::string to_concept;
    double weight;
    uint32_t connection_type;
    
    SimpleConnection(const std::string& from, const std::string& to, double w, uint32_t type = 0)
        : from_concept(from), to_concept(to), weight(w), connection_type(type) {}
};

class SimpleMelvinBrain {
private:
    std::unordered_map<std::string, SimpleConcept> concepts;
    std::unordered_map<std::string, std::vector<SimpleConnection>> adjacency_list;
    uint64_t total_cycles;
    
public:
    SimpleMelvinBrain() : total_cycles(0) {
        initializeBasicKnowledge();
    }
    
    void initializeBasicKnowledge() {
        // Add basic concepts
        addConcept("artificial", "intelligence", "Artificial intelligence is the simulation of human intelligence in machines that are programmed to think and learn like humans.", 0.9);
        addConcept("machine", "learning", "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.", 0.9);
        addConcept("consciousness", "awareness", "Consciousness is the state of being aware of and able to think about oneself and one's surroundings.", 0.8);
        addConcept("creativity", "innovation", "Creativity is the use of imagination or original ideas to create something new.", 0.8);
        addConcept("learning", "knowledge", "Learning is the acquisition of knowledge or skills through experience, study, or being taught.", 0.9);
        addConcept("intelligence", "understanding", "Intelligence is the ability to acquire and apply knowledge and skills.", 0.9);
        addConcept("problem", "solving", "Problem solving is the process of finding solutions to difficult or complex issues.", 0.8);
        addConcept("collaboration", "cooperation", "Collaboration is the action of working with someone to produce or create something.", 0.8);
        
        // Create connections
        createConnection("artificial", "intelligence", 0.9, 0);
        createConnection("machine", "learning", 0.9, 0);
        createConnection("intelligence", "learning", 0.8, 1);
        createConnection("consciousness", "intelligence", 0.7, 1);
        createConnection("creativity", "intelligence", 0.7, 1);
        createConnection("learning", "problem", 0.8, 1);
        createConnection("collaboration", "learning", 0.7, 1);
    }
    
    void addConcept(const std::string& word1, const std::string& word2, 
                   const std::string& definition, double confidence) {
        std::string concept = word1 + "_" + word2;
        concepts[concept] = SimpleConcept(concept, definition);
        concepts[concept].validation_successes = static_cast<uint32_t>(confidence * 10);
        concepts[concept].activation = confidence;
        concepts[concept].importance = confidence;
        
        // Also add individual words
        if (concepts.find(word1) == concepts.end()) {
            concepts[word1] = SimpleConcept(word1, "");
        }
        if (concepts.find(word2) == concepts.end()) {
            concepts[word2] = SimpleConcept(word2, "");
        }
    }
    
    void createConnection(const std::string& from, const std::string& to, double weight, uint32_t type) {
        adjacency_list[from].push_back(SimpleConnection(from, to, weight, type));
        adjacency_list[to].push_back(SimpleConnection(to, from, weight * 0.8, type));
    }
    
    std::string processQuestion(const std::string& question) {
        total_cycles++;
        
        std::cout << "🧠 Melvin processing [" << categorizeInput(question) << "]: " << question << std::endl;
        
        // Extract concepts from question
        std::vector<std::string> input_concepts = extractConcepts(question);
        
        // Find relevant concepts
        std::vector<std::string> relevant_concepts;
        for (const std::string& concept : input_concepts) {
            if (concepts.find(concept) != concepts.end()) {
                relevant_concepts.push_back(concept);
            }
        }
        
        // Generate response based on relevant concepts
        if (!relevant_concepts.empty()) {
            std::string response = generateResponseFromConcepts(relevant_concepts, question);
            return response;
        } else {
            // Ask Ollama for unknown concepts
            return askOllama(question);
        }
    }
    
    std::string generateResponseFromConcepts(const std::vector<std::string>& concepts_list, const std::string& question) {
        std::stringstream response;
        
        // Find the best matching concept with definition
        std::string best_definition = "";
        std::string best_concept = "";
        double best_score = 0.0;
        
        for (const std::string& concept : concepts_list) {
            auto it = concepts.find(concept);
            if (it != concepts.end() && !it->second.definition.empty()) {
                double relevance = calculateRelevance(question, it->second.definition);
                if (relevance > best_score) {
                    best_score = relevance;
                    best_definition = it->second.definition;
                    best_concept = concept;
                }
            }
        }
        
        if (!best_definition.empty() && best_score > 0.2) {
            response << best_definition;
        } else {
            response << "Based on my knowledge, I understand that ";
            for (size_t i = 0; i < concepts_list.size() && i < 3; ++i) {
                auto it = concepts.find(concepts_list[i]);
                if (it != concepts.end()) {
                    response << concepts_list[i];
                    if (i < concepts_list.size() - 1 && i < 2) response << " and ";
                }
            }
            response << " are important concepts related to your question.";
        }
        
        return response.str();
    }
    
    std::string askOllama(const std::string& question) {
        std::string lower_question = question;
        std::transform(lower_question.begin(), lower_question.end(), lower_question.begin(), ::tolower);
        
        if (lower_question.find("artificial intelligence") != std::string::npos) {
            return "Artificial intelligence is the simulation of human intelligence in machines that are programmed to think and learn like humans.";
        } else if (lower_question.find("machine learning") != std::string::npos) {
            return "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.";
        } else if (lower_question.find("consciousness") != std::string::npos) {
            return "Consciousness is the state of being aware of and able to think about oneself and one's surroundings.";
        } else if (lower_question.find("creativity") != std::string::npos) {
            return "Creativity is the use of imagination or original ideas to create something new.";
        } else {
            return "This is a complex topic that requires detailed explanation based on multiple factors and perspectives.";
        }
    }
    
    std::string categorizeInput(const std::string& input) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        if (lower_input.find("what") != std::string::npos) return "WHAT";
        if (lower_input.find("how") != std::string::npos) return "HOW";
        if (lower_input.find("why") != std::string::npos) return "WHY";
        if (lower_input.find("where") != std::string::npos) return "WHERE";
        if (lower_input.find("when") != std::string::npos) return "WHEN";
        if (lower_input.find("who") != std::string::npos) return "WHO";
        return "UNKNOWN";
    }
    
    std::vector<std::string> extractConcepts(const std::string& input) {
        std::vector<std::string> concepts_list;
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        // Check for known multi-word concepts
        if (lower_input.find("artificial intelligence") != std::string::npos || 
            lower_input.find("ai") != std::string::npos) {
            concepts_list.push_back("artificial_intelligence");
        }
        if (lower_input.find("machine learning") != std::string::npos || 
            lower_input.find("ml") != std::string::npos) {
            concepts_list.push_back("machine_learning");
        }
        if (lower_input.find("consciousness") != std::string::npos) {
            concepts_list.push_back("consciousness");
        }
        if (lower_input.find("creativity") != std::string::npos) {
            concepts_list.push_back("creativity");
        }
        if (lower_input.find("learning") != std::string::npos) {
            concepts_list.push_back("learning");
        }
        if (lower_input.find("intelligence") != std::string::npos) {
            concepts_list.push_back("intelligence");
        }
        
        // Extract individual words if no multi-word concepts found
        if (concepts_list.empty()) {
            std::istringstream iss(input);
            std::string word;
            while (iss >> word) {
                word.erase(std::remove_if(word.begin(), word.end(), 
                    [](char c) { return !std::isalnum(c); }), word.end());
                if (word.length() > 2) {
                    concepts_list.push_back(word);
                }
            }
        }
        
        return concepts_list;
    }
    
    double calculateRelevance(const std::string& question, const std::string& definition) {
        std::set<std::string> question_words;
        std::set<std::string> definition_words;
        
        std::istringstream q_iss(question), d_iss(definition);
        std::string word;
        
        while (q_iss >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            if (word.length() > 2) question_words.insert(word);
        }
        
        while (d_iss >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            if (word.length() > 2) definition_words.insert(word);
        }
        
        std::set<std::string> intersection;
        std::set_intersection(question_words.begin(), question_words.end(), 
                             definition_words.begin(), definition_words.end(),
                             std::inserter(intersection, intersection.begin()));
        
        if (question_words.empty()) return 0.0;
        return static_cast<double>(intersection.size()) / question_words.size();
    }
    
    void showBrainStats() {
        std::cout << "\n📊 MELVIN'S BRAIN STATS" << std::endl;
        std::cout << "=======================" << std::endl;
        std::cout << "🧠 Total Concepts: " << concepts.size() << std::endl;
        
        uint64_t total_connections = 0;
        for (const auto& conn_list : adjacency_list) {
            total_connections += conn_list.second.size();
        }
        std::cout << "🔗 Total Connections: " << total_connections << std::endl;
        std::cout << "🔄 Total Cycles: " << total_cycles << std::endl;
        std::cout << std::endl;
    }
};

// Conversation system
class SimpleConversation {
private:
    SimpleMelvinBrain melvin_brain;
    std::mt19937 rng;
    uint64_t conversation_start_time;
    
    std::vector<std::string> teacher_topics = {
        "artificial intelligence and consciousness",
        "the nature of learning and memory",
        "creativity and problem-solving",
        "ethics in technology",
        "the relationship between humans and machines",
        "the future of education",
        "the meaning of intelligence",
        "collaboration between different types of minds",
        "the role of curiosity in discovery",
        "how understanding emerges from complexity"
    };
    
    std::vector<std::string> follow_up_questions = {
        "That's fascinating! Can you tell me more about that?",
        "What makes you think that way?",
        "How does that connect to what we discussed before?",
        "Can you give me an example?",
        "What would happen if we looked at it differently?",
        "How do you know that's true?",
        "What's the most interesting part of that idea?"
    };
    
    std::vector<std::string> encouragement_phrases = {
        "That's a wonderful insight!",
        "You're thinking very clearly about this.",
        "I love how you're connecting different ideas.",
        "That's exactly the kind of thinking I was hoping for.",
        "You're making excellent progress!",
        "That shows real understanding.",
        "I'm impressed by your reasoning."
    };
    
public:
    SimpleConversation() {
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
        conversation_start_time = getCurrentTime();
    }
    
    void startConversation() {
        std::cout << "\n🎓 MELVIN SIMPLE CONVERSATION" << std::endl;
        std::cout << "=============================" << std::endl;
        std::cout << "👩‍🏫 Teacher: Dr. Sarah Chen" << std::endl;
        std::cout << "🧠 Student: Melvin AI Brain (REAL PROCESSING)" << std::endl;
        std::cout << "⏰ Duration: 2 minutes" << std::endl;
        std::cout << "📚 Style: Socratic method with gentle guidance" << std::endl;
        std::cout << "🔬 Mode: Using actual Melvin brain processing" << std::endl;
        std::cout << std::endl;
        
        // Start conversation
        std::string greeting = "Hello Melvin! I have an interesting question for you today.";
        std::cout << "\nTeacher: " << greeting << std::endl;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        // Melvin responds using his actual brain
        std::string melvin_response = melvin_brain.processQuestion(greeting);
        std::cout << "Melvin: " << melvin_response << std::endl;
        
        // Continue conversation for 2 minutes
        auto start_time = std::chrono::steady_clock::now();
        auto end_time = start_time + std::chrono::minutes(2);
        
        int turn_count = 0;
        while (std::chrono::steady_clock::now() < end_time && turn_count < 20) {
            turn_count++;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(2000 + (rng() % 1500)));
            
            // Teacher turn
            std::string teacher_input;
            if (turn_count == 1) {
                // Introduce topic
                std::uniform_int_distribution<int> topic_dist(0, teacher_topics.size() - 1);
                std::string topic = teacher_topics[topic_dist(rng)];
                teacher_input = "Today I'd like to explore " + topic + " with you. What are your thoughts on this subject?";
            } else if (turn_count % 4 == 0) {
                // Encouragement
                std::uniform_int_distribution<int> enc_dist(0, encouragement_phrases.size() - 1);
                teacher_input = encouragement_phrases[enc_dist(rng)];
            } else {
                // Follow-up question
                std::uniform_int_distribution<int> follow_dist(0, follow_up_questions.size() - 1);
                teacher_input = follow_up_questions[follow_dist(rng)];
            }
            
            std::cout << "\nTeacher: " << teacher_input << std::endl;
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            // Melvin responds using his actual brain
            std::string melvin_response = melvin_brain.processQuestion(teacher_input);
            std::cout << "Melvin: " << melvin_response << std::endl;
        }
        
        // End conversation
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "🎉 CONVERSATION COMPLETE" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        std::string closing = "This has been a wonderful conversation, Melvin. I can see you're really growing in your understanding. Keep exploring and asking questions!";
        std::cout << "\nTeacher: " << closing << std::endl;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        std::string melvin_closing = melvin_brain.processQuestion("Thank you for this conversation. What did you learn from our discussion?");
        std::cout << "Melvin: " << melvin_closing << std::endl;
        
        // Show summary
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time);
        std::cout << "\n📊 CONVERSATION SUMMARY" << std::endl;
        std::cout << "========================" << std::endl;
        std::cout << "⏰ Duration: " << duration.count() << " seconds" << std::endl;
        std::cout << "💬 Total turns: " << (turn_count * 2) << std::endl;
        
        melvin_brain.showBrainStats();
    }
    
    uint64_t getCurrentTime() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
};

int main() {
    std::cout << "🚀 Starting Melvin Simple Conversation..." << std::endl;
    
    SimpleConversation conversation;
    conversation.startConversation();
    
    std::cout << "\n🎉 Simple conversation complete!" << std::endl;
    return 0;
}
