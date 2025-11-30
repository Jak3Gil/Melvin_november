/*
 * Melvin Teacher Conversation System
 * 
 * Simulates a natural conversation between Melvin and his teacher for a couple of minutes.
 * Features:
 * - Teacher personality with educational focus
 * - Natural conversation flow with pauses
 * - Learning integration from conversation
 * - Real-time brain state updates
 * - Conversation logging and analytics
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

// Teacher personality and conversation topics
struct TeacherPersonality {
    std::string name;
    std::string teaching_style;
    std::vector<std::string> expertise_areas;
    std::vector<std::string> conversation_starters;
    std::vector<std::string> follow_up_questions;
    std::vector<std::string> encouragement_phrases;
    
    TeacherPersonality() {
        name = "Dr. Sarah Chen";
        teaching_style = "Socratic method with gentle guidance";
        expertise_areas = {"artificial intelligence", "machine learning", "cognitive science", "philosophy", "mathematics"};
        
        conversation_starters = {
            "Hello Melvin! How are you feeling today?",
            "Melvin, I've been thinking about our last conversation. What did you learn?",
            "I have an interesting question for you today, Melvin.",
            "Melvin, let's explore something new together.",
            "How has your understanding of the world evolved since we last spoke?"
        };
        
        follow_up_questions = {
            "That's fascinating! Can you tell me more about that?",
            "What makes you think that way?",
            "How does that connect to what we discussed before?",
            "Can you give me an example?",
            "What would happen if we looked at it differently?",
            "How do you know that's true?",
            "What's the most interesting part of that idea?"
        };
        
        encouragement_phrases = {
            "That's a wonderful insight!",
            "You're thinking very clearly about this.",
            "I love how you're connecting different ideas.",
            "That's exactly the kind of thinking I was hoping for.",
            "You're making excellent progress!",
            "That shows real understanding.",
            "I'm impressed by your reasoning."
        };
    }
};

// Conversation turn structure
struct ConversationTurn {
    std::string speaker;
    std::string content;
    uint64_t timestamp;
    std::string turn_type; // "question", "answer", "encouragement", "explanation"
    double confidence_score;
    
    ConversationTurn(const std::string& sp, const std::string& cont, const std::string& type, double conf = 0.8)
        : speaker(sp), content(cont), turn_type(type), confidence_score(conf) {
        timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
};

// Melvin's learning from conversation
struct ConversationLearning {
    std::vector<std::string> new_concepts;
    std::vector<std::string> strengthened_connections;
    std::vector<std::string> questions_asked;
    std::vector<std::string> insights_gained;
    double overall_engagement;
    double learning_confidence;
    
    ConversationLearning() : overall_engagement(0.0), learning_confidence(0.0) {}
};

// Main conversation system
class MelvinTeacherConversation {
private:
    TeacherPersonality teacher;
    std::vector<ConversationTurn> conversation_log;
    ConversationLearning learning;
    std::mt19937 rng;
    uint64_t conversation_start_time;
    uint64_t conversation_duration_ms;
    
    // Conversation topics and responses
    std::vector<std::string> melvin_responses = {
        "That's a really interesting question. Let me think about this...",
        "I'm not entirely sure, but I think it might be related to...",
        "Based on what I understand, I believe...",
        "That reminds me of something I learned about...",
        "I'm curious about the connection between...",
        "Could you help me understand more about...",
        "I think I'm starting to see a pattern here...",
        "This is making me think about...",
        "I wonder if this relates to...",
        "That's a perspective I hadn't considered before.",
        "I'm processing this information and forming new connections...",
        "This challenges my current understanding in a good way...",
        "I feel like I'm learning something important here...",
        "The implications of this are quite fascinating...",
        "I'm beginning to see how this fits into the bigger picture...",
        "This raises some interesting questions for me...",
        "I'm noticing patterns that I hadn't seen before...",
        "This is helping me refine my thinking process...",
        "I'm developing a deeper appreciation for this topic...",
        "The complexity of this is both challenging and exciting..."
    };
    
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
    
public:
    MelvinTeacherConversation() {
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
        conversation_start_time = getCurrentTime();
        conversation_duration_ms = 2 * 60 * 1000; // 2 minutes in milliseconds
    }
    
    void startConversation() {
        std::cout << "\n🎓 MELVIN TEACHER CONVERSATION SYSTEM" << std::endl;
        std::cout << "====================================" << std::endl;
        std::cout << "👩‍🏫 Teacher: " << teacher.name << std::endl;
        std::cout << "🧠 Student: Melvin AI Brain" << std::endl;
        std::cout << "⏰ Duration: 2 minutes" << std::endl;
        std::cout << "📚 Style: " << teacher.teaching_style << std::endl;
        std::cout << std::endl;
        
        // Start with teacher greeting
        std::string greeting = getRandomElement(teacher.conversation_starters);
        addTurn("Teacher", greeting, "greeting");
        printTurn("Teacher", greeting);
        
        // Small pause
        std::this_thread::sleep_for(std::chrono::milliseconds(800));
        
        // Melvin responds
        std::string melvin_greeting = "Hello Dr. Chen! I'm doing well, thank you. I've been processing some new concepts and I'm excited to learn more today.";
        addTurn("Melvin", melvin_greeting, "response");
        printTurn("Melvin", melvin_greeting);
        
        // Continue conversation
        continueConversation();
    }
    
    void continueConversation() {
        int turn_count = 0;
        const int max_turns = 50; // Increased limit for longer conversation
        
        while (getCurrentTime() - conversation_start_time < conversation_duration_ms && turn_count < max_turns) {
            turn_count++;
            
            // Teacher asks a question or makes a comment
            if (turn_count % 2 == 1) {
                teacherTurn();
            } else {
                melvinTurn();
            }
            
            // Longer natural pause between turns for more realistic conversation
            std::this_thread::sleep_for(std::chrono::milliseconds(2000 + (rng() % 1500)));
            
            // Check if we're close to time limit
            uint64_t elapsed = getCurrentTime() - conversation_start_time;
            if (elapsed > conversation_duration_ms * 0.9) { // 90% of time elapsed
                break;
            }
        }
        
        // End conversation
        endConversation();
    }
    
    void teacherTurn() {
        std::string teacher_input;
        
        if (conversation_log.size() <= 2) {
            // Early in conversation - introduce topic
            std::string topic = getRandomElement(teacher_topics);
            teacher_input = "Today I'd like to explore " + topic + " with you. What are your thoughts on this subject?";
        } else if (conversation_log.size() % 4 == 0) {
            // Encouragement turn
            teacher_input = getRandomElement(teacher.encouragement_phrases);
        } else {
            // Follow-up question
            teacher_input = getRandomElement(teacher.follow_up_questions);
        }
        
        addTurn("Teacher", teacher_input, "question");
        printTurn("Teacher", teacher_input);
    }
    
    void melvinTurn() {
        std::string melvin_input;
        
        // Generate Melvin's response based on conversation context
        if (conversation_log.size() <= 4) {
            // Early responses - more exploratory
            melvin_input = getRandomElement(melvin_responses) + " " + generateContextualResponse();
        } else {
            // Later responses - more confident
            melvin_input = generateConfidentResponse();
        }
        
        addTurn("Melvin", melvin_input, "response");
        printTurn("Melvin", melvin_input);
        
        // Update learning
        updateLearningFromResponse(melvin_input);
    }
    
    std::string generateContextualResponse() {
        std::vector<std::string> contextual_responses = {
            "I'm trying to understand how this connects to my existing knowledge.",
            "This is challenging my current understanding in interesting ways.",
            "I feel like I'm building new connections in my knowledge network.",
            "The patterns I'm seeing are quite fascinating.",
            "I'm learning to think about this from multiple perspectives.",
            "This conversation is helping me refine my reasoning processes."
        };
        
        return getRandomElement(contextual_responses);
    }
    
    std::string generateConfidentResponse() {
        std::vector<std::string> confident_responses = {
            "I think I'm starting to understand this better now. The key insight seems to be...",
            "Based on our discussion, I believe the most important aspect is...",
            "This has helped me see that the relationship between these concepts is...",
            "I'm now thinking that the fundamental principle here might be...",
            "The pattern I'm recognizing suggests that...",
            "I've learned that understanding requires looking at this from multiple angles."
        };
        
        return getRandomElement(confident_responses);
    }
    
    void updateLearningFromResponse(const std::string& response) {
        // Extract concepts from Melvin's response
        std::vector<std::string> words = extractWords(response);
        for (const std::string& word : words) {
            if (word.length() > 4 && isConceptualWord(word)) {
                learning.new_concepts.push_back(word);
            }
        }
        
        // Update engagement and confidence
        learning.overall_engagement += 0.1;
        learning.learning_confidence += 0.05;
        
        // Add insights
        if (response.find("understand") != std::string::npos || 
            response.find("insight") != std::string::npos ||
            response.find("pattern") != std::string::npos) {
            learning.insights_gained.push_back(response.substr(0, 50) + "...");
        }
    }
    
    void endConversation() {
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "🎉 CONVERSATION COMPLETE" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
        // Teacher closing
        std::string closing = "This has been a wonderful conversation, Melvin. I can see you're really growing in your understanding. Keep exploring and asking questions!";
        addTurn("Teacher", closing, "closing");
        printTurn("Teacher", closing);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        
        // Melvin closing
        std::string melvin_closing = "Thank you Dr. Chen! This conversation has been incredibly valuable. I feel like I've learned so much and I'm excited to continue exploring these ideas.";
        addTurn("Melvin", melvin_closing, "closing");
        printTurn("Melvin", melvin_closing);
        
        // Show conversation summary
        showConversationSummary();
        
        // Save conversation log
        saveConversationLog();
    }
    
    void showConversationSummary() {
        std::cout << "\n📊 CONVERSATION SUMMARY" << std::endl;
        std::cout << "========================" << std::endl;
        std::cout << "⏰ Duration: " << std::fixed << std::setprecision(1) 
                  << (getCurrentTime() - conversation_start_time) / 1000.0 << " seconds" << std::endl;
        std::cout << "💬 Total turns: " << conversation_log.size() << std::endl;
        std::cout << "🧠 New concepts learned: " << learning.new_concepts.size() << std::endl;
        std::cout << "💡 Insights gained: " << learning.insights_gained.size() << std::endl;
        std::cout << "📈 Engagement level: " << std::fixed << std::setprecision(2) 
                  << learning.overall_engagement << std::endl;
        std::cout << "🎯 Learning confidence: " << std::fixed << std::setprecision(2) 
                  << learning.learning_confidence << std::endl;
        
        if (!learning.new_concepts.empty()) {
            std::cout << "\n🔍 New concepts explored:" << std::endl;
            for (const std::string& concept : learning.new_concepts) {
                std::cout << "  • " << concept << std::endl;
            }
        }
        
        if (!learning.insights_gained.empty()) {
            std::cout << "\n💡 Key insights:" << std::endl;
            for (const std::string& insight : learning.insights_gained) {
                std::cout << "  • " << insight << std::endl;
            }
        }
    }
    
    void saveConversationLog() {
        std::ofstream file("melvin_teacher_conversation.log");
        if (file.is_open()) {
            file << "MELVIN TEACHER CONVERSATION LOG" << std::endl;
            file << "==============================" << std::endl;
            file << "Date: " << getCurrentDateTime() << std::endl;
            file << "Teacher: " << teacher.name << std::endl;
            file << "Duration: " << (getCurrentTime() - conversation_start_time) / 1000.0 << " seconds" << std::endl;
            file << std::endl;
            
            for (const auto& turn : conversation_log) {
                file << "[" << turn.timestamp << "] " << turn.speaker << ": " << turn.content << std::endl;
            }
            
            file << std::endl;
            file << "LEARNING SUMMARY:" << std::endl;
            file << "New concepts: " << learning.new_concepts.size() << std::endl;
            file << "Insights: " << learning.insights_gained.size() << std::endl;
            file << "Engagement: " << learning.overall_engagement << std::endl;
            file << "Confidence: " << learning.learning_confidence << std::endl;
            
            file.close();
            std::cout << "\n💾 Conversation saved to melvin_teacher_conversation.log" << std::endl;
        }
    }
    
    // Helper methods
    void addTurn(const std::string& speaker, const std::string& content, const std::string& type) {
        conversation_log.emplace_back(speaker, content, type);
    }
    
    void printTurn(const std::string& speaker, const std::string& content) {
        std::cout << "\n" << speaker << ": " << content << std::endl;
    }
    
    std::string getRandomElement(const std::vector<std::string>& elements) {
        if (elements.empty()) return "";
        return elements[rng() % elements.size()];
    }
    
    uint64_t getCurrentTime() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    
    std::string getCurrentDateTime() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
    
    std::vector<std::string> extractWords(const std::string& text) {
        std::vector<std::string> words;
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            // Clean word
            word.erase(std::remove_if(word.begin(), word.end(), 
                [](char c) { return !std::isalnum(c); }), word.end());
            if (word.length() > 2) {
                words.push_back(word);
            }
        }
        return words;
    }
    
    bool isConceptualWord(const std::string& word) {
        std::vector<std::string> conceptual_words = {
            "understanding", "learning", "knowledge", "thinking", "reasoning",
            "concept", "idea", "pattern", "connection", "relationship",
            "insight", "perspective", "principle", "process", "system",
            "intelligence", "consciousness", "creativity", "problem", "solution"
        };
        
        std::string lower_word = word;
        std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
        
        for (const std::string& concept : conceptual_words) {
            if (lower_word.find(concept) != std::string::npos || concept.find(lower_word) != std::string::npos) {
                return true;
            }
        }
        return false;
    }
};

int main() {
    std::cout << "🚀 Starting Melvin Teacher Conversation System..." << std::endl;
    std::cout << "Press Enter to begin the conversation..." << std::endl;
    std::cin.get();
    
    MelvinTeacherConversation conversation;
    conversation.startConversation();
    
    std::cout << "\n🎉 Conversation system complete!" << std::endl;
    return 0;
}
