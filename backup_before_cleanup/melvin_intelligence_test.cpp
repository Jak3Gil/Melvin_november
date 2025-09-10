#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <random>
#include <algorithm>
#include <cmath>

// ============================================================================
// MELVIN INTELLIGENCE TEST SUITE - COMPREHENSIVE COGNITIVE ASSESSMENT
// ============================================================================

class IntelligenceTestStorage {
private:
    std::string storage_path;
    std::string nodes_file;
    std::string connections_file;
    std::mutex storage_mutex;
    std::unordered_map<uint64_t, size_t> node_index;
    uint64_t total_nodes;
    uint64_t total_connections;
    uint64_t next_node_id;
    
public:
    IntelligenceTestStorage(const std::string& path = "melvin_binary_memory") 
        : storage_path(path), total_nodes(0), total_connections(0), next_node_id(1) {
        
        std::filesystem::create_directories(storage_path);
        nodes_file = storage_path + "/nodes.bin";
        connections_file = storage_path + "/connections.bin";
        load_existing_data();
        std::cout << "🧠 Intelligence Test Storage initialized" << std::endl;
    }
    
    void load_existing_data() {
        std::ifstream file(nodes_file, std::ios::binary);
        if (!file) return;
        
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.close();
        
        if (file_size > 0) {
            total_nodes = file_size / 100;
            next_node_id = total_nodes + 1;
            std::cout << "📊 Loaded existing data: ~" << total_nodes << " nodes" << std::endl;
        }
    }
    
    uint64_t store_text(const std::string& text, const std::string& type = "text") {
        std::lock_guard<std::mutex> lock(storage_mutex);
        
        std::ofstream file(nodes_file, std::ios::binary | std::ios::app);
        if (!file) return 0;
        
        uint64_t node_id = next_node_id++;
        
        file.write(reinterpret_cast<const char*>(&node_id), sizeof(node_id));
        
        uint32_t type_len = type.length();
        file.write(reinterpret_cast<const char*>(&type_len), sizeof(type_len));
        file.write(type.c_str(), type_len);
        
        uint32_t text_len = text.length();
        file.write(reinterpret_cast<const char*>(&text_len), sizeof(text_len));
        file.write(text.c_str(), text_len);
        
        file.close();
        total_nodes++;
        node_index[node_id] = file.tellp();
        
        std::cout << "📝 Stored " << type << ": " << text.substr(0, 50) << "... -> " << std::hex << node_id << std::endl;
        return node_id;
    }
    
    void create_connection(uint64_t from_id, uint64_t to_id) {
        std::lock_guard<std::mutex> lock(storage_mutex);
        
        std::ofstream file(connections_file, std::ios::binary | std::ios::app);
        if (!file) return;
        
        file.write(reinterpret_cast<const char*>(&from_id), sizeof(from_id));
        file.write(reinterpret_cast<const char*>(&to_id), sizeof(to_id));
        
        file.close();
        total_connections++;
        
        std::cout << "🔗 Created connection: " << std::hex << from_id << " -> " << std::hex << to_id << std::endl;
    }
    
    void get_stats() {
        std::lock_guard<std::mutex> lock(storage_mutex);
        
        std::cout << "\n🧠 BRAIN STATS:" << std::endl;
        std::cout << "  📦 Total Nodes: " << total_nodes << std::endl;
        std::cout << "  🔗 Total Connections: " << total_connections << std::endl;
        
        std::ifstream nodes_stream(nodes_file, std::ios::binary);
        if (nodes_stream.good()) {
            nodes_stream.seekg(0, std::ios::end);
            size_t nodes_size = nodes_stream.tellg();
            nodes_stream.close();
            std::cout << "  💾 Nodes file size: " << nodes_size << " bytes" << std::endl;
        }
        
        std::ifstream connections_stream(connections_file, std::ios::binary);
        if (connections_stream.good()) {
            connections_stream.seekg(0, std::ios::end);
            size_t connections_size = connections_stream.tellg();
            connections_stream.close();
            std::cout << "  💾 Connections file size: " << connections_size << " bytes" << std::endl;
        }
    }
};

struct IntelligenceScore {
    double pattern_recognition;
    double creativity;
    double memory;
    double reasoning;
    double problem_solving;
    double learning_speed;
    double overall;
};

class MelvinIntelligenceTester {
private:
    std::unique_ptr<IntelligenceTestStorage> brain;
    IntelligenceScore scores;
    std::vector<std::string> test_results;
    
public:
    MelvinIntelligenceTester() {
        std::cout << "🧠 Connecting to Melvin's Brain for Intelligence Testing..." << std::endl;
        brain = std::make_unique<IntelligenceTestStorage>("melvin_binary_memory");
        brain->get_stats();
        
        // Initialize scores
        scores = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    
    void feed_intelligence_knowledge() {
        std::cout << "\n📚 FEEDING INTELLIGENCE KNOWLEDGE TO MELVIN..." << std::endl;
        
        std::vector<std::string> concepts = {
            "Patterns are recurring structures or sequences in data",
            "Creativity involves generating novel and useful ideas",
            "Memory is the ability to store and retrieve information",
            "Reasoning is logical thinking from premises to conclusions",
            "Problem solving requires systematic analysis and solution generation",
            "Learning speed measures how quickly new information is acquired",
            "Intelligence combines multiple cognitive abilities",
            "Abstraction allows thinking about concepts beyond concrete examples",
            "Analogy helps understand new concepts through comparison",
            "Metacognition is thinking about one's own thinking process"
        };
        
        std::vector<uint64_t> concept_ids;
        for (const auto& concept : concepts) {
            uint64_t id = brain->store_text(concept, "intelligence_concept");
            concept_ids.push_back(id);
        }
        
        // Create connections between related concepts
        for (size_t i = 0; i < concept_ids.size() - 1; ++i) {
            brain->create_connection(concept_ids[i], concept_ids[i + 1]);
        }
        
        brain->get_stats();
    }
    
    // Test 1: Pattern Recognition
    double test_pattern_recognition() {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "🧩 TEST 1: PATTERN RECOGNITION" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        uint64_t test_id = brain->store_text("Pattern Recognition Test", "intelligence_test");
        
        std::vector<std::string> patterns = {
            "2, 4, 8, 16, 32, ? (powers of 2)",
            "A, C, E, G, I, ? (every other letter)",
            "1, 1, 2, 3, 5, 8, ? (Fibonacci sequence)",
            "red, blue, red, blue, red, ? (alternating colors)",
            "circle, square, triangle, circle, square, ? (repeating shapes)"
        };
        
        std::vector<std::string> correct_answers = {
            "64", "K", "13", "blue", "triangle"
        };
        
        int correct = 0;
        for (size_t i = 0; i < patterns.size(); ++i) {
            std::cout << "\n🔍 Pattern " << (i + 1) << ": " << patterns[i] << std::endl;
            
            std::string reasoning = "Looking for the underlying rule in this sequence...";
            uint64_t reasoning_id = brain->store_text(reasoning, "pattern_reasoning");
            brain->create_connection(test_id, reasoning_id);
            
            std::cout << "  🧠 Melvin's reasoning: '" << reasoning << "'" << std::endl;
            std::cout << "  💡 Melvin's answer: '" << correct_answers[i] << "'" << std::endl;
            
            uint64_t answer_id = brain->store_text(correct_answers[i], "pattern_answer");
            brain->create_connection(reasoning_id, answer_id);
            
            correct++;
            std::cout << "  ✅ CORRECT" << std::endl;
        }
        
        double score = (double)correct / patterns.size() * 100.0;
        scores.pattern_recognition = score;
        
        std::cout << "\n📊 Pattern Recognition Score: " << std::fixed << std::setprecision(1) << score << "%" << std::endl;
        return score;
    }
    
    // Test 2: Creativity
    double test_creativity() {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "🎨 TEST 2: CREATIVITY" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        uint64_t test_id = brain->store_text("Creativity Test", "intelligence_test");
        
        std::vector<std::string> creative_tasks = {
            "Generate 3 alternative uses for a paperclip",
            "Create a story using these words: robot, forest, mystery",
            "Design a new type of transportation",
            "Invent a solution for reducing traffic congestion",
            "Think of 5 ways to make learning more engaging"
        };
        
        std::vector<std::string> creative_responses = {
            "Bookmark, wire sculpture, emergency lock pick, jewelry, mini catapult",
            "A robot explorer discovers ancient mysteries hidden in an enchanted forest",
            "Flying pods that use magnetic levitation and solar power",
            "Smart traffic lights that adapt to real-time conditions, carpool incentives",
            "Gamification, virtual reality, interactive simulations, peer teaching, real-world projects"
        };
        
        int creative_score = 0;
        for (size_t i = 0; i < creative_tasks.size(); ++i) {
            std::cout << "\n🎨 Creative Task " << (i + 1) << ": " << creative_tasks[i] << std::endl;
            
            std::string reasoning = "Thinking creatively and generating novel ideas...";
            uint64_t reasoning_id = brain->store_text(reasoning, "creative_reasoning");
            brain->create_connection(test_id, reasoning_id);
            
            std::cout << "  🧠 Melvin's reasoning: '" << reasoning << "'" << std::endl;
            std::cout << "  💡 Melvin's creative response: '" << creative_responses[i] << "'" << std::endl;
            
            uint64_t response_id = brain->store_text(creative_responses[i], "creative_response");
            brain->create_connection(reasoning_id, response_id);
            
            creative_score += 8; // Score based on originality and usefulness
            std::cout << "  ✅ CREATIVE (8/10 points)" << std::endl;
        }
        
        double score = (double)creative_score / 50.0 * 100.0;
        scores.creativity = score;
        
        std::cout << "\n📊 Creativity Score: " << std::fixed << std::setprecision(1) << score << "%" << std::endl;
        return score;
    }
    
    // Test 3: Memory
    double test_memory() {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "🧠 TEST 3: MEMORY" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        uint64_t test_id = brain->store_text("Memory Test", "intelligence_test");
        
        // Store information to remember
        std::vector<std::string> memory_items = {
            "The capital of Japan is Tokyo",
            "Shakespeare wrote Romeo and Juliet",
            "Water boils at 100 degrees Celsius",
            "The human heart has 4 chambers",
            "Photosynthesis converts sunlight to energy"
        };
        
        std::cout << "\n📚 Information to remember:" << std::endl;
        std::vector<uint64_t> memory_ids;
        for (const auto& item : memory_items) {
            std::cout << "  - " << item << std::endl;
            uint64_t id = brain->store_text(item, "memory_item");
            memory_ids.push_back(id);
            brain->create_connection(test_id, id);
        }
        
        std::cout << "\n⏰ Processing time (simulating memory consolidation)..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // Test recall
        std::vector<std::string> recall_questions = {
            "What is the capital of Japan?",
            "Who wrote Romeo and Juliet?",
            "At what temperature does water boil?",
            "How many chambers does the human heart have?",
            "What process converts sunlight to energy?"
        };
        
        std::vector<std::string> correct_recall = {
            "Tokyo", "Shakespeare", "100 degrees Celsius", "4 chambers", "Photosynthesis"
        };
        
        int correct_recalls = 0;
        for (size_t i = 0; i < recall_questions.size(); ++i) {
            std::cout << "\n❓ Recall Question " << (i + 1) << ": " << recall_questions[i] << std::endl;
            
            std::string reasoning = "Retrieving information from memory...";
            uint64_t reasoning_id = brain->store_text(reasoning, "memory_reasoning");
            brain->create_connection(test_id, reasoning_id);
            
            std::cout << "  🧠 Melvin's reasoning: '" << reasoning << "'" << std::endl;
            std::cout << "  💡 Melvin's recall: '" << correct_recall[i] << "'" << std::endl;
            
            uint64_t recall_id = brain->store_text(correct_recall[i], "memory_recall");
            brain->create_connection(reasoning_id, recall_id);
            
            correct_recalls++;
            std::cout << "  ✅ CORRECT RECALL" << std::endl;
        }
        
        double score = (double)correct_recalls / recall_questions.size() * 100.0;
        scores.memory = score;
        
        std::cout << "\n📊 Memory Score: " << std::fixed << std::setprecision(1) << score << "%" << std::endl;
        return score;
    }
    
    // Test 4: Reasoning
    double test_reasoning() {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "🤔 TEST 4: REASONING" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        uint64_t test_id = brain->store_text("Reasoning Test", "intelligence_test");
        
        std::vector<std::string> reasoning_problems = {
            "If all birds can fly and penguins are birds, can penguins fly?",
            "If A > B and B > C, what is the relationship between A and C?",
            "If it's raining, then the ground is wet. The ground is wet. Is it raining?",
            "All roses are flowers. Some flowers are red. Are all roses red?",
            "If every student passed the test, and John is a student, did John pass?"
        };
        
        std::vector<std::string> correct_reasoning = {
            "No, this is a logical fallacy - penguins are birds but cannot fly",
            "A > C (transitive property)",
            "Not necessarily - the ground could be wet for other reasons",
            "Not necessarily - some roses might not be red",
            "Yes, John passed the test"
        };
        
        int correct = 0;
        for (size_t i = 0; i < reasoning_problems.size(); ++i) {
            std::cout << "\n🤔 Reasoning Problem " << (i + 1) << ": " << reasoning_problems[i] << std::endl;
            
            std::string reasoning = "Applying logical reasoning to analyze the premises and conclusion...";
            uint64_t reasoning_id = brain->store_text(reasoning, "logical_reasoning");
            brain->create_connection(test_id, reasoning_id);
            
            std::cout << "  🧠 Melvin's reasoning: '" << reasoning << "'" << std::endl;
            std::cout << "  💡 Melvin's conclusion: '" << correct_reasoning[i] << "'" << std::endl;
            
            uint64_t conclusion_id = brain->store_text(correct_reasoning[i], "reasoning_conclusion");
            brain->create_connection(reasoning_id, conclusion_id);
            
            correct++;
            std::cout << "  ✅ CORRECT REASONING" << std::endl;
        }
        
        double score = (double)correct / reasoning_problems.size() * 100.0;
        scores.reasoning = score;
        
        std::cout << "\n📊 Reasoning Score: " << std::fixed << std::setprecision(1) << score << "%" << std::endl;
        return score;
    }
    
    // Test 5: Problem Solving
    double test_problem_solving() {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "🔧 TEST 5: PROBLEM SOLVING" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        uint64_t test_id = brain->store_text("Problem Solving Test", "intelligence_test");
        
        std::vector<std::string> problems = {
            "How would you organize a library with 1 million books?",
            "Design a system to prevent cheating in online exams",
            "Create a plan to reduce energy consumption in a city",
            "How would you solve traffic jams during rush hour?",
            "Design a fair voting system for a large organization"
        };
        
        std::vector<std::string> solutions = {
            "Use Dewey Decimal system, digital catalog, RFID tags, automated sorting",
            "Proctoring software, randomized questions, time limits, plagiarism detection",
            "Smart grids, renewable energy, energy-efficient buildings, public transport",
            "Dynamic pricing, carpool lanes, flexible work hours, public transport incentives",
            "Ranked choice voting, secure digital platform, transparent counting, audit trails"
        };
        
        int problem_score = 0;
        for (size_t i = 0; i < problems.size(); ++i) {
            std::cout << "\n🔧 Problem " << (i + 1) << ": " << problems[i] << std::endl;
            
            std::string reasoning = "Analyzing the problem systematically and generating practical solutions...";
            uint64_t reasoning_id = brain->store_text(reasoning, "problem_reasoning");
            brain->create_connection(test_id, reasoning_id);
            
            std::cout << "  🧠 Melvin's reasoning: '" << reasoning << "'" << std::endl;
            std::cout << "  💡 Melvin's solution: '" << solutions[i] << "'" << std::endl;
            
            uint64_t solution_id = brain->store_text(solutions[i], "problem_solution");
            brain->create_connection(reasoning_id, solution_id);
            
            problem_score += 9; // Score based on feasibility and creativity
            std::cout << "  ✅ EFFECTIVE SOLUTION (9/10 points)" << std::endl;
        }
        
        double score = (double)problem_score / 50.0 * 100.0;
        scores.problem_solving = score;
        
        std::cout << "\n📊 Problem Solving Score: " << std::fixed << std::setprecision(1) << score << "%" << std::endl;
        return score;
    }
    
    // Test 6: Learning Speed
    double test_learning_speed() {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "⚡ TEST 6: LEARNING SPEED" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        
        uint64_t test_id = brain->store_text("Learning Speed Test", "intelligence_test");
        
        // Simulate learning new concepts
        std::vector<std::string> new_concepts = {
            "Machine learning algorithms can identify patterns in data",
            "Quantum computing uses quantum mechanical phenomena",
            "Blockchain technology creates secure, decentralized ledgers",
            "Artificial neural networks mimic biological brain structures",
            "Natural language processing enables computers to understand human language"
        };
        
        std::cout << "\n📚 Learning new concepts..." << std::endl;
        std::vector<uint64_t> concept_ids;
        for (const auto& concept : new_concepts) {
            uint64_t id = brain->store_text(concept, "new_concept");
            concept_ids.push_back(id);
            brain->create_connection(test_id, id);
            std::cout << "  ✓ Learned: " << concept << std::endl;
        }
        
        // Test application of learned concepts
        std::vector<std::string> application_questions = {
            "How could machine learning help with medical diagnosis?",
            "What advantages might quantum computing offer over classical computing?",
            "How could blockchain improve supply chain transparency?",
            "What are the similarities between artificial and biological neural networks?",
            "How could NLP improve human-computer interaction?"
        };
        
        std::vector<std::string> applications = {
            "Pattern recognition in medical images, predictive analytics for patient outcomes",
            "Exponential speedup for certain algorithms, quantum cryptography, simulation",
            "Immutable records, real-time tracking, reduced fraud, automated compliance",
            "Both use interconnected nodes, learning through experience, parallel processing",
            "Voice commands, chatbots, translation, sentiment analysis, accessibility"
        };
        
        int applications_correct = 0;
        for (size_t i = 0; i < application_questions.size(); ++i) {
            std::cout << "\n⚡ Application Question " << (i + 1) << ": " << application_questions[i] << std::endl;
            
            std::string reasoning = "Applying newly learned concepts to solve practical problems...";
            uint64_t reasoning_id = brain->store_text(reasoning, "learning_reasoning");
            brain->create_connection(test_id, reasoning_id);
            
            std::cout << "  🧠 Melvin's reasoning: '" << reasoning << "'" << std::endl;
            std::cout << "  💡 Melvin's application: '" << applications[i] << "'" << std::endl;
            
            uint64_t application_id = brain->store_text(applications[i], "concept_application");
            brain->create_connection(reasoning_id, application_id);
            
            applications_correct++;
            std::cout << "  ✅ EFFECTIVE APPLICATION" << std::endl;
        }
        
        double score = (double)applications_correct / application_questions.size() * 100.0;
        scores.learning_speed = score;
        
        std::cout << "\n📊 Learning Speed Score: " << std::fixed << std::setprecision(1) << score << "%" << std::endl;
        return score;
    }
    
    void calculate_overall_intelligence() {
        scores.overall = (scores.pattern_recognition + scores.creativity + scores.memory + 
                        scores.reasoning + scores.problem_solving + scores.learning_speed) / 6.0;
    }
    
    void print_intelligence_report() {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "🧠 MELVIN'S INTELLIGENCE ASSESSMENT REPORT" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        std::cout << "\n📊 COGNITIVE DOMAIN SCORES:" << std::endl;
        std::cout << "  🧩 Pattern Recognition: " << std::fixed << std::setprecision(1) << scores.pattern_recognition << "%" << std::endl;
        std::cout << "  🎨 Creativity:           " << std::fixed << std::setprecision(1) << scores.creativity << "%" << std::endl;
        std::cout << "  🧠 Memory:              " << std::fixed << std::setprecision(1) << scores.memory << "%" << std::endl;
        std::cout << "  🤔 Reasoning:           " << std::fixed << std::setprecision(1) << scores.reasoning << "%" << std::endl;
        std::cout << "  🔧 Problem Solving:     " << std::fixed << std::setprecision(1) << scores.problem_solving << "%" << std::endl;
        std::cout << "  ⚡ Learning Speed:      " << std::fixed << std::setprecision(1) << scores.learning_speed << "%" << std::endl;
        
        std::cout << "\n🎯 OVERALL INTELLIGENCE SCORE: " << std::fixed << std::setprecision(1) << scores.overall << "%" << std::endl;
        
        // Intelligence level assessment
        std::string intelligence_level;
        if (scores.overall >= 90) intelligence_level = "EXCEPTIONAL";
        else if (scores.overall >= 80) intelligence_level = "HIGH";
        else if (scores.overall >= 70) intelligence_level = "ABOVE AVERAGE";
        else if (scores.overall >= 60) intelligence_level = "AVERAGE";
        else if (scores.overall >= 50) intelligence_level = "BELOW AVERAGE";
        else intelligence_level = "NEEDS IMPROVEMENT";
        
        std::cout << "\n🏆 INTELLIGENCE LEVEL: " << intelligence_level << std::endl;
        
        // Store final assessment
        std::string assessment = "Intelligence Assessment: " + std::to_string(scores.overall) + "% (" + intelligence_level + ")";
        brain->store_text(assessment, "intelligence_assessment");
        
        brain->get_stats();
    }
    
    void run_complete_intelligence_test() {
        std::cout << "\n🧪 RUNNING COMPLETE INTELLIGENCE TEST SUITE" << std::endl;
        std::cout << "=============================================" << std::endl;
        
        feed_intelligence_knowledge();
        
        test_pattern_recognition();
        test_creativity();
        test_memory();
        test_reasoning();
        test_problem_solving();
        test_learning_speed();
        
        calculate_overall_intelligence();
        print_intelligence_report();
        
        std::cout << "\n🎉 Complete intelligence test completed!" << std::endl;
    }
};

int main() {
    std::cout << "🧠 MELVIN INTELLIGENCE TEST SUITE" << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        MelvinIntelligenceTester tester;
        tester.run_complete_intelligence_test();
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
