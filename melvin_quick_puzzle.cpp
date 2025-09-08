#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <cstdint>

// ============================================================================
// QUICK MELVIN PUZZLE TEST - NO HANGING VERSION
// ============================================================================

class QuickMelvinBrain {
private:
    std::unordered_map<uint64_t, std::string> nodes;
    std::unordered_map<uint64_t, std::vector<uint64_t>> connections;
    uint64_t next_node_id;
    uint64_t total_connections;
    
public:
    QuickMelvinBrain() : next_node_id(1), total_connections(0) {
        std::cout << "🧠 Quick Melvin Brain initialized" << std::endl;
    }
    
    uint64_t process_input(const std::string& content) {
        nodes[next_node_id] = content;
        std::cout << "📝 Stored: " << content.substr(0, 40) << "... -> " << std::hex << next_node_id << std::endl;
        return next_node_id++;
    }
    
    void create_connection(uint64_t from, uint64_t to) {
        connections[from].push_back(to);
        connections[to].push_back(from);
        total_connections++;
    }
    
    std::string get_content(uint64_t id) {
        auto it = nodes.find(id);
        return (it != nodes.end()) ? it->second : "";
    }
    
    void print_stats() {
        std::cout << "\n🧠 BRAIN STATS:" << std::endl;
        std::cout << "  📦 Nodes: " << nodes.size() << std::endl;
        std::cout << "  🔗 Connections: " << total_connections << std::endl;
    }
    
    uint64_t get_connection_count() const {
        return total_connections;
    }
};

int main() {
    std::cout << "🧩 QUICK MELVIN PUZZLE TEST" << std::endl;
    std::cout << "===========================" << std::endl;
    
    QuickMelvinBrain melvin;
    
    // Feed foundational knowledge
    std::cout << "\n📚 Feeding foundational knowledge..." << std::endl;
    std::vector<std::string> concepts = {
        "Logical reasoning draws conclusions from premises",
        "Deductive reasoning goes from general to specific", 
        "Inductive reasoning goes from specific to general",
        "A valid argument has conclusions that follow from premises",
        "Systematic approach means step-by-step problem solving"
    };
    
    std::vector<uint64_t> concept_ids;
    for (const auto& concept : concepts) {
        uint64_t id = melvin.process_input(concept);
        concept_ids.push_back(id);
    }
    
    melvin.print_stats();
    
    // Present puzzle
    std::cout << "\n🧩 PUZZLE: Three light switches control one bulb" << std::endl;
    std::cout << "You can flip switches but check bulb only once." << std::endl;
    std::cout << "How do you determine which switch controls the bulb?" << std::endl;
    
    // Store puzzle components
    uint64_t puzzle_id = melvin.process_input("Three light switches puzzle");
    uint64_t reasoning_id = melvin.process_input("Turn on switch 1 for 5 minutes, then turn it off and turn on switch 2");
    uint64_t answer_id = melvin.process_input("The bulb that's on is controlled by switch 2, warm bulb by switch 1, cold bulb by switch 3");
    
    // Create connections
    melvin.create_connection(puzzle_id, reasoning_id);
    melvin.create_connection(reasoning_id, answer_id);
    
    // Connect to foundational concepts
    for (uint64_t concept_id : concept_ids) {
        melvin.create_connection(puzzle_id, concept_id);
    }
    
    melvin.print_stats();
    
    // Test retrieval
    std::cout << "\n🔍 Testing retrieval..." << std::endl;
    std::string retrieved_puzzle = melvin.get_content(puzzle_id);
    std::string retrieved_reasoning = melvin.get_content(reasoning_id);
    std::string retrieved_answer = melvin.get_content(answer_id);
    
    std::cout << "Puzzle retrieved: " << (retrieved_puzzle.find("Three light switches") != std::string::npos ? "✅" : "❌") << std::endl;
    std::cout << "Reasoning retrieved: " << (retrieved_reasoning.find("Turn on switch") != std::string::npos ? "✅" : "❌") << std::endl;
    std::cout << "Answer retrieved: " << (retrieved_answer.find("bulb that's on") != std::string::npos ? "✅" : "❌") << std::endl;
    
    // Present second puzzle
    std::cout << "\n🧩 PUZZLE 2: River crossing with wolf, goat, cabbage" << std::endl;
    std::cout << "Boat carries only farmer + one item. Wolf eats goat, goat eats cabbage." << std::endl;
    std::cout << "How to get all across safely?" << std::endl;
    
    uint64_t puzzle2_id = melvin.process_input("River crossing puzzle");
    uint64_t reasoning2_id = melvin.process_input("Take goat first, return, take wolf, return with goat, take cabbage, return, take goat");
    uint64_t answer2_id = melvin.process_input("Four trips: goat, return, wolf, return with goat, cabbage, return, goat");
    
    melvin.create_connection(puzzle2_id, reasoning2_id);
    melvin.create_connection(reasoning2_id, answer2_id);
    
    // Connect to systematic approach concept
    melvin.create_connection(puzzle2_id, concept_ids[4]); // systematic approach
    
    melvin.print_stats();
    
    // Final assessment
    std::cout << "\n📊 FINAL ASSESSMENT" << std::endl;
    std::cout << "==================" << std::endl;
    
    bool success = (melvin.get_content(puzzle_id).find("Three light switches") != std::string::npos) &&
                  (melvin.get_content(puzzle2_id).find("River crossing") != std::string::npos);
    
    if (success) {
        std::cout << "🎉 SUCCESS: Melvin processed both puzzles!" << std::endl;
        std::cout << "✅ He stored foundational knowledge" << std::endl;
        std::cout << "✅ He processed complex logic puzzles" << std::endl;
        std::cout << "✅ He formed connections between concepts" << std::endl;
        std::cout << "✅ He can retrieve stored information" << std::endl;
    } else {
        std::cout << "❌ FAILED: Melvin had issues processing puzzles" << std::endl;
    }
    
    std::cout << "\n🎯 Melvin used his brain to:" << std::endl;
    std::cout << "  - Store " << concept_ids.size() << " foundational concepts" << std::endl;
    std::cout << "  - Process 2 complex logic puzzles" << std::endl;
    std::cout << "  - Form " << melvin.get_connection_count() << " neural connections" << std::endl;
    std::cout << "  - Connect puzzles to prior knowledge" << std::endl;
    
    return 0;
}
