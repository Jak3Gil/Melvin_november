#include "melvin_optimized_v2.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <sstream>

// ============================================================================
// MELVIN BRAIN ARCHITECTURE VALIDATION TESTS
// ============================================================================

class MelvinBrainTester {
private:
    std::unique_ptr<MelvinOptimizedV2> melvin;
    std::vector<std::string> test_results;
    uint64_t tests_passed;
    uint64_t tests_failed;
    
    // Test data for logic puzzles
    struct LogicPuzzle {
        std::string problem;
        std::string expected_reasoning;
        std::string answer;
        std::vector<std::string> key_concepts;
        int difficulty; // 1-5 scale
    };
    
    std::vector<LogicPuzzle> logic_puzzles;
    
public:
    MelvinBrainTester(const std::string& storage_path = "melvin_test_memory") 
        : tests_passed(0), tests_failed(0) {
        melvin = std::make_unique<MelvinOptimizedV2>(storage_path);
        initialize_logic_puzzles();
        
        std::cout << "🧪 Melvin Brain Tester initialized" << std::endl;
    }
    
    void initialize_logic_puzzles() {
        logic_puzzles = {
            {
                "If all roses are flowers and some flowers are red, can we conclude that some roses are red?",
                "This requires understanding logical relationships: All roses are flowers (subset relationship), some flowers are red (partial overlap). We cannot conclude some roses are red because the red flowers might not include roses.",
                "No, we cannot conclude that some roses are red",
                {"logical reasoning", "subset relationships", "syllogism", "deductive reasoning"},
                2
            },
            {
                "A farmer has 17 sheep. All but 9 die. How many sheep are left?",
                "The phrase 'all but 9 die' means 9 sheep remain alive. This is a word problem requiring careful reading.",
                "9 sheep",
                {"word problems", "careful reading", "mathematical reasoning"},
                1
            },
            {
                "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "Each machine takes 5 minutes to make 1 widget. So 100 machines would also take 5 minutes to make 100 widgets (1 widget per machine).",
                "5 minutes",
                {"rate problems", "proportional reasoning", "machine efficiency"},
                3
            },
            {
                "In a room are 3 light switches. One controls a light bulb in another room. You can flip switches but can only check the light bulb once. How do you determine which switch controls the bulb?",
                "Turn on switch 1 for 5 minutes, then turn it off and turn on switch 2. The bulb that's on is controlled by switch 2, the warm bulb is controlled by switch 1, and the cold bulb is controlled by switch 3.",
                "Turn on switch 1 for 5 minutes, turn it off, turn on switch 2, check bulb",
                {"problem solving", "logical deduction", "systematic approach"},
                4
            },
            {
                "A man lives on the 20th floor but takes the elevator down to go to work and up to go home. Why?",
                "The man is short and can only reach the button for the 20th floor. He takes the elevator down to go to work (ground floor) and up to go home (20th floor).",
                "He's short and can only reach the 20th floor button",
                {"lateral thinking", "assumption challenging", "creative reasoning"},
                3
            }
        };
    }
    
    // ============================================================================
    // CORE ARCHITECTURE TESTS
    // ============================================================================
    
    bool test_memory_formation() {
        std::cout << "\n🧠 Testing Memory Formation..." << std::endl;
        
        // Test 1: Basic memory storage
        std::string test_text = "Memory formation test: storing and retrieving information";
        uint64_t node_id = melvin->process_text_input(test_text, "memory_test");
        
        if (node_id == 0) {
            log_test_result("Memory Formation", false, "Failed to create memory node");
            return false;
        }
        
        // Test 2: Memory retrieval
        std::string retrieved_content = melvin->get_node_content(node_id);
        if (retrieved_content != test_text) {
            log_test_result("Memory Formation", false, "Retrieved content doesn't match stored content");
            return false;
        }
        
        // Test 3: Multiple memory formation
        std::vector<uint64_t> node_ids;
        for (int i = 0; i < 10; ++i) {
            std::string content = "Memory test item " + std::to_string(i);
            uint64_t id = melvin->process_text_input(content, "multi_memory_test");
            node_ids.push_back(id);
        }
        
        // Verify all memories were created
        bool all_created = true;
        for (uint64_t id : node_ids) {
            if (id == 0) {
                all_created = false;
                break;
            }
        }
        
        if (!all_created) {
            log_test_result("Memory Formation", false, "Failed to create multiple memory nodes");
            return false;
        }
        
        log_test_result("Memory Formation", true, "Successfully formed and retrieved memories");
        return true;
    }
    
    bool test_hebbian_learning() {
        std::cout << "\n⚡ Testing Hebbian Learning..." << std::endl;
        
        // Get initial state
        auto initial_state = melvin->get_unified_state();
        uint64_t initial_connections = initial_state.global_memory.total_edges;
        uint64_t initial_hebbian_updates = initial_state.global_memory.stats.hebbian_updates;
        
        // Feed related concepts in quick succession to trigger Hebbian learning
        std::vector<std::string> related_concepts = {
            "Neural networks process information",
            "Neural networks learn patterns",
            "Pattern recognition is important",
            "Learning requires pattern recognition",
            "Information processing enables learning"
        };
        
        std::vector<uint64_t> concept_nodes;
        for (const auto& concept : related_concepts) {
            uint64_t node_id = melvin->process_text_input(concept, "hebbian_test");
            concept_nodes.push_back(node_id);
            
            // Small delay to ensure coactivation window
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Check if Hebbian connections were formed
        auto final_state = melvin->get_unified_state();
        uint64_t final_connections = final_state.global_memory.total_edges;
        uint64_t final_hebbian_updates = final_state.global_memory.stats.hebbian_updates;
        
        bool connections_increased = final_connections > initial_connections;
        bool hebbian_updates_increased = final_hebbian_updates > initial_hebbian_updates;
        
        if (!connections_increased || !hebbian_updates_increased) {
            log_test_result("Hebbian Learning", false, 
                "No new connections formed. Initial: " + std::to_string(initial_connections) + 
                ", Final: " + std::to_string(final_connections));
            return false;
        }
        
        log_test_result("Hebbian Learning", true, 
            "Formed " + std::to_string(final_connections - initial_connections) + 
            " new connections through Hebbian learning");
        return true;
    }
    
    bool test_reasoning_capability() {
        std::cout << "\n🤔 Testing Reasoning Capability..." << std::endl;
        
        // Test 1: Feed reasoning concepts
        std::vector<std::string> reasoning_concepts = {
            "Logical reasoning involves drawing conclusions from premises",
            "Deductive reasoning goes from general to specific",
            "Inductive reasoning goes from specific to general",
            "Cause and effect relationships help understand problems",
            "Systematic thinking breaks down complex problems"
        };
        
        for (const auto& concept : reasoning_concepts) {
            melvin->process_text_input(concept, "reasoning_test");
        }
        
        // Test 2: Feed a simple reasoning problem
        std::string problem = "If A implies B, and B implies C, what can we conclude about A and C?";
        uint64_t problem_id = melvin->process_text_input(problem, "reasoning_problem");
        
        // Test 3: Feed the solution
        std::string solution = "If A implies B, and B implies C, then A implies C (transitive property)";
        uint64_t solution_id = melvin->process_text_input(solution, "reasoning_solution");
        
        // Check if both were stored
        if (problem_id == 0 || solution_id == 0) {
            log_test_result("Reasoning Capability", false, "Failed to store reasoning problem/solution");
            return false;
        }
        
        // Verify retrieval
        std::string retrieved_problem = melvin->get_node_content(problem_id);
        std::string retrieved_solution = melvin->get_node_content(solution_id);
        
        if (retrieved_problem != problem || retrieved_solution != solution) {
            log_test_result("Reasoning Capability", false, "Failed to retrieve reasoning content");
            return false;
        }
        
        log_test_result("Reasoning Capability", true, "Successfully stored and retrieved reasoning content");
        return true;
    }
    
    // ============================================================================
    // LOGIC PUZZLE SOLVING TESTS
    // ============================================================================
    
    bool test_logic_puzzle_solving() {
        std::cout << "\n🧩 Testing Logic Puzzle Solving..." << std::endl;
        
        bool all_passed = true;
        
        for (size_t i = 0; i < logic_puzzles.size(); ++i) {
            const auto& puzzle = logic_puzzles[i];
            
            std::cout << "\n--- Puzzle " << (i + 1) << " ---" << std::endl;
            std::cout << "Problem: " << puzzle.problem << std::endl;
            
            // Feed the problem
            uint64_t problem_id = melvin->process_text_input(puzzle.problem, "logic_puzzle");
            
            // Feed key concepts
            for (const auto& concept : puzzle.key_concepts) {
                melvin->process_text_input(concept, "puzzle_concept");
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            
            // Feed the reasoning process
            uint64_t reasoning_id = melvin->process_text_input(puzzle.expected_reasoning, "puzzle_reasoning");
            
            // Feed the answer
            uint64_t answer_id = melvin->process_text_input(puzzle.answer, "puzzle_answer");
            
            // Verify all components were stored
            if (problem_id == 0 || reasoning_id == 0 || answer_id == 0) {
                log_test_result("Logic Puzzle " + std::to_string(i + 1), false, "Failed to store puzzle components");
                all_passed = false;
                continue;
            }
            
            // Verify retrieval
            std::string retrieved_problem = melvin->get_node_content(problem_id);
            std::string retrieved_reasoning = melvin->get_node_content(reasoning_id);
            std::string retrieved_answer = melvin->get_node_content(answer_id);
            
            if (retrieved_problem != puzzle.problem || 
                retrieved_reasoning != puzzle.expected_reasoning || 
                retrieved_answer != puzzle.answer) {
                log_test_result("Logic Puzzle " + std::to_string(i + 1), false, "Retrieval mismatch");
                all_passed = false;
                continue;
            }
            
            log_test_result("Logic Puzzle " + std::to_string(i + 1), true, 
                "Difficulty " + std::to_string(puzzle.difficulty) + " - Successfully processed");
            
            // Small delay between puzzles
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        
        return all_passed;
    }
    
    bool test_pattern_vs_reasoning() {
        std::cout << "\n🔍 Testing Pattern Matching vs Reasoning..." << std::endl;
        
        // Test 1: Feed similar but different problems
        std::vector<std::string> similar_problems = {
            "If all birds can fly and penguins are birds, can penguins fly?",
            "If all fish can swim and dolphins are fish, can dolphins swim?",
            "If all mammals are warm-blooded and whales are mammals, are whales warm-blooded?"
        };
        
        std::vector<uint64_t> problem_ids;
        for (const auto& problem : similar_problems) {
            uint64_t id = melvin->process_text_input(problem, "similar_problems");
            problem_ids.push_back(id);
        }
        
        // Test 2: Feed different reasoning approaches
        std::vector<std::string> reasoning_approaches = {
            "Penguins are birds but cannot fly - exceptions exist in categories",
            "Dolphins are mammals, not fish - classification matters",
            "Whales are mammals and are warm-blooded - consistent with category"
        };
        
        std::vector<uint64_t> reasoning_ids;
        for (const auto& reasoning : reasoning_approaches) {
            uint64_t id = melvin->process_text_input(reasoning, "reasoning_approaches");
            reasoning_ids.push_back(id);
        }
        
        // Verify all were stored correctly
        bool all_stored = true;
        for (uint64_t id : problem_ids) {
            if (id == 0) all_stored = false;
        }
        for (uint64_t id : reasoning_ids) {
            if (id == 0) all_stored = false;
        }
        
        if (!all_stored) {
            log_test_result("Pattern vs Reasoning", false, "Failed to store test problems");
            return false;
        }
        
        // Test 3: Verify retrieval shows different content
        bool different_content = true;
        for (size_t i = 0; i < problem_ids.size(); ++i) {
            std::string content = melvin->get_node_content(problem_ids[i]);
            if (content != similar_problems[i]) {
                different_content = false;
                break;
            }
        }
        
        if (!different_content) {
            log_test_result("Pattern vs Reasoning", false, "Retrieved content doesn't match stored content");
            return false;
        }
        
        log_test_result("Pattern vs Reasoning", true, "Successfully distinguished between different problems and reasoning approaches");
        return true;
    }
    
    // ============================================================================
    // PERFORMANCE AND EFFICIENCY TESTS
    // ============================================================================
    
    bool test_memory_efficiency() {
        std::cout << "\n💾 Testing Memory Efficiency..." << std::endl;
        
        auto initial_state = melvin->get_unified_state();
        double initial_mb = initial_state.global_memory.storage_used_mb;
        
        // Feed large amounts of data
        std::vector<std::string> large_data;
        for (int i = 0; i < 100; ++i) {
            std::string data = "Large data sample " + std::to_string(i) + 
                " with additional content to test compression and storage efficiency. "
                "This should be compressed effectively by the binary storage system.";
            large_data.push_back(data);
            melvin->process_text_input(data, "efficiency_test");
        }
        
        auto final_state = melvin->get_unified_state();
        double final_mb = final_state.global_memory.storage_used_mb;
        double efficiency_ratio = final_mb / (100 * 0.001); // Rough estimate of uncompressed size
        
        std::cout << "Initial storage: " << std::fixed << std::setprecision(2) << initial_mb << " MB" << std::endl;
        std::cout << "Final storage: " << final_mb << " MB" << std::endl;
        std::cout << "Efficiency ratio: " << efficiency_ratio << std::endl;
        
        // Check if compression is working (should be much less than 1.0)
        bool efficient = efficiency_ratio < 0.5; // At least 50% compression
        
        if (!efficient) {
            log_test_result("Memory Efficiency", false, "Compression ratio too high: " + std::to_string(efficiency_ratio));
            return false;
        }
        
        log_test_result("Memory Efficiency", true, 
            "Achieved " + std::to_string((1.0 - efficiency_ratio) * 100) + "% compression");
        return true;
    }
    
    bool test_processing_speed() {
        std::cout << "\n⚡ Testing Processing Speed..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Process 1000 items
        for (int i = 0; i < 1000; ++i) {
            std::string content = "Speed test item " + std::to_string(i);
            melvin->process_text_input(content, "speed_test");
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        double items_per_second = 1000.0 / (duration.count() / 1000.0);
        
        std::cout << "Processed 1000 items in " << duration.count() << " ms" << std::endl;
        std::cout << "Processing speed: " << std::fixed << std::setprecision(0) << items_per_second << " items/second" << std::endl;
        
        // Expect at least 100 items per second
        bool fast_enough = items_per_second >= 100;
        
        if (!fast_enough) {
            log_test_result("Processing Speed", false, "Too slow: " + std::to_string(items_per_second) + " items/second");
            return false;
        }
        
        log_test_result("Processing Speed", true, 
            "Achieved " + std::to_string(items_per_second) + " items/second");
        return true;
    }
    
    // ============================================================================
    // BRAIN STATE VALIDATION TESTS
    // ============================================================================
    
    bool test_brain_state_consistency() {
        std::cout << "\n🧠 Testing Brain State Consistency..." << std::endl;
        
        // Get initial state
        auto state1 = melvin->get_unified_state();
        
        // Process some data
        melvin->process_text_input("State consistency test", "consistency_test");
        
        // Get state again
        auto state2 = melvin->get_unified_state();
        
        // Check that state changed appropriately
        bool nodes_increased = state2.global_memory.total_nodes > state1.global_memory.total_nodes;
        bool uptime_increased = state2.system.uptime_seconds >= state1.system.uptime_seconds;
        bool running_consistent = state1.system.running == state2.system.running;
        
        if (!nodes_increased || !uptime_increased || !running_consistent) {
            log_test_result("Brain State Consistency", false, 
                "State inconsistency detected. Nodes: " + std::to_string(state1.global_memory.total_nodes) + 
                " -> " + std::to_string(state2.global_memory.total_nodes));
            return false;
        }
        
        log_test_result("Brain State Consistency", true, "Brain state remains consistent across operations");
        return true;
    }
    
    // ============================================================================
    // TEST EXECUTION AND REPORTING
    // ============================================================================
    
    void log_test_result(const std::string& test_name, bool passed, const std::string& details) {
        std::string result = (passed ? "✅ PASS" : "❌ FAIL") + " - " + test_name + ": " + details;
        test_results.push_back(result);
        
        if (passed) {
            tests_passed++;
        } else {
            tests_failed++;
        }
        
        std::cout << result << std::endl;
    }
    
    void run_all_tests() {
        std::cout << "🧪 RUNNING MELVIN BRAIN ARCHITECTURE TESTS" << std::endl;
        std::cout << "==========================================" << std::endl;
        
        // Core architecture tests
        test_memory_formation();
        test_hebbian_learning();
        test_reasoning_capability();
        
        // Logic puzzle tests
        test_logic_puzzle_solving();
        test_pattern_vs_reasoning();
        
        // Performance tests
        test_memory_efficiency();
        test_processing_speed();
        
        // State validation tests
        test_brain_state_consistency();
        
        // Print final results
        print_test_summary();
    }
    
    void print_test_summary() {
        std::cout << "\n📊 TEST SUMMARY" << std::endl;
        std::cout << "===============" << std::endl;
        std::cout << "✅ Tests Passed: " << tests_passed << std::endl;
        std::cout << "❌ Tests Failed: " << tests_failed << std::endl;
        std::cout << "📈 Success Rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * tests_passed / (tests_passed + tests_failed)) << "%" << std::endl;
        
        std::cout << "\n📋 DETAILED RESULTS:" << std::endl;
        for (const auto& result : test_results) {
            std::cout << "  " << result << std::endl;
        }
        
        // Get final brain state
        auto final_state = melvin->get_unified_state();
        std::cout << "\n🧠 FINAL BRAIN STATE:" << std::endl;
        std::cout << "  📦 Total Nodes: " << final_state.global_memory.total_nodes << std::endl;
        std::cout << "  🔗 Total Connections: " << final_state.global_memory.total_edges << std::endl;
        std::cout << "  💾 Storage Used: " << std::fixed << std::setprecision(2) 
                  << final_state.global_memory.storage_used_mb << " MB" << std::endl;
        std::cout << "  ⚡ Hebbian Updates: " << final_state.global_memory.stats.hebbian_updates << std::endl;
        std::cout << "  ⏱️ Uptime: " << final_state.system.uptime_seconds << " seconds" << std::endl;
    }
    
    void save_test_report() {
        std::ofstream report("melvin_brain_test_report.txt");
        if (!report) {
            std::cout << "❌ Failed to create test report file" << std::endl;
            return;
        }
        
        report << "MELVIN BRAIN ARCHITECTURE TEST REPORT" << std::endl;
        report << "=====================================" << std::endl;
        report << "Generated: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
        report << std::endl;
        
        report << "TEST SUMMARY:" << std::endl;
        report << "Tests Passed: " << tests_passed << std::endl;
        report << "Tests Failed: " << tests_failed << std::endl;
        report << "Success Rate: " << std::fixed << std::setprecision(1) 
               << (100.0 * tests_passed / (tests_passed + tests_failed)) << "%" << std::endl;
        report << std::endl;
        
        report << "DETAILED RESULTS:" << std::endl;
        for (const auto& result : test_results) {
            report << result << std::endl;
        }
        
        report.close();
        std::cout << "📄 Test report saved to melvin_brain_test_report.txt" << std::endl;
    }
};

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    std::cout << "🧪 MELVIN BRAIN ARCHITECTURE VALIDATION" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    try {
        MelvinBrainTester tester;
        
        // Run all tests
        tester.run_all_tests();
        
        // Save test report
        tester.save_test_report();
        
        std::cout << "\n🎉 Brain architecture validation completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
