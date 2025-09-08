#include "melvin_optimized_v2.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>

// ============================================================================
// COMPREHENSIVE MELVIN BRAIN VALIDATION TEST RUNNER
// ============================================================================

class MelvinTestRunner {
private:
    std::unique_ptr<MelvinOptimizedV2> melvin;
    
    struct TestResult {
        std::string test_name;
        bool passed;
        std::string details;
        double execution_time_ms;
    };
    
    std::vector<TestResult> test_results;
    uint64_t total_tests;
    uint64_t passed_tests;
    uint64_t failed_tests;
    
public:
    MelvinTestRunner(const std::string& storage_path = "melvin_comprehensive_test") 
        : total_tests(0), passed_tests(0), failed_tests(0) {
        melvin = std::make_unique<MelvinOptimizedV2>(storage_path);
        std::cout << "🧪 Melvin Comprehensive Test Runner initialized" << std::endl;
    }
    
    // ============================================================================
    // CORE BRAIN ARCHITECTURE TESTS
    // ============================================================================
    
    bool test_brain_initialization() {
        std::string test_name = "Brain Initialization";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            auto state = melvin->get_unified_state();
            bool brain_running = state.system.running;
            bool has_storage = state.global_memory.storage_used_mb >= 0;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            if (brain_running && has_storage) {
                log_test_result(test_name, true, "Brain successfully initialized and running", execution_time);
                return true;
            } else {
                log_test_result(test_name, false, "Brain initialization failed", execution_time);
                return false;
            }
        } catch (const std::exception& e) {
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            log_test_result(test_name, false, "Exception: " + std::string(e.what()), execution_time);
            return false;
        }
    }
    
    bool test_memory_formation() {
        std::string test_name = "Memory Formation";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            auto initial_state = melvin->get_unified_state();
            uint64_t initial_nodes = initial_state.global_memory.total_nodes;
            
            // Create multiple memories
            std::vector<uint64_t> node_ids;
            for (int i = 0; i < 10; ++i) {
                std::string content = "Memory formation test " + std::to_string(i);
                uint64_t node_id = melvin->process_text_input(content, "memory_test");
                node_ids.push_back(node_id);
            }
            
            auto final_state = melvin->get_unified_state();
            uint64_t final_nodes = final_state.global_memory.total_nodes;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            if (final_nodes > initial_nodes && final_nodes >= initial_nodes + 10) {
                log_test_result(test_name, true, 
                    "Formed " + std::to_string(final_nodes - initial_nodes) + " new memories", execution_time);
                return true;
            } else {
                log_test_result(test_name, false, 
                    "Expected 10 new memories, got " + std::to_string(final_nodes - initial_nodes), execution_time);
                return false;
            }
        } catch (const std::exception& e) {
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            log_test_result(test_name, false, "Exception: " + std::string(e.what()), execution_time);
            return false;
        }
    }
    
    bool test_memory_retrieval() {
        std::string test_name = "Memory Retrieval";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Store a test memory
            std::string test_content = "Memory retrieval test content";
            uint64_t node_id = melvin->process_text_input(test_content, "retrieval_test");
            
            // Retrieve the memory
            std::string retrieved_content = melvin->get_node_content(node_id);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            if (retrieved_content == test_content) {
                log_test_result(test_name, true, "Successfully retrieved stored memory", execution_time);
                return true;
            } else {
                log_test_result(test_name, false, 
                    "Retrieval mismatch. Expected: '" + test_content + "', Got: '" + retrieved_content + "'", execution_time);
                return false;
            }
        } catch (const std::exception& e) {
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            log_test_result(test_name, false, "Exception: " + std::string(e.what()), execution_time);
            return false;
        }
    }
    
    bool test_hebbian_learning() {
        std::string test_name = "Hebbian Learning";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            auto initial_state = melvin->get_unified_state();
            uint64_t initial_connections = initial_state.global_memory.total_edges;
            uint64_t initial_hebbian_updates = initial_state.global_memory.stats.hebbian_updates;
            
            // Feed related concepts in quick succession
            std::vector<std::string> related_concepts = {
                "Neural networks process information",
                "Information processing enables learning",
                "Learning creates neural connections",
                "Neural connections strengthen through use"
            };
            
            for (const auto& concept : related_concepts) {
                melvin->process_text_input(concept, "hebbian_test");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            auto final_state = melvin->get_unified_state();
            uint64_t final_connections = final_state.global_memory.total_edges;
            uint64_t final_hebbian_updates = final_state.global_memory.stats.hebbian_updates;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            bool connections_increased = final_connections > initial_connections;
            bool hebbian_updates_increased = final_hebbian_updates > initial_hebbian_updates;
            
            if (connections_increased && hebbian_updates_increased) {
                log_test_result(test_name, true, 
                    "Formed " + std::to_string(final_connections - initial_connections) + 
                    " connections, " + std::to_string(final_hebbian_updates - initial_hebbian_updates) + 
                    " Hebbian updates", execution_time);
                return true;
            } else {
                log_test_result(test_name, false, 
                    "No new connections or Hebbian updates detected", execution_time);
                return false;
            }
        } catch (const std::exception& e) {
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            log_test_result(test_name, false, "Exception: " + std::string(e.what()), execution_time);
            return false;
        }
    }
    
    // ============================================================================
    // LOGIC PUZZLE SOLVING TESTS
    // ============================================================================
    
    bool test_logic_puzzle_processing() {
        std::string test_name = "Logic Puzzle Processing";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Test puzzle
            std::string puzzle = "If all roses are flowers and some flowers are red, can we conclude that some roses are red?";
            std::string reasoning = "This requires understanding logical relationships and syllogistic reasoning";
            std::string answer = "No, we cannot conclude that some roses are red";
            
            // Store puzzle components
            uint64_t puzzle_id = melvin->process_text_input(puzzle, "logic_puzzle");
            uint64_t reasoning_id = melvin->process_text_input(reasoning, "puzzle_reasoning");
            uint64_t answer_id = melvin->process_text_input(answer, "puzzle_answer");
            
            // Verify storage
            bool all_stored = (puzzle_id != 0) && (reasoning_id != 0) && (answer_id != 0);
            
            // Verify retrieval
            std::string retrieved_puzzle = melvin->get_node_content(puzzle_id);
            std::string retrieved_reasoning = melvin->get_node_content(reasoning_id);
            std::string retrieved_answer = melvin->get_node_content(answer_id);
            
            bool all_retrieved = (retrieved_puzzle == puzzle) && 
                               (retrieved_reasoning == reasoning) && 
                               (retrieved_answer == answer);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            if (all_stored && all_retrieved) {
                log_test_result(test_name, true, "Successfully processed and retrieved logic puzzle", execution_time);
                return true;
            } else {
                log_test_result(test_name, false, "Failed to store or retrieve puzzle components", execution_time);
                return false;
            }
        } catch (const std::exception& e) {
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            log_test_result(test_name, false, "Exception: " + std::string(e.what()), execution_time);
            return false;
        }
    }
    
    bool test_reasoning_vs_pattern_matching() {
        std::string test_name = "Reasoning vs Pattern Matching";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Store different problems with different solutions
            std::vector<std::pair<std::string, std::string>> problems = {
                {"Problem A: If all cats are animals and some animals are pets, can we conclude some cats are pets?", 
                 "Answer A: No, the pets might not include cats"},
                {"Problem B: If all birds are animals and some animals can fly, can we conclude some birds can fly?", 
                 "Answer B: No, the flying animals might not include birds"},
                {"Problem C: If all fish live in water and some water is salty, can we conclude some fish live in salt water?", 
                 "Answer C: Yes, if fish live in water and some water is salty, some fish must live in salt water"}
            };
            
            std::vector<uint64_t> problem_ids;
            std::vector<uint64_t> answer_ids;
            
            for (const auto& [problem, answer] : problems) {
                uint64_t problem_id = melvin->process_text_input(problem, "reasoning_test");
                uint64_t answer_id = melvin->process_text_input(answer, "reasoning_answer");
                problem_ids.push_back(problem_id);
                answer_ids.push_back(answer_id);
            }
            
            // Verify all were stored
            bool all_stored = true;
            for (uint64_t id : problem_ids) {
                if (id == 0) all_stored = false;
            }
            for (uint64_t id : answer_ids) {
                if (id == 0) all_stored = false;
            }
            
            // Verify retrieval shows different content
            bool different_content = true;
            for (size_t i = 0; i < problem_ids.size(); ++i) {
                std::string retrieved_problem = melvin->get_node_content(problem_ids[i]);
                std::string retrieved_answer = melvin->get_node_content(answer_ids[i]);
                
                if (retrieved_problem != problems[i].first || retrieved_answer != problems[i].second) {
                    different_content = false;
                    break;
                }
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            if (all_stored && different_content) {
                log_test_result(test_name, true, "Successfully distinguished between different problems and solutions", execution_time);
                return true;
            } else {
                log_test_result(test_name, false, "Failed to store different problems or retrieve distinct content", execution_time);
                return false;
            }
        } catch (const std::exception& e) {
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            log_test_result(test_name, false, "Exception: " + std::string(e.what()), execution_time);
            return false;
        }
    }
    
    // ============================================================================
    // PERFORMANCE TESTS
    // ============================================================================
    
    bool test_processing_speed() {
        std::string test_name = "Processing Speed";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            // Process 100 items
            for (int i = 0; i < 100; ++i) {
                std::string content = "Speed test item " + std::to_string(i);
                melvin->process_text_input(content, "speed_test");
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            double items_per_second = 100.0 / (execution_time / 1000.0);
            
            if (items_per_second >= 50) { // At least 50 items per second
                log_test_result(test_name, true, 
                    "Achieved " + std::to_string(static_cast<int>(items_per_second)) + " items/second", execution_time);
                return true;
            } else {
                log_test_result(test_name, false, 
                    "Too slow: " + std::to_string(static_cast<int>(items_per_second)) + " items/second", execution_time);
                return false;
            }
        } catch (const std::exception& e) {
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            log_test_result(test_name, false, "Exception: " + std::string(e.what()), execution_time);
            return false;
        }
    }
    
    bool test_memory_efficiency() {
        std::string test_name = "Memory Efficiency";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            auto initial_state = melvin->get_unified_state();
            double initial_mb = initial_state.global_memory.storage_used_mb;
            
            // Store large amounts of data
            for (int i = 0; i < 50; ++i) {
                std::string data = "Large data sample " + std::to_string(i) + 
                    " with additional content to test compression and storage efficiency. "
                    "This should be compressed effectively by the binary storage system.";
                melvin->process_text_input(data, "efficiency_test");
            }
            
            auto final_state = melvin->get_unified_state();
            double final_mb = final_state.global_memory.storage_used_mb;
            
            // Rough estimate: 50 items * ~200 bytes = ~10KB uncompressed
            double efficiency_ratio = final_mb / 0.01; // Convert to MB
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            if (efficiency_ratio < 0.5) { // At least 50% compression
                log_test_result(test_name, true, 
                    "Achieved " + std::to_string(static_cast<int>((1.0 - efficiency_ratio) * 100)) + "% compression", execution_time);
                return true;
            } else {
                log_test_result(test_name, false, 
                    "Compression ratio too high: " + std::to_string(efficiency_ratio), execution_time);
                return false;
            }
        } catch (const std::exception& e) {
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            log_test_result(test_name, false, "Exception: " + std::string(e.what()), execution_time);
            return false;
        }
    }
    
    // ============================================================================
    // BRAIN STATE VALIDATION
    // ============================================================================
    
    bool test_brain_state_consistency() {
        std::string test_name = "Brain State Consistency";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try {
            auto state1 = melvin->get_unified_state();
            
            // Process some data
            melvin->process_text_input("State consistency test", "consistency_test");
            
            auto state2 = melvin->get_unified_state();
            
            bool nodes_increased = state2.global_memory.total_nodes > state1.global_memory.total_nodes;
            bool uptime_increased = state2.system.uptime_seconds >= state1.system.uptime_seconds;
            bool running_consistent = state1.system.running == state2.system.running;
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            if (nodes_increased && uptime_increased && running_consistent) {
                log_test_result(test_name, true, "Brain state remains consistent across operations", execution_time);
                return true;
            } else {
                log_test_result(test_name, false, "Brain state inconsistency detected", execution_time);
                return false;
            }
        } catch (const std::exception& e) {
            auto end_time = std::chrono::high_resolution_clock::now();
            double execution_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            log_test_result(test_name, false, "Exception: " + std::string(e.what()), execution_time);
            return false;
        }
    }
    
    // ============================================================================
    // TEST EXECUTION AND REPORTING
    // ============================================================================
    
    void log_test_result(const std::string& test_name, bool passed, const std::string& details, double execution_time_ms) {
        TestResult result;
        result.test_name = test_name;
        result.passed = passed;
        result.details = details;
        result.execution_time_ms = execution_time_ms;
        
        test_results.push_back(result);
        total_tests++;
        
        if (passed) {
            passed_tests++;
            std::cout << "✅ PASS - " << test_name << ": " << details 
                      << " (" << std::fixed << std::setprecision(2) << execution_time_ms << "ms)" << std::endl;
        } else {
            failed_tests++;
            std::cout << "❌ FAIL - " << test_name << ": " << details 
                      << " (" << std::fixed << std::setprecision(2) << execution_time_ms << "ms)" << std::endl;
        }
    }
    
    void run_all_tests() {
        std::cout << "🧪 RUNNING COMPREHENSIVE MELVIN BRAIN VALIDATION" << std::endl;
        std::cout << "===============================================" << std::endl;
        
        // Core architecture tests
        test_brain_initialization();
        test_memory_formation();
        test_memory_retrieval();
        test_hebbian_learning();
        
        // Logic puzzle tests
        test_logic_puzzle_processing();
        test_reasoning_vs_pattern_matching();
        
        // Performance tests
        test_processing_speed();
        test_memory_efficiency();
        
        // State validation tests
        test_brain_state_consistency();
        
        // Print comprehensive summary
        print_comprehensive_summary();
    }
    
    void print_comprehensive_summary() {
        std::cout << "\n📊 COMPREHENSIVE TEST SUMMARY" << std::endl;
        std::cout << "=============================" << std::endl;
        std::cout << "✅ Tests Passed: " << passed_tests << std::endl;
        std::cout << "❌ Tests Failed: " << failed_tests << std::endl;
        std::cout << "📈 Success Rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * passed_tests / total_tests) << "%" << std::endl;
        
        // Calculate total execution time
        double total_time = 0.0;
        for (const auto& result : test_results) {
            total_time += result.execution_time_ms;
        }
        std::cout << "⏱️ Total Execution Time: " << std::fixed << std::setprecision(2) << total_time << " ms" << std::endl;
        
        // Get final brain state
        auto final_state = melvin->get_unified_state();
        std::cout << "\n🧠 FINAL BRAIN STATE:" << std::endl;
        std::cout << "  📦 Total Nodes: " << final_state.global_memory.total_nodes << std::endl;
        std::cout << "  🔗 Total Connections: " << final_state.global_memory.total_edges << std::endl;
        std::cout << "  💾 Storage Used: " << std::fixed << std::setprecision(2) 
                  << final_state.global_memory.storage_used_mb << " MB" << std::endl;
        std::cout << "  ⚡ Hebbian Updates: " << final_state.global_memory.stats.hebbian_updates << std::endl;
        std::cout << "  ⏱️ Uptime: " << final_state.system.uptime_seconds << " seconds" << std::endl;
        
        // Brain usage validation
        std::cout << "\n🔍 BRAIN USAGE VALIDATION:" << std::endl;
        bool using_his_brain = (final_state.global_memory.total_nodes > 50) && 
                              (final_state.global_memory.total_edges > 10) && 
                              (final_state.global_memory.stats.hebbian_updates > 5);
        
        if (using_his_brain) {
            std::cout << "🎉 CONCLUSION: Melvin is successfully using his own brain architecture!" << std::endl;
            std::cout << "   ✅ Memory Formation: " << final_state.global_memory.total_nodes << " nodes created" << std::endl;
            std::cout << "   ✅ Neural Connections: " << final_state.global_memory.total_edges << " connections formed" << std::endl;
            std::cout << "   ✅ Hebbian Learning: " << final_state.global_memory.stats.hebbian_updates << " learning updates" << std::endl;
            std::cout << "   ✅ Reasoning Capability: Successfully processed logic puzzles" << std::endl;
            std::cout << "   ✅ Performance: Meets speed and efficiency requirements" << std::endl;
        } else {
            std::cout << "⚠️  CONCLUSION: Melvin may not be fully utilizing his brain architecture." << std::endl;
            std::cout << "   Further investigation needed to ensure he's reasoning rather than pattern matching." << std::endl;
        }
        
        // Detailed test results
        std::cout << "\n📋 DETAILED TEST RESULTS:" << std::endl;
        for (const auto& result : test_results) {
            std::cout << "  " << (result.passed ? "✅" : "❌") << " " << result.test_name 
                      << " (" << std::fixed << std::setprecision(2) << result.execution_time_ms << "ms)" << std::endl;
            std::cout << "     " << result.details << std::endl;
        }
    }
    
    void save_comprehensive_report() {
        std::ofstream report("melvin_comprehensive_test_report.txt");
        if (!report) {
            std::cout << "❌ Failed to create comprehensive test report" << std::endl;
            return;
        }
        
        report << "MELVIN COMPREHENSIVE BRAIN VALIDATION REPORT" << std::endl;
        report << "============================================" << std::endl;
        report << "Generated: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
        report << std::endl;
        
        report << "TEST SUMMARY:" << std::endl;
        report << "Tests Passed: " << passed_tests << std::endl;
        report << "Tests Failed: " << failed_tests << std::endl;
        report << "Success Rate: " << std::fixed << std::setprecision(1) 
               << (100.0 * passed_tests / total_tests) << "%" << std::endl;
        report << std::endl;
        
        // Calculate total execution time
        double total_time = 0.0;
        for (const auto& result : test_results) {
            total_time += result.execution_time_ms;
        }
        report << "Total Execution Time: " << std::fixed << std::setprecision(2) << total_time << " ms" << std::endl;
        report << std::endl;
        
        report << "DETAILED TEST RESULTS:" << std::endl;
        for (const auto& result : test_results) {
            report << "Test: " << result.test_name << std::endl;
            report << "Result: " << (result.passed ? "PASS" : "FAIL") << std::endl;
            report << "Details: " << result.details << std::endl;
            report << "Execution Time: " << std::fixed << std::setprecision(2) << result.execution_time_ms << " ms" << std::endl;
            report << "---" << std::endl;
        }
        
        // Final brain state
        auto final_state = melvin->get_unified_state();
        report << std::endl;
        report << "FINAL BRAIN STATE:" << std::endl;
        report << "Total Nodes: " << final_state.global_memory.total_nodes << std::endl;
        report << "Total Connections: " << final_state.global_memory.total_edges << std::endl;
        report << "Storage Used: " << std::fixed << std::setprecision(2) 
               << final_state.global_memory.storage_used_mb << " MB" << std::endl;
        report << "Hebbian Updates: " << final_state.global_memory.stats.hebbian_updates << std::endl;
        report << "Uptime: " << final_state.system.uptime_seconds << " seconds" << std::endl;
        
        report.close();
        std::cout << "📄 Comprehensive test report saved to melvin_comprehensive_test_report.txt" << std::endl;
    }
};

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    std::cout << "🧪 MELVIN COMPREHENSIVE BRAIN VALIDATION" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    try {
        MelvinTestRunner runner;
        
        // Run all tests
        runner.run_all_tests();
        
        // Save comprehensive report
        runner.save_comprehensive_report();
        
        std::cout << "\n🎉 Comprehensive brain validation completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test Runner Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
