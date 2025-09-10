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
#include <map>
#include <set>
#include <thread>

// ============================================================================
// MELVIN MEMORY FORMATION AND RETRIEVAL VALIDATION
// ============================================================================

class MelvinMemoryTester {
private:
    std::unique_ptr<MelvinOptimizedV2> melvin;
    
    // Memory test structures
    struct MemoryTest {
        std::string content;
        std::string category;
        uint64_t node_id;
        uint64_t creation_time;
        bool retrieved_successfully;
        double retrieval_time_ms;
    };
    
    std::vector<MemoryTest> memory_tests;
    std::map<std::string, std::vector<uint64_t>> category_nodes;
    
public:
    MelvinMemoryTester(const std::string& storage_path = "melvin_memory_test") {
        melvin = std::make_unique<MelvinOptimizedV2>(storage_path);
        std::cout << "🧠 Melvin Memory Tester initialized" << std::endl;
    }
    
    // ============================================================================
    // BASIC MEMORY FORMATION TESTS
    // ============================================================================
    
    bool test_basic_memory_storage() {
        std::cout << "\n📦 Testing Basic Memory Storage..." << std::endl;
        
        std::vector<std::string> test_contents = {
            "Memory test: storing simple text content",
            "Memory test: storing numerical data 12345",
            "Memory test: storing special characters !@#$%^&*()",
            "Memory test: storing unicode characters αβγδε",
            "Memory test: storing long content with multiple words and sentences to test memory capacity and retrieval mechanisms"
        };
        
        std::vector<uint64_t> stored_ids;
        
        for (const auto& content : test_contents) {
            auto start_time = std::chrono::high_resolution_clock::now();
            uint64_t node_id = melvin->process_text_input(content, "basic_memory_test");
            auto end_time = std::chrono::high_resolution_clock::now();
            
            if (node_id == 0) {
                std::cout << "❌ Failed to store content: " << content.substr(0, 50) << "..." << std::endl;
                return false;
            }
            
            double storage_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            MemoryTest test;
            test.content = content;
            test.category = "basic";
            test.node_id = node_id;
            test.creation_time = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            test.retrieved_successfully = false;
            test.retrieval_time_ms = 0.0;
            
            memory_tests.push_back(test);
            stored_ids.push_back(node_id);
            
            std::cout << "✅ Stored: " << content.substr(0, 30) << "... (ID: " << std::hex << node_id << ") in " 
                      << std::fixed << std::setprecision(2) << storage_time << "ms" << std::endl;
        }
        
        std::cout << "✅ Successfully stored " << stored_ids.size() << " memory items" << std::endl;
        return true;
    }
    
    bool test_memory_retrieval() {
        std::cout << "\n🔍 Testing Memory Retrieval..." << std::endl;
        
        bool all_retrieved = true;
        
        for (auto& test : memory_tests) {
            auto start_time = std::chrono::high_resolution_clock::now();
            std::string retrieved_content = melvin->get_node_content(test.node_id);
            auto end_time = std::chrono::high_resolution_clock::now();
            
            test.retrieval_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            if (retrieved_content == test.content) {
                test.retrieved_successfully = true;
                std::cout << "✅ Retrieved: " << retrieved_content.substr(0, 30) << "... in " 
                          << std::fixed << std::setprecision(2) << test.retrieval_time_ms << "ms" << std::endl;
            } else {
                test.retrieved_successfully = false;
                std::cout << "❌ Retrieval mismatch for ID " << std::hex << test.node_id << std::endl;
                std::cout << "   Expected: " << test.content.substr(0, 50) << "..." << std::endl;
                std::cout << "   Retrieved: " << retrieved_content.substr(0, 50) << "..." << std::endl;
                all_retrieved = false;
            }
        }
        
        if (all_retrieved) {
            std::cout << "✅ All " << memory_tests.size() << " memories retrieved successfully" << std::endl;
        } else {
            std::cout << "❌ Some memories failed retrieval" << std::endl;
        }
        
        return all_retrieved;
    }
    
    // ============================================================================
    // CATEGORICAL MEMORY TESTS
    // ============================================================================
    
    bool test_categorical_memory_formation() {
        std::cout << "\n📚 Testing Categorical Memory Formation..." << std::endl;
        
        std::map<std::string, std::vector<std::string>> categories = {
            {"mathematics", {
                "Addition is commutative: a + b = b + a",
                "Multiplication is associative: (a * b) * c = a * (b * c)",
                "The distributive property: a * (b + c) = a * b + a * c",
                "Zero is the additive identity: a + 0 = a",
                "One is the multiplicative identity: a * 1 = a"
            }},
            {"science", {
                "Water boils at 100 degrees Celsius at standard pressure",
                "The speed of light in vacuum is approximately 299,792,458 m/s",
                "Photosynthesis converts light energy into chemical energy",
                "DNA contains the genetic information of living organisms",
                "Gravity causes objects to accelerate toward each other"
            }},
            {"programming", {
                "Variables store data values in computer programs",
                "Functions are reusable blocks of code that perform specific tasks",
                "Loops allow repeated execution of code blocks",
                "Conditional statements control program flow based on conditions",
                "Arrays store multiple values of the same data type"
            }},
            {"philosophy", {
                "Cogito ergo sum: I think, therefore I am",
                "The unexamined life is not worth living",
                "Knowledge is justified true belief",
                "The categorical imperative: act only according to maxims you can will as universal laws",
                "Existence precedes essence in human beings"
            }}
        };
        
        for (const auto& [category, contents] : categories) {
            std::cout << "\n--- Category: " << category << " ---" << std::endl;
            
            for (const auto& content : contents) {
                uint64_t node_id = melvin->process_text_input(content, category);
                
                if (node_id == 0) {
                    std::cout << "❌ Failed to store " << category << " content" << std::endl;
                    return false;
                }
                
                category_nodes[category].push_back(node_id);
                
                MemoryTest test;
                test.content = content;
                test.category = category;
                test.node_id = node_id;
                test.creation_time = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                test.retrieved_successfully = false;
                test.retrieval_time_ms = 0.0;
                
                memory_tests.push_back(test);
                
                std::cout << "✅ Stored: " << content.substr(0, 40) << "..." << std::endl;
                
                // Small delay to simulate natural learning
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        
        std::cout << "\n✅ Successfully formed categorical memories:" << std::endl;
        for (const auto& [category, nodes] : category_nodes) {
            std::cout << "   " << category << ": " << nodes.size() << " memories" << std::endl;
        }
        
        return true;
    }
    
    bool test_categorical_memory_retrieval() {
        std::cout << "\n🔍 Testing Categorical Memory Retrieval..." << std::endl;
        
        bool all_categories_retrieved = true;
        
        for (const auto& [category, nodes] : category_nodes) {
            std::cout << "\n--- Retrieving " << category << " memories ---" << std::endl;
            
            bool category_success = true;
            for (uint64_t node_id : nodes) {
                std::string retrieved_content = melvin->get_node_content(node_id);
                
                // Find the corresponding test
                auto test_it = std::find_if(memory_tests.begin(), memory_tests.end(),
                    [node_id](const MemoryTest& test) { return test.node_id == node_id; });
                
                if (test_it != memory_tests.end()) {
                    if (retrieved_content == test_it->content) {
                        std::cout << "✅ Retrieved: " << retrieved_content.substr(0, 40) << "..." << std::endl;
                        test_it->retrieved_successfully = true;
                    } else {
                        std::cout << "❌ Retrieval mismatch for " << category << " memory" << std::endl;
                        category_success = false;
                        test_it->retrieved_successfully = false;
                    }
                } else {
                    std::cout << "❌ Could not find test for node ID " << std::hex << node_id << std::endl;
                    category_success = false;
                }
            }
            
            if (category_success) {
                std::cout << "✅ All " << category << " memories retrieved successfully" << std::endl;
            } else {
                std::cout << "❌ Some " << category << " memories failed retrieval" << std::endl;
                all_categories_retrieved = false;
            }
        }
        
        return all_categories_retrieved;
    }
    
    // ============================================================================
    // MEMORY PERSISTENCE TESTS
    // ============================================================================
    
    bool test_memory_persistence() {
        std::cout << "\n💾 Testing Memory Persistence..." << std::endl;
        
        // Get current brain state
        auto initial_state = melvin->get_unified_state();
        uint64_t initial_nodes = initial_state.global_memory.total_nodes;
        
        // Create a new Melvin instance with the same storage path
        std::cout << "Creating new Melvin instance to test persistence..." << std::endl;
        auto persistent_melvin = std::make_unique<MelvinOptimizedV2>("melvin_memory_test");
        
        // Get state from new instance
        auto persistent_state = persistent_melvin->get_unified_state();
        uint64_t persistent_nodes = persistent_state.global_memory.total_nodes;
        
        std::cout << "Initial nodes: " << initial_nodes << std::endl;
        std::cout << "Persistent nodes: " << persistent_nodes << std::endl;
        
        if (persistent_nodes >= initial_nodes) {
            std::cout << "✅ Memory persistence verified - nodes maintained across instances" << std::endl;
            
            // Test retrieval from persistent instance
            bool persistent_retrieval_success = true;
            for (const auto& test : memory_tests) {
                std::string retrieved_content = persistent_melvin->get_node_content(test.node_id);
                if (retrieved_content != test.content) {
                    std::cout << "❌ Persistent retrieval failed for node " << std::hex << test.node_id << std::endl;
                    persistent_retrieval_success = false;
                }
            }
            
            if (persistent_retrieval_success) {
                std::cout << "✅ All memories retrieved successfully from persistent instance" << std::endl;
                return true;
            } else {
                std::cout << "❌ Some memories failed retrieval from persistent instance" << std::endl;
                return false;
            }
        } else {
            std::cout << "❌ Memory persistence failed - nodes lost across instances" << std::endl;
            return false;
        }
    }
    
    // ============================================================================
    // MEMORY ASSOCIATION TESTS
    // ============================================================================
    
    bool test_memory_associations() {
        std::cout << "\n🔗 Testing Memory Associations..." << std::endl;
        
        // Feed related concepts in quick succession to trigger associations
        std::vector<std::string> related_concepts = {
            "Memory is the faculty of the mind by which information is encoded, stored, and retrieved",
            "Encoding is the process of converting information into a form that can be stored in memory",
            "Storage is the retention of encoded information over time",
            "Retrieval is the process of accessing stored information when needed",
            "Short-term memory has limited capacity and duration",
            "Long-term memory has virtually unlimited capacity and duration"
        };
        
        std::vector<uint64_t> concept_ids;
        for (const auto& concept : related_concepts) {
            uint64_t node_id = melvin->process_text_input(concept, "memory_association");
            concept_ids.push_back(node_id);
            
            // Small delay to stay within coactivation window
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Check if associations were formed by examining brain state
        auto state = melvin->get_unified_state();
        uint64_t total_connections = state.global_memory.total_edges;
        uint64_t hebbian_updates = state.global_memory.stats.hebbian_updates;
        
        std::cout << "Total connections: " << total_connections << std::endl;
        std::cout << "Hebbian updates: " << hebbian_updates << std::endl;
        
        if (total_connections > 0 && hebbian_updates > 0) {
            std::cout << "✅ Memory associations formed through Hebbian learning" << std::endl;
            return true;
        } else {
            std::cout << "❌ No memory associations detected" << std::endl;
            return false;
        }
    }
    
    // ============================================================================
    // MEMORY PERFORMANCE TESTS
    // ============================================================================
    
    bool test_memory_performance() {
        std::cout << "\n⚡ Testing Memory Performance..." << std::endl;
        
        // Test storage performance
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<uint64_t> performance_test_ids;
        for (int i = 0; i < 100; ++i) {
            std::string content = "Performance test item " + std::to_string(i) + 
                " with additional content to test memory storage and retrieval performance";
            uint64_t node_id = melvin->process_text_input(content, "performance_test");
            performance_test_ids.push_back(node_id);
        }
        
        auto storage_end_time = std::chrono::high_resolution_clock::now();
        double storage_time = std::chrono::duration<double, std::milli>(storage_end_time - start_time).count();
        
        // Test retrieval performance
        auto retrieval_start_time = std::chrono::high_resolution_clock::now();
        
        bool all_retrieved = true;
        for (uint64_t node_id : performance_test_ids) {
            std::string retrieved_content = melvin->get_node_content(node_id);
            if (retrieved_content.empty()) {
                all_retrieved = false;
            }
        }
        
        auto retrieval_end_time = std::chrono::high_resolution_clock::now();
        double retrieval_time = std::chrono::duration<double, std::milli>(retrieval_end_time - retrieval_start_time).count();
        
        // Calculate performance metrics
        double storage_rate = 100.0 / (storage_time / 1000.0); // items per second
        double retrieval_rate = 100.0 / (retrieval_time / 1000.0); // items per second
        
        std::cout << "Storage Performance:" << std::endl;
        std::cout << "   Time: " << std::fixed << std::setprecision(2) << storage_time << " ms" << std::endl;
        std::cout << "   Rate: " << storage_rate << " items/second" << std::endl;
        
        std::cout << "Retrieval Performance:" << std::endl;
        std::cout << "   Time: " << retrieval_time << " ms" << std::endl;
        std::cout << "   Rate: " << retrieval_rate << " items/second" << std::endl;
        
        // Performance thresholds
        bool storage_fast_enough = storage_rate >= 50; // At least 50 items/second
        bool retrieval_fast_enough = retrieval_rate >= 100; // At least 100 items/second
        
        if (storage_fast_enough && retrieval_fast_enough && all_retrieved) {
            std::cout << "✅ Memory performance meets requirements" << std::endl;
            return true;
        } else {
            std::cout << "❌ Memory performance below requirements" << std::endl;
            if (!storage_fast_enough) std::cout << "   Storage too slow: " << storage_rate << " items/second" << std::endl;
            if (!retrieval_fast_enough) std::cout << "   Retrieval too slow: " << retrieval_rate << " items/second" << std::endl;
            if (!all_retrieved) std::cout << "   Some retrievals failed" << std::endl;
            return false;
        }
    }
    
    // ============================================================================
    // COMPREHENSIVE MEMORY VALIDATION
    // ============================================================================
    
    void run_comprehensive_memory_validation() {
        std::cout << "🧠 MELVIN MEMORY FORMATION AND RETRIEVAL VALIDATION" << std::endl;
        std::cout << "=================================================" << std::endl;
        
        uint64_t tests_passed = 0;
        uint64_t tests_failed = 0;
        
        // Basic memory tests
        if (test_basic_memory_storage()) {
            tests_passed++;
        } else {
            tests_failed++;
        }
        
        if (test_memory_retrieval()) {
            tests_passed++;
        } else {
            tests_failed++;
        }
        
        // Categorical memory tests
        if (test_categorical_memory_formation()) {
            tests_passed++;
        } else {
            tests_failed++;
        }
        
        if (test_categorical_memory_retrieval()) {
            tests_passed++;
        } else {
            tests_failed++;
        }
        
        // Persistence tests
        if (test_memory_persistence()) {
            tests_passed++;
        } else {
            tests_failed++;
        }
        
        // Association tests
        if (test_memory_associations()) {
            tests_passed++;
        } else {
            tests_failed++;
        }
        
        // Performance tests
        if (test_memory_performance()) {
            tests_passed++;
        } else {
            tests_failed++;
        }
        
        // Print comprehensive summary
        print_memory_validation_summary(tests_passed, tests_failed);
    }
    
    void print_memory_validation_summary(uint64_t tests_passed, uint64_t tests_failed) {
        std::cout << "\n📊 MEMORY VALIDATION SUMMARY" << std::endl;
        std::cout << "============================" << std::endl;
        std::cout << "✅ Tests Passed: " << tests_passed << std::endl;
        std::cout << "❌ Tests Failed: " << tests_failed << std::endl;
        std::cout << "📈 Success Rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * tests_passed / (tests_passed + tests_failed)) << "%" << std::endl;
        
        // Memory statistics
        std::cout << "\n🧠 MEMORY STATISTICS:" << std::endl;
        std::cout << "Total Memory Tests: " << memory_tests.size() << std::endl;
        
        uint64_t successful_retrievals = 0;
        double total_retrieval_time = 0.0;
        
        for (const auto& test : memory_tests) {
            if (test.retrieved_successfully) {
                successful_retrievals++;
                total_retrieval_time += test.retrieval_time_ms;
            }
        }
        
        std::cout << "Successful Retrievals: " << successful_retrievals << "/" << memory_tests.size() << std::endl;
        if (successful_retrievals > 0) {
            double avg_retrieval_time = total_retrieval_time / successful_retrievals;
            std::cout << "Average Retrieval Time: " << std::fixed << std::setprecision(2) 
                      << avg_retrieval_time << " ms" << std::endl;
        }
        
        // Category breakdown
        std::cout << "\n📚 MEMORY CATEGORIES:" << std::endl;
        for (const auto& [category, nodes] : category_nodes) {
            std::cout << category << ": " << nodes.size() << " memories" << std::endl;
        }
        
        // Final brain state
        auto final_state = melvin->get_unified_state();
        std::cout << "\n🧠 FINAL BRAIN STATE:" << std::endl;
        std::cout << "  📦 Total Nodes: " << final_state.global_memory.total_nodes << std::endl;
        std::cout << "  🔗 Total Connections: " << final_state.global_memory.total_edges << std::endl;
        std::cout << "  💾 Storage Used: " << std::fixed << std::setprecision(2) 
                  << final_state.global_memory.storage_used_mb << " MB" << std::endl;
        std::cout << "  ⚡ Hebbian Updates: " << final_state.global_memory.stats.hebbian_updates << std::endl;
        
        // Memory formation validation
        bool memory_formation_successful = (final_state.global_memory.total_nodes > 50) && 
                                           (successful_retrievals >= memory_tests.size() * 0.9); // 90% success rate
        
        if (memory_formation_successful) {
            std::cout << "\n🎉 MEMORY FORMATION VALIDATION: SUCCESSFUL" << std::endl;
            std::cout << "   Melvin has successfully formed and retrieved memories using his brain architecture." << std::endl;
        } else {
            std::cout << "\n⚠️  MEMORY FORMATION VALIDATION: NEEDS IMPROVEMENT" << std::endl;
            std::cout << "   Memory formation or retrieval may need optimization." << std::endl;
        }
    }
    
    void save_memory_validation_report() {
        std::ofstream report("melvin_memory_validation_report.txt");
        if (!report) {
            std::cout << "❌ Failed to create memory validation report" << std::endl;
            return;
        }
        
        report << "MELVIN MEMORY FORMATION AND RETRIEVAL VALIDATION REPORT" << std::endl;
        report << "=====================================================" << std::endl;
        report << "Generated: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
        report << std::endl;
        
        report << "MEMORY TEST SUMMARY:" << std::endl;
        report << "Total Tests: " << memory_tests.size() << std::endl;
        
        uint64_t successful_retrievals = 0;
        for (const auto& test : memory_tests) {
            if (test.retrieved_successfully) successful_retrievals++;
        }
        
        report << "Successful Retrievals: " << successful_retrievals << std::endl;
        report << "Success Rate: " << std::fixed << std::setprecision(1) 
               << (100.0 * successful_retrievals / memory_tests.size()) << "%" << std::endl;
        report << std::endl;
        
        report << "CATEGORY BREAKDOWN:" << std::endl;
        for (const auto& [category, nodes] : category_nodes) {
            report << category << ": " << nodes.size() << " memories" << std::endl;
        }
        report << std::endl;
        
        report << "DETAILED TEST RESULTS:" << std::endl;
        for (const auto& test : memory_tests) {
            report << "Category: " << test.category << std::endl;
            report << "Content: " << test.content.substr(0, 50) << "..." << std::endl;
            report << "Retrieved: " << (test.retrieved_successfully ? "YES" : "NO") << std::endl;
            report << "Retrieval Time: " << std::fixed << std::setprecision(2) 
                   << test.retrieval_time_ms << " ms" << std::endl;
            report << "---" << std::endl;
        }
        
        report.close();
        std::cout << "📄 Memory validation report saved to melvin_memory_validation_report.txt" << std::endl;
    }
};

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    std::cout << "🧠 MELVIN MEMORY FORMATION AND RETRIEVAL VALIDATION" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    try {
        MelvinMemoryTester tester;
        
        // Run comprehensive memory validation
        tester.run_comprehensive_memory_validation();
        
        // Save validation report
        tester.save_memory_validation_report();
        
        std::cout << "\n🎉 Memory validation completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Memory Validation Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
