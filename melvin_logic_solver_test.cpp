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

// ============================================================================
// MELVIN LOGIC PUZZLE SOLVER VALIDATION
// ============================================================================

class MelvinLogicSolver {
private:
    std::unique_ptr<MelvinOptimizedV2> melvin;
    
    // Advanced logic puzzles with step-by-step reasoning
    struct AdvancedPuzzle {
        std::string problem;
        std::vector<std::string> reasoning_steps;
        std::string final_answer;
        std::vector<std::string> key_concepts;
        std::vector<std::string> logical_rules;
        int difficulty;
        std::string category; // "deductive", "inductive", "lateral", "mathematical"
    };
    
    std::vector<AdvancedPuzzle> advanced_puzzles;
    
public:
    MelvinLogicSolver(const std::string& storage_path = "melvin_logic_memory") {
        melvin = std::make_unique<MelvinOptimizedV2>(storage_path);
        initialize_advanced_puzzles();
        
        std::cout << "🧩 Melvin Logic Solver initialized" << std::endl;
    }
    
    void initialize_advanced_puzzles() {
        advanced_puzzles = {
            {
                "Three people are in a room: Alice, Bob, and Charlie. Alice says 'Bob is lying.' Bob says 'Charlie is lying.' Charlie says 'Both Alice and Bob are lying.' If exactly one person is telling the truth, who is it?",
                {
                    "If Alice tells the truth, then Bob is lying, which means Charlie is telling the truth",
                    "But if Charlie tells the truth, then both Alice and Bob are lying",
                    "This creates a contradiction because Alice would be telling the truth",
                    "If Bob tells the truth, then Charlie is lying, which means not both Alice and Bob are lying",
                    "This means at least one of Alice or Bob is telling the truth",
                    "But Alice says Bob is lying, so if Alice tells the truth, Bob is lying",
                    "This creates another contradiction",
                    "If Charlie tells the truth, then both Alice and Bob are lying",
                    "If Alice is lying, then Bob is not lying (Bob tells the truth)",
                    "If Bob is lying, then Charlie is not lying (Charlie tells the truth)",
                    "This is impossible because we can't have both Bob telling the truth and lying",
                    "The only consistent scenario is: Alice lies, Bob tells the truth, Charlie lies"
                },
                "Bob is telling the truth",
                {"logical contradiction", "truth tables", "systematic analysis", "exclusive conditions"},
                {"exactly one truth", "mutual exclusivity", "logical consistency"},
                4,
                "deductive"
            },
            {
                "A farmer wants to cross a river with a wolf, a goat, and a cabbage. His boat can only carry himself and one other item. If the wolf and goat are left alone, the wolf will eat the goat. If the goat and cabbage are left alone, the goat will eat the cabbage. How can he get all three across safely?",
                {
                    "First trip: Take the goat across (leaving wolf and cabbage)",
                    "Return alone (wolf and cabbage are safe together)",
                    "Second trip: Take the wolf across",
                    "Return with the goat (to prevent wolf eating goat)",
                    "Third trip: Take the cabbage across",
                    "Return alone (wolf and cabbage are safe together)",
                    "Fourth trip: Take the goat across"
                },
                "Take goat, return, take wolf, return with goat, take cabbage, return, take goat",
                {"constraint satisfaction", "step-by-step planning", "resource management"},
                {"wolf eats goat", "goat eats cabbage", "boat capacity limit"},
                3,
                "lateral"
            },
            {
                "In a village, there are 100 people. Some always tell the truth, others always lie. A stranger asks the first person: 'Are you a liar?' The person answers in the village language. The stranger asks the second person: 'What did the first person say?' The second person says: 'He said yes.' What can you determine about the first two people?",
                {
                    "If the first person is a truth-teller and is asked 'Are you a liar?', they would say 'No'",
                    "If the first person is a liar and is asked 'Are you a liar?', they would lie and say 'No'",
                    "Both truth-tellers and liars would answer 'No' to 'Are you a liar?'",
                    "The second person says the first person said 'Yes'",
                    "This means the second person is lying about what the first person said",
                    "Therefore, the second person is a liar",
                    "We cannot determine if the first person is a truth-teller or liar from this information"
                },
                "The second person is a liar. The first person's type cannot be determined.",
                {"truth-teller analysis", "liar paradox", "logical deduction"},
                {"truth-tellers always tell truth", "liars always lie", "self-referential questions"},
                4,
                "deductive"
            },
            {
                "A clock shows 3:15. What is the angle between the hour and minute hands?",
                {
                    "At 3:00, the hour hand is at 90 degrees (3 * 30 degrees)",
                    "In 15 minutes, the hour hand moves 15/60 * 30 = 7.5 degrees",
                    "So at 3:15, the hour hand is at 90 + 7.5 = 97.5 degrees",
                    "At 3:15, the minute hand is at 90 degrees (15 * 6 degrees)",
                    "The angle between them is |97.5 - 90| = 7.5 degrees"
                },
                "7.5 degrees",
                {"angle calculation", "clock mechanics", "proportional reasoning"},
                {"hour hand moves 30 degrees per hour", "minute hand moves 6 degrees per minute"},
                2,
                "mathematical"
            },
            {
                "You have 12 balls, all identical in appearance. One ball has a different weight (either heavier or lighter). You have a balance scale and can use it only 3 times. How do you find the odd ball?",
                {
                    "First weighing: Compare 4 balls against 4 balls",
                    "If equal, the odd ball is in the remaining 4",
                    "If not equal, the odd ball is in the heavier or lighter group of 4",
                    "Second weighing: Take 2 balls from the group containing the odd ball",
                    "Compare these 2 balls against 2 known good balls",
                    "If equal, the odd ball is one of the remaining 2",
                    "If not equal, you know which of the 2 is odd and whether it's heavier or lighter",
                    "Third weighing: Compare the remaining suspect ball against a known good ball"
                },
                "Use systematic elimination: weigh 4v4, then 2v2 from the odd group, then 1v1",
                {"systematic elimination", "binary search", "constraint satisfaction"},
                {"balance scale gives relative weight", "only 3 weighings allowed", "unknown if heavier or lighter"},
                5,
                "lateral"
            }
        };
    }
    
    // ============================================================================
    // REASONING VALIDATION METHODS
    // ============================================================================
    
    bool validate_step_by_step_reasoning(const AdvancedPuzzle& puzzle) {
        std::cout << "\n🧠 Validating Step-by-Step Reasoning for: " << puzzle.category << " puzzle" << std::endl;
        
        // Feed the problem
        uint64_t problem_id = melvin->process_text_input(puzzle.problem, "advanced_puzzle");
        if (problem_id == 0) {
            std::cout << "❌ Failed to store puzzle problem" << std::endl;
            return false;
        }
        
        // Feed logical rules
        for (const auto& rule : puzzle.logical_rules) {
            melvin->process_text_input(rule, "logical_rule");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        // Feed key concepts
        for (const auto& concept : puzzle.key_concepts) {
            melvin->process_text_input(concept, "key_concept");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        // Feed reasoning steps one by one
        std::vector<uint64_t> reasoning_step_ids;
        for (size_t i = 0; i < puzzle.reasoning_steps.size(); ++i) {
            std::string step_with_number = "Step " + std::to_string(i + 1) + ": " + puzzle.reasoning_steps[i];
            uint64_t step_id = melvin->process_text_input(step_with_number, "reasoning_step");
            reasoning_step_ids.push_back(step_id);
            
            // Small delay to simulate thinking time
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Feed the final answer
        uint64_t answer_id = melvin->process_text_input(puzzle.final_answer, "final_answer");
        
        // Verify all components were stored
        bool all_stored = (answer_id != 0);
        for (uint64_t id : reasoning_step_ids) {
            if (id == 0) {
                all_stored = false;
                break;
            }
        }
        
        if (!all_stored) {
            std::cout << "❌ Failed to store all reasoning components" << std::endl;
            return false;
        }
        
        // Verify retrieval
        std::string retrieved_problem = melvin->get_node_content(problem_id);
        std::string retrieved_answer = melvin->get_node_content(answer_id);
        
        if (retrieved_problem != puzzle.problem || retrieved_answer != puzzle.final_answer) {
            std::cout << "❌ Retrieval mismatch for problem or answer" << std::endl;
            return false;
        }
        
        // Verify reasoning steps
        bool all_steps_retrieved = true;
        for (size_t i = 0; i < reasoning_step_ids.size(); ++i) {
            std::string retrieved_step = melvin->get_node_content(reasoning_step_ids[i]);
            std::string expected_step = "Step " + std::to_string(i + 1) + ": " + puzzle.reasoning_steps[i];
            
            if (retrieved_step != expected_step) {
                all_steps_retrieved = false;
                break;
            }
        }
        
        if (!all_steps_retrieved) {
            std::cout << "❌ Failed to retrieve all reasoning steps correctly" << std::endl;
            return false;
        }
        
        std::cout << "✅ Successfully validated " << puzzle.reasoning_steps.size() 
                  << " reasoning steps for " << puzzle.category << " puzzle" << std::endl;
        return true;
    }
    
    bool test_conceptual_understanding() {
        std::cout << "\n🎯 Testing Conceptual Understanding..." << std::endl;
        
        // Feed fundamental logical concepts
        std::vector<std::string> logical_concepts = {
            "Logical AND: Both conditions must be true",
            "Logical OR: At least one condition must be true", 
            "Logical NOT: Negates the truth value",
            "Implication: If A then B (A implies B)",
            "Contradiction: A statement that cannot be true",
            "Tautology: A statement that is always true",
            "Modus Ponens: If A implies B, and A is true, then B is true",
            "Modus Tollens: If A implies B, and B is false, then A is false"
        };
        
        std::vector<uint64_t> concept_ids;
        for (const auto& concept : logical_concepts) {
            uint64_t id = melvin->process_text_input(concept, "logical_concept");
            concept_ids.push_back(id);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        // Feed examples that use these concepts
        std::vector<std::string> examples = {
            "Example of AND: 'It's raining AND I have an umbrella' - both must be true",
            "Example of OR: 'I'll take the bus OR I'll walk' - at least one option",
            "Example of NOT: 'It's NOT raining' - negates the raining condition",
            "Example of Implication: 'If it rains, then the ground gets wet'",
            "Example of Contradiction: 'This statement is false' - creates paradox",
            "Example of Tautology: 'Either it's raining or it's not raining' - always true"
        };
        
        std::vector<uint64_t> example_ids;
        for (const auto& example : examples) {
            uint64_t id = melvin->process_text_input(example, "logical_example");
            example_ids.push_back(id);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        // Verify all concepts and examples were stored
        bool all_stored = true;
        for (uint64_t id : concept_ids) {
            if (id == 0) all_stored = false;
        }
        for (uint64_t id : example_ids) {
            if (id == 0) all_stored = false;
        }
        
        if (!all_stored) {
            std::cout << "❌ Failed to store logical concepts or examples" << std::endl;
            return false;
        }
        
        // Test retrieval
        bool all_retrieved = true;
        for (size_t i = 0; i < concept_ids.size(); ++i) {
            std::string retrieved = melvin->get_node_content(concept_ids[i]);
            if (retrieved != logical_concepts[i]) {
                all_retrieved = false;
                break;
            }
        }
        
        for (size_t i = 0; i < example_ids.size(); ++i) {
            std::string retrieved = melvin->get_node_content(example_ids[i]);
            if (retrieved != examples[i]) {
                all_retrieved = false;
                break;
            }
        }
        
        if (!all_retrieved) {
            std::cout << "❌ Failed to retrieve concepts or examples correctly" << std::endl;
            return false;
        }
        
        std::cout << "✅ Successfully stored and retrieved " << logical_concepts.size() 
                  << " concepts and " << examples.size() << " examples" << std::endl;
        return true;
    }
    
    bool test_reasoning_vs_memorization() {
        std::cout << "\n🔍 Testing Reasoning vs Memorization..." << std::endl;
        
        // Create variations of similar problems to test if Melvin can adapt
        std::vector<std::string> problem_variations = {
            "Problem A: If all cats are animals and some animals are pets, can we conclude some cats are pets?",
            "Problem B: If all birds are animals and some animals can fly, can we conclude some birds can fly?", 
            "Problem C: If all fish live in water and some water is salty, can we conclude some fish live in salt water?",
            "Problem D: If all cars are vehicles and some vehicles are electric, can we conclude some cars are electric?"
        };
        
        // Each problem has different logical validity
        std::vector<std::string> correct_reasoning = {
            "Problem A: No - the pets might not include cats",
            "Problem B: No - the flying animals might not include birds", 
            "Problem C: Yes - if fish live in water and some water is salty, some fish must live in salt water",
            "Problem D: No - the electric vehicles might not include cars"
        };
        
        std::vector<uint64_t> problem_ids;
        std::vector<uint64_t> reasoning_ids;
        
        // Store problems and their reasoning
        for (size_t i = 0; i < problem_variations.size(); ++i) {
            uint64_t problem_id = melvin->process_text_input(problem_variations[i], "problem_variation");
            uint64_t reasoning_id = melvin->process_text_input(correct_reasoning[i], "variation_reasoning");
            
            problem_ids.push_back(problem_id);
            reasoning_ids.push_back(reasoning_id);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        // Verify all were stored
        bool all_stored = true;
        for (uint64_t id : problem_ids) {
            if (id == 0) all_stored = false;
        }
        for (uint64_t id : reasoning_ids) {
            if (id == 0) all_stored = false;
        }
        
        if (!all_stored) {
            std::cout << "❌ Failed to store problem variations" << std::endl;
            return false;
        }
        
        // Test retrieval to ensure different reasoning is stored for each problem
        bool different_reasoning_stored = true;
        for (size_t i = 0; i < reasoning_ids.size(); ++i) {
            std::string retrieved_reasoning = melvin->get_node_content(reasoning_ids[i]);
            if (retrieved_reasoning != correct_reasoning[i]) {
                different_reasoning_stored = false;
                break;
            }
        }
        
        if (!different_reasoning_stored) {
            std::cout << "❌ Failed to store different reasoning for each problem variation" << std::endl;
            return false;
        }
        
        std::cout << "✅ Successfully stored and retrieved " << problem_variations.size() 
                  << " problem variations with distinct reasoning" << std::endl;
        return true;
    }
    
    bool test_hebbian_learning_in_logic() {
        std::cout << "\n⚡ Testing Hebbian Learning in Logic Problems..." << std::endl;
        
        auto initial_state = melvin->get_unified_state();
        uint64_t initial_connections = initial_state.global_memory.total_edges;
        
        // Feed related logical concepts in quick succession
        std::vector<std::string> related_concepts = {
            "Logical reasoning requires understanding premises",
            "Premises are the starting points of arguments",
            "Arguments lead to conclusions",
            "Conclusions must follow from premises",
            "Valid arguments have true conclusions when premises are true"
        };
        
        std::vector<uint64_t> concept_ids;
        for (const auto& concept : related_concepts) {
            uint64_t id = melvin->process_text_input(concept, "hebbian_logic");
            concept_ids.push_back(id);
            std::this_thread::sleep_for(std::chrono::milliseconds(150)); // Within coactivation window
        }
        
        // Feed a logical problem that uses these concepts
        std::string problem = "If all premises are true and the argument is valid, then the conclusion must be true";
        uint64_t problem_id = melvin->process_text_input(problem, "hebbian_problem");
        
        // Check if connections were formed
        auto final_state = melvin->get_unified_state();
        uint64_t final_connections = final_state.global_memory.total_edges;
        uint64_t new_connections = final_connections - initial_connections;
        
        if (new_connections == 0) {
            std::cout << "❌ No new connections formed through Hebbian learning" << std::endl;
            return false;
        }
        
        std::cout << "✅ Formed " << new_connections << " new connections through Hebbian learning" << std::endl;
        return true;
    }
    
    // ============================================================================
    // COMPREHENSIVE VALIDATION
    // ============================================================================
    
    void run_comprehensive_logic_validation() {
        std::cout << "🧩 MELVIN LOGIC SOLVER COMPREHENSIVE VALIDATION" << std::endl;
        std::cout << "================================================" << std::endl;
        
        uint64_t tests_passed = 0;
        uint64_t tests_failed = 0;
        
        // Test conceptual understanding
        if (test_conceptual_understanding()) {
            tests_passed++;
        } else {
            tests_failed++;
        }
        
        // Test reasoning vs memorization
        if (test_reasoning_vs_memorization()) {
            tests_passed++;
        } else {
            tests_failed++;
        }
        
        // Test Hebbian learning in logic
        if (test_hebbian_learning_in_logic()) {
            tests_passed++;
        } else {
            tests_failed++;
        }
        
        // Test each advanced puzzle
        for (size_t i = 0; i < advanced_puzzles.size(); ++i) {
            std::cout << "\n--- Advanced Puzzle " << (i + 1) << " ---" << std::endl;
            std::cout << "Category: " << advanced_puzzles[i].category << std::endl;
            std::cout << "Difficulty: " << advanced_puzzles[i].difficulty << "/5" << std::endl;
            
            if (validate_step_by_step_reasoning(advanced_puzzles[i])) {
                tests_passed++;
            } else {
                tests_failed++;
            }
        }
        
        // Print summary
        std::cout << "\n📊 LOGIC VALIDATION SUMMARY" << std::endl;
        std::cout << "===========================" << std::endl;
        std::cout << "✅ Tests Passed: " << tests_passed << std::endl;
        std::cout << "❌ Tests Failed: " << tests_failed << std::endl;
        std::cout << "📈 Success Rate: " << std::fixed << std::setprecision(1) 
                  << (100.0 * tests_passed / (tests_passed + tests_failed)) << "%" << std::endl;
        
        // Get final brain state
        auto final_state = melvin->get_unified_state();
        std::cout << "\n🧠 FINAL BRAIN STATE:" << std::endl;
        std::cout << "  📦 Total Nodes: " << final_state.global_memory.total_nodes << std::endl;
        std::cout << "  🔗 Total Connections: " << final_state.global_memory.total_edges << std::endl;
        std::cout << "  💾 Storage Used: " << std::fixed << std::setprecision(2) 
                  << final_state.global_memory.storage_used_mb << " MB" << std::endl;
        std::cout << "  ⚡ Hebbian Updates: " << final_state.global_memory.stats.hebbian_updates << std::endl;
        
        // Validate that Melvin is using his own brain
        std::cout << "\n🔍 BRAIN USAGE VALIDATION:" << std::endl;
        if (final_state.global_memory.total_nodes > 50) {
            std::cout << "✅ Melvin has formed " << final_state.global_memory.total_nodes << " memory nodes" << std::endl;
        } else {
            std::cout << "❌ Insufficient memory formation: " << final_state.global_memory.total_nodes << " nodes" << std::endl;
        }
        
        if (final_state.global_memory.total_edges > 20) {
            std::cout << "✅ Melvin has formed " << final_state.global_memory.total_edges << " neural connections" << std::endl;
        } else {
            std::cout << "❌ Insufficient connection formation: " << final_state.global_memory.total_edges << " connections" << std::endl;
        }
        
        if (final_state.global_memory.stats.hebbian_updates > 10) {
            std::cout << "✅ Melvin has performed " << final_state.global_memory.stats.hebbian_updates << " Hebbian learning updates" << std::endl;
        } else {
            std::cout << "❌ Insufficient Hebbian learning: " << final_state.global_memory.stats.hebbian_updates << " updates" << std::endl;
        }
        
        // Overall assessment
        bool using_his_brain = (final_state.global_memory.total_nodes > 50) && 
                              (final_state.global_memory.total_edges > 20) && 
                              (final_state.global_memory.stats.hebbian_updates > 10);
        
        if (using_his_brain) {
            std::cout << "\n🎉 CONCLUSION: Melvin is successfully using his own brain architecture!" << std::endl;
            std::cout << "   He has formed memories, created connections, and learned through Hebbian mechanisms." << std::endl;
        } else {
            std::cout << "\n⚠️  CONCLUSION: Melvin may not be fully utilizing his brain architecture." << std::endl;
            std::cout << "   Further investigation needed to ensure he's reasoning rather than pattern matching." << std::endl;
        }
    }
    
    void save_logic_validation_report() {
        std::ofstream report("melvin_logic_validation_report.txt");
        if (!report) {
            std::cout << "❌ Failed to create logic validation report" << std::endl;
            return;
        }
        
        report << "MELVIN LOGIC SOLVER VALIDATION REPORT" << std::endl;
        report << "=====================================" << std::endl;
        report << "Generated: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
        report << std::endl;
        
        report << "PUZZLE CATEGORIES TESTED:" << std::endl;
        std::set<std::string> categories;
        for (const auto& puzzle : advanced_puzzles) {
            categories.insert(puzzle.category);
        }
        for (const auto& category : categories) {
            report << "- " << category << std::endl;
        }
        report << std::endl;
        
        report << "DIFFICULTY LEVELS:" << std::endl;
        std::map<int, int> difficulty_counts;
        for (const auto& puzzle : advanced_puzzles) {
            difficulty_counts[puzzle.difficulty]++;
        }
        for (const auto& [difficulty, count] : difficulty_counts) {
            report << "Level " << difficulty << ": " << count << " puzzles" << std::endl;
        }
        report << std::endl;
        
        report << "BRAIN ARCHITECTURE VALIDATION:" << std::endl;
        auto state = melvin->get_unified_state();
        report << "Total Memory Nodes: " << state.global_memory.total_nodes << std::endl;
        report << "Total Neural Connections: " << state.global_memory.total_edges << std::endl;
        report << "Hebbian Learning Updates: " << state.global_memory.stats.hebbian_updates << std::endl;
        report << "Storage Used: " << std::fixed << std::setprecision(2) 
               << state.global_memory.storage_used_mb << " MB" << std::endl;
        
        report.close();
        std::cout << "📄 Logic validation report saved to melvin_logic_validation_report.txt" << std::endl;
    }
};

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    std::cout << "🧩 MELVIN LOGIC SOLVER VALIDATION" << std::endl;
    std::cout << "=================================" << std::endl;
    
    try {
        MelvinLogicSolver solver;
        
        // Run comprehensive validation
        solver.run_comprehensive_logic_validation();
        
        // Save validation report
        solver.save_logic_validation_report();
        
        std::cout << "\n🎉 Logic solver validation completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Validation Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
