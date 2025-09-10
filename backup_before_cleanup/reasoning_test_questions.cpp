#include <iostream>
#include <string>
#include <vector>
#include "melvin_optimized_v2.h"

class ReasoningTestGenerator {
private:
    std::vector<std::string> test_questions;
    
public:
    ReasoningTestGenerator() {
        initialize_test_questions();
    }
    
    void initialize_test_questions() {
        test_questions = {
            // Conceptual Metaphors & Personification
            "If shadows could remember the objects they came from, how would they describe them?",
            "What would a conversation between a river and a mountain sound like?",
            "If silence had a texture, how would it feel in your hands?",
            "What changes about 'friendship' if gravity suddenly stopped working?",
            "If a mirror broke but still wanted to reflect, how would it try?",
            
            // Abstract Concept Explanations
            "How would you explain the concept of 'sleep' to a machine that never turns off?",
            "If two memories collided, what new thing might they form?",
            "What might a tree dream about if it had dreams?",
            "If numbers had personalities, what would zero be like at a party?",
            "How would language evolve if humans could only speak once per day?",
            
            // Counterfactual Scenarios
            "What if colors had sounds instead of visual properties?",
            "If time flowed backwards, how would we experience nostalgia?",
            "What would happen if emotions were contagious diseases?",
            "If gravity worked in reverse, how would hugs feel?",
            "What if thoughts had weight and could be measured?",
            
            // Novel Combinations
            "How would you teach a cloud to count?",
            "What would a library look like if books were alive?",
            "If laughter could be stored in jars, what would you do with it?",
            "How would you explain 'home' to someone who's never had one?",
            "What if shadows could cast light instead of darkness?",
            
            // Temporal & Causal Reasoning
            "If yesterday and tomorrow switched places, what would today become?",
            "How would you describe 'waiting' to someone who exists in a single moment?",
            "What if cause and effect worked in reverse?",
            "If the past could see the future, what would it think?",
            "How would you explain 'hope' to someone who's never been disappointed?",
            
            // Sensory & Perceptual
            "If you could taste music, what would jazz sound like?",
            "What would it feel like to touch a rainbow?",
            "If you could smell emotions, what would joy smell like?",
            "How would you describe the color of a whisper?",
            "What if you could hear the sound of growing?",
            
            // Identity & Existence
            "If you were a different person every day, how would you know you were you?",
            "What would happen if your reflection had its own thoughts?",
            "If memories could choose their owners, which ones would leave?",
            "How would you explain 'self' to someone who's never been alone?",
            "What if your shadow had a different personality than you?",
            
            // Abstract Relationships
            "If distance and time were the same thing, how would you measure closeness?",
            "What would happen if trust was a physical object you could hold?",
            "If love had a shape, what would it look like?",
            "How would you explain 'missing someone' to an alien?",
            "What if understanding was a place you could visit?",
            
            // Creative Problem Solving
            "How would you build a bridge between two dreams?",
            "If you could plant ideas like seeds, what would grow?",
            "How would you teach a stone to dance?",
            "What if you could fold space like paper?",
            "How would you explain 'beauty' to someone who's never seen?",
            
            // Philosophical Paradoxes
            "If nothing existed, would that nothing be something?",
            "What would happen if the question 'why?' had an answer?",
            "If you could know everything, what would you not know?",
            "How would you explain 'freedom' to someone who's never been trapped?",
            "What if the answer to every question was another question?"
        };
    }
    
    void run_reasoning_tests(MelvinOptimizedV2& melvin, int num_tests = 10) {
        std::cout << "🧠 MELVIN REASONING TEST SUITE" << std::endl;
        std::cout << "=============================" << std::endl;
        std::cout << "Testing Blended Reasoning Protocol with exploration-heavy questions" << std::endl;
        std::cout << "These questions have no stored answers - forcing exploration track dominance" << std::endl;
        std::cout << "\n";
        
        for (int i = 0; i < num_tests && i < test_questions.size(); ++i) {
            std::cout << "[Test Question " << (i + 1) << "] " << test_questions[i] << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            
            // Process through blended reasoning
            std::string response = melvin.generate_intelligent_response(test_questions[i]);
            
            std::cout << "🤖 Melvin's Response:" << std::endl;
            std::cout << response << std::endl;
            
            // Analyze the reasoning tracks
            auto cognitive_result = melvin.process_cognitive_input(test_questions[i]);
            
            std::cout << "📊 Reasoning Analysis:" << std::endl;
            std::cout << "   • Overall Confidence: " << std::fixed << std::setprecision(2) 
                      << cognitive_result.blended_reasoning.overall_confidence << std::endl;
            std::cout << "   • Recall Weight: " << std::fixed << std::setprecision(0) 
                      << (cognitive_result.blended_reasoning.recall_weight * 100) << "%" << std::endl;
            std::cout << "   • Exploration Weight: " << std::fixed << std::setprecision(0) 
                      << (cognitive_result.blended_reasoning.exploration_weight * 100) << "%" << std::endl;
            
            // Update context for next iteration
            for (auto node_id : cognitive_result.activated_nodes) {
                melvin.update_conversation_context(node_id);
            }
            
            std::cout << "\n" << std::string(60, '=') << "\n" << std::endl;
        }
    }
    
    void run_confidence_analysis(MelvinOptimizedV2& melvin) {
        std::cout << "📈 CONFIDENCE ANALYSIS" << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "Analyzing how confidence affects track weighting" << std::endl;
        std::cout << "\n";
        
        // Test different types of questions
        std::vector<std::pair<std::string, std::string>> test_cases = {
            {"High Recall Potential", "What is the capital of France?"},
            {"Medium Recall/Exploration", "How do magnets work?"},
            {"Low Recall, High Exploration", "What would happen if gravity worked backwards?"},
            {"Pure Exploration", "If shadows could remember, how would they describe objects?"},
            {"Abstract Reasoning", "What is the color of a whisper?"}
        };
        
        for (const auto& [category, question] : test_cases) {
            std::cout << "🔍 " << category << ": " << question << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            
            auto result = melvin.process_cognitive_input(question);
            
            std::cout << "📊 Results:" << std::endl;
            std::cout << "   • Recall Confidence: " << std::fixed << std::setprecision(2) 
                      << result.blended_reasoning.recall_track.recall_confidence << std::endl;
            std::cout << "   • Exploration Confidence: " << std::fixed << std::setprecision(2) 
                      << result.blended_reasoning.exploration_track.exploration_confidence << std::endl;
            std::cout << "   • Overall Confidence: " << std::fixed << std::setprecision(2) 
                      << result.blended_reasoning.overall_confidence << std::endl;
            std::cout << "   • Recall Weight: " << std::fixed << std::setprecision(0) 
                      << (result.blended_reasoning.recall_weight * 100) << "%" << std::endl;
            std::cout << "   • Exploration Weight: " << std::fixed << std::setprecision(0) 
                      << (result.blended_reasoning.exploration_weight * 100) << "%" << std::endl;
            
            std::cout << "\n";
        }
    }
    
    void run_exploration_dominance_test(MelvinOptimizedV2& melvin) {
        std::cout << "🎯 EXPLORATION DOMINANCE TEST" << std::endl;
        std::cout << "============================" << std::endl;
        std::cout << "Testing questions that should force exploration track dominance" << std::endl;
        std::cout << "\n";
        
        std::vector<std::string> exploration_heavy_questions = {
            "If colors had personalities, what would red be like at a party?",
            "How would you explain 'loneliness' to someone who's never been alone?",
            "What if thoughts were visible and floated around your head?",
            "If you could taste emotions, what would sadness taste like?",
            "How would you teach a cloud to count?"
        };
        
        for (size_t i = 0; i < exploration_heavy_questions.size(); ++i) {
            std::cout << "[Exploration Test " << (i + 1) << "] " << exploration_heavy_questions[i] << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            
            auto result = melvin.process_cognitive_input(exploration_heavy_questions[i]);
            
            std::cout << "🧠 Blended Reasoning Response:" << std::endl;
            std::string response = melvin.generate_intelligent_response(exploration_heavy_questions[i]);
            std::cout << response << std::endl;
            
            // Check if exploration is dominant
            bool exploration_dominant = result.blended_reasoning.exploration_weight > result.blended_reasoning.recall_weight;
            std::cout << "📊 Exploration Dominant: " << (exploration_dominant ? "✅ YES" : "❌ NO") << std::endl;
            std::cout << "   • Exploration Weight: " << std::fixed << std::setprecision(0) 
                      << (result.blended_reasoning.exploration_weight * 100) << "%" << std::endl;
            std::cout << "   • Recall Weight: " << std::fixed << std::setprecision(0) 
                      << (result.blended_reasoning.recall_weight * 100) << "%" << std::endl;
            
            std::cout << "\n" << std::string(60, '=') << "\n" << std::endl;
        }
    }
    
    void print_all_questions() {
        std::cout << "📋 COMPLETE REASONING TEST QUESTION BANK" << std::endl;
        std::cout << "=======================================" << std::endl;
        std::cout << "Total questions: " << test_questions.size() << std::endl;
        std::cout << "\n";
        
        for (size_t i = 0; i < test_questions.size(); ++i) {
            std::cout << "[Test Question " << (i + 1) << "] " << test_questions[i] << std::endl;
        }
    }
};

int main() {
    try {
        MelvinOptimizedV2 melvin;
        ReasoningTestGenerator test_generator;
        
        std::cout << "🧠 MELVIN REASONING TEST GENERATOR" << std::endl;
        std::cout << "==================================" << std::endl;
        std::cout << "Testing Blended Reasoning Protocol with exploration-heavy questions" << std::endl;
        std::cout << "\n";
        
        // Run reasoning tests
        test_generator.run_reasoning_tests(melvin, 10);
        
        // Run confidence analysis
        test_generator.run_confidence_analysis(melvin);
        
        // Run exploration dominance test
        test_generator.run_exploration_dominance_test(melvin);
        
        // Print all available questions
        test_generator.print_all_questions();
        
        std::cout << "\n🎉 Reasoning Test Suite Complete!" << std::endl;
        std::cout << "These questions are designed to force Melvin into exploration-heavy reasoning," << std::endl;
        std::cout << "testing the Blended Reasoning Protocol under limited data scenarios." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during testing: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
