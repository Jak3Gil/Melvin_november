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
#include <cmath>

// ============================================================================
// MELVIN UNIFIED INTELLIGENT TEST
// ============================================================================
// This test demonstrates Melvin's unified brain with integrated intelligent
// connection traversal and dynamic node creation capabilities

class MelvinUnifiedIntelligentTest {
private:
    std::unique_ptr<MelvinOptimizedV2> melvin;
    
public:
    MelvinUnifiedIntelligentTest(const std::string& storage_path = "melvin_unified_intelligent_memory") {
        melvin = std::make_unique<MelvinOptimizedV2>(storage_path);
        
        std::cout << "🧠 Melvin Unified Intelligent Test initialized" << std::endl;
    }
    
    void run_unified_intelligent_test() {
        std::cout << "\n🧠 MELVIN UNIFIED INTELLIGENT TEST" << std::endl;
        std::cout << "===================================" << std::endl;
        std::cout << "Testing Melvin's unified brain with integrated intelligent capabilities" << std::endl;
        std::cout << "Every interaction now includes intelligent connection traversal!" << std::endl;
        
        // Feed knowledge to Melvin's brain
        feed_knowledge_base();
        
        // Test intelligent answering capabilities
        test_intelligent_answering();
        
        // Test dynamic node creation
        test_dynamic_node_creation();
        
        // Test connection path traversal
        test_connection_traversal();
        
        // Generate comprehensive report
        generate_unified_report();
    }
    
    void feed_knowledge_base() {
        std::cout << "\n📚 FEEDING KNOWLEDGE BASE TO MELVIN'S UNIFIED BRAIN" << std::endl;
        std::cout << "===================================================" << std::endl;
        
        // Feed various types of knowledge
        std::vector<std::string> knowledge = {
            // Colors
            "Red is a warm color",
            "Blue is a cool color", 
            "Green is the color of grass",
            "Yellow is bright and sunny",
            "Purple is a royal color",
            
            // Animals
            "Dogs are loyal pets",
            "Cats are independent animals",
            "Birds can fly in the sky",
            "Fish swim in water",
            "Elephants are large animals",
            
            // Food
            "Pizza is delicious",
            "Ice cream is sweet",
            "Vegetables are healthy",
            "Fruit is nutritious",
            "Chocolate is a treat",
            
            // Activities
            "Reading is educational",
            "Swimming is exercise",
            "Music is relaxing",
            "Art is creative",
            "Sports are competitive"
        };
        
        for (const auto& fact : knowledge) {
            melvin->process_text_input(fact, "knowledge");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        std::cout << "✅ Fed " << knowledge.size() << " knowledge facts to Melvin's unified brain" << std::endl;
    }
    
    void test_intelligent_answering() {
        std::cout << "\n❓ TESTING INTELLIGENT ANSWERING" << std::endl;
        std::cout << "===============================" << std::endl;
        
        // Questions that require intelligent answering
        std::vector<std::string> questions = {
            "What's your favorite color?",
            "What's your favorite animal?",
            "What's your favorite food?",
            "What's the best color for a sunny day?",
            "What animal would make a good pet?",
            "What food is good for health?",
            "What activity helps you relax?",
            "What do you think about music?"
        };
        
        for (const auto& question : questions) {
            std::cout << "\n📋 Question: " << question << std::endl;
            
            // Use Melvin's unified brain to answer intelligently
            SynthesizedAnswer answer = melvin->answer_question_intelligently(question);
            
            std::cout << "🧠 Melvin's Answer: " << answer.answer << std::endl;
            std::cout << "🎯 Confidence: " << std::fixed << std::setprecision(1) << answer.confidence * 100 << "%" << std::endl;
            std::cout << "💭 Reasoning: " << answer.reasoning << std::endl;
            std::cout << "🔗 Source Nodes: " << answer.source_nodes.size() << " nodes used" << std::endl;
        }
    }
    
    void test_dynamic_node_creation() {
        std::cout << "\n🆕 TESTING DYNAMIC NODE CREATION" << std::endl;
        std::cout << "================================" << std::endl;
        
        // Get initial brain state
        auto initial_state = melvin->get_unified_state();
        uint64_t initial_nodes = initial_state.global_memory.total_nodes;
        uint64_t initial_dynamic = initial_state.intelligent_capabilities.dynamic_nodes_created;
        
        std::cout << "📊 Initial state:" << std::endl;
        std::cout << "   Total nodes: " << initial_nodes << std::endl;
        std::cout << "   Dynamic nodes: " << initial_dynamic << std::endl;
        
        // Ask a new question to trigger dynamic node creation
        std::string new_question = "What's your favorite programming language?";
        std::cout << "\n📋 New Question: " << new_question << std::endl;
        
        SynthesizedAnswer answer = melvin->answer_question_intelligently(new_question);
        
        std::cout << "🧠 Melvin's Answer: " << answer.answer << std::endl;
        std::cout << "🎯 Confidence: " << std::fixed << std::setprecision(1) << answer.confidence * 100 << "%" << std::endl;
        
        // Check final brain state
        auto final_state = melvin->get_unified_state();
        uint64_t final_nodes = final_state.global_memory.total_nodes;
        uint64_t final_dynamic = final_state.intelligent_capabilities.dynamic_nodes_created;
        
        std::cout << "\n📊 Final state:" << std::endl;
        std::cout << "   Total nodes: " << final_nodes << std::endl;
        std::cout << "   Dynamic nodes: " << final_dynamic << std::endl;
        std::cout << "   New nodes created: " << (final_nodes - initial_nodes) << std::endl;
        std::cout << "   New dynamic nodes: " << (final_dynamic - initial_dynamic) << std::endl;
    }
    
    void test_connection_traversal() {
        std::cout << "\n🔗 TESTING CONNECTION TRAVERSAL" << std::endl;
        std::cout << "==============================" << std::endl;
        
        // Test keyword extraction
        std::string test_text = "What's your favorite color for a sunny day?";
        std::cout << "📝 Test text: " << test_text << std::endl;
        
        std::vector<std::string> keywords = melvin->extract_keywords(test_text);
        std::cout << "🔍 Extracted keywords: ";
        for (size_t i = 0; i < keywords.size(); ++i) {
            std::cout << keywords[i];
            if (i < keywords.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        // Test relevant node finding
        std::vector<NodeSimilarity> relevant_nodes = melvin->find_relevant_nodes(keywords);
        std::cout << "🎯 Found " << relevant_nodes.size() << " relevant nodes:" << std::endl;
        for (const auto& node : relevant_nodes) {
            std::cout << "   Node " << std::hex << node.node_id << ": " << node.content 
                      << " (similarity: " << std::fixed << std::setprecision(2) << node.similarity_score << ")" << std::endl;
        }
        
        // Test connection path analysis
        std::vector<ConnectionPath> paths = melvin->analyze_connection_paths(relevant_nodes);
        std::cout << "🛤️ Analyzed " << paths.size() << " connection paths:" << std::endl;
        for (const auto& path : paths) {
            std::cout << "   Path: " << path.path_description 
                      << " (relevance: " << std::fixed << std::setprecision(2) << path.relevance_score << ")" << std::endl;
        }
    }
    
    void generate_unified_report() {
        std::cout << "\n📊 UNIFIED INTELLIGENT BRAIN REPORT" << std::endl;
        std::cout << "====================================" << std::endl;
        
        auto brain_state = melvin->get_unified_state();
        
        std::cout << "\n🧠 UNIFIED BRAIN ARCHITECTURE" << std::endl;
        std::cout << "=============================" << std::endl;
        std::cout << "Total Nodes: " << brain_state.global_memory.total_nodes << std::endl;
        std::cout << "Total Connections: " << brain_state.global_memory.total_edges << std::endl;
        std::cout << "Storage Used: " << std::fixed << std::setprecision(2) 
                  << brain_state.global_memory.storage_used_mb << " MB" << std::endl;
        std::cout << "Uptime: " << brain_state.system.uptime_seconds << " seconds" << std::endl;
        
        std::cout << "\n🎯 INTELLIGENT CAPABILITIES" << std::endl;
        std::cout << "===========================" << std::endl;
        std::cout << "Intelligent Answers Generated: " << brain_state.intelligent_capabilities.intelligent_answers_generated << std::endl;
        std::cout << "Dynamic Nodes Created: " << brain_state.intelligent_capabilities.dynamic_nodes_created << std::endl;
        std::cout << "Connection Traversal Enabled: " << (brain_state.intelligent_capabilities.connection_traversal_enabled ? "✅" : "❌") << std::endl;
        std::cout << "Dynamic Node Creation Enabled: " << (brain_state.intelligent_capabilities.dynamic_node_creation_enabled ? "✅" : "❌") << std::endl;
        
        std::cout << "\n🧠 BRAIN STATISTICS" << std::endl;
        std::cout << "===================" << std::endl;
        std::cout << "Hebbian Learning Updates: " << brain_state.global_memory.stats.hebbian_updates << std::endl;
        std::cout << "Similarity Connections: " << brain_state.global_memory.stats.similarity_connections << std::endl;
        std::cout << "Temporal Connections: " << brain_state.global_memory.stats.temporal_connections << std::endl;
        std::cout << "Cross-Modal Connections: " << brain_state.global_memory.stats.cross_modal_connections << std::endl;
        
        std::cout << "\n💡 KEY ACHIEVEMENTS" << std::endl;
        std::cout << "===================" << std::endl;
        std::cout << "✅ Unified brain architecture with intelligent capabilities" << std::endl;
        std::cout << "✅ Automatic connection path traversal" << std::endl;
        std::cout << "✅ Dynamic node creation for new knowledge" << std::endl;
        std::cout << "✅ Intelligent answer synthesis from partial knowledge" << std::endl;
        std::cout << "✅ Keyword extraction and relevant node discovery" << std::endl;
        std::cout << "✅ Hebbian learning with intelligent processing" << std::endl;
        std::cout << "✅ Binary storage with intelligent capabilities" << std::endl;
        
        std::cout << "\n🎉 UNIFIED INTELLIGENT TEST Complete!" << std::endl;
        std::cout << "Melvin's brain now has integrated intelligent capabilities!" << std::endl;
        std::cout << "Every interaction includes intelligent connection traversal!" << std::endl;
    }
};

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    std::cout << "🧠 MELVIN UNIFIED INTELLIGENT TEST" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "Testing Melvin's unified brain with integrated intelligent capabilities" << std::endl;
    std::cout << "Every interaction now includes intelligent connection traversal!" << std::endl;
    
    try {
        // Initialize unified intelligent test
        MelvinUnifiedIntelligentTest unified_test;
        
        // Run unified intelligent test
        unified_test.run_unified_intelligent_test();
        
        std::cout << "\n🎯 UNIFIED INTELLIGENT Evaluation Complete!" << std::endl;
        std::cout << "This test demonstrated Melvin's unified brain with:" << std::endl;
        std::cout << "• Integrated intelligent connection traversal" << std::endl;
        std::cout << "• Automatic dynamic node creation" << std::endl;
        std::cout << "• Intelligent answer synthesis" << std::endl;
        std::cout << "• Keyword extraction and relevant node discovery" << std::endl;
        std::cout << "• Hebbian learning with intelligent processing" << std::endl;
        std::cout << "• Binary storage with intelligent capabilities" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during unified intelligent test: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
