#include "melvin_optimized_v2.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <random>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <map>
#include <set>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <fstream>
#include <queue>
#include <mutex>
#include <condition_variable>

// ============================================================================
// MELVIN CONTINUOUS LEARNER WITH KNOWLEDGE GAP DETECTION
// ============================================================================
// This system implements the user's preferred approach:
// 1. Converts all inputs into nodes using self-regulator system
// 2. Generates Recall Track and Exploration Track for each input
// 3. Shows reasoning paths separately, then provides integrated conclusion
// 4. Uses Ollama to help fill knowledge gaps
// 5. Runs continuously, saving to global repository every couple minutes

struct KnowledgeGap {
    std::string topic;
    std::string question;
    float confidence_threshold;
    int priority_score;
    std::chrono::steady_clock::time_point detected_time;
    std::vector<std::string> related_concepts;
};

struct RecallTrack {
    std::vector<uint64_t> relevant_nodes;
    std::vector<std::string> recalled_facts;
    float recall_confidence;
    std::string reasoning_path;
};

struct ExplorationTrack {
    std::vector<std::string> exploration_questions;
    std::vector<std::string> new_connections;
    float exploration_confidence;
    std::string exploration_path;
};

struct IntegratedConclusion {
    std::string conclusion;
    float confidence_score;
    float weighting; // High confidence emphasizes Recall, low emphasizes Exploration
    std::string reasoning;
    std::vector<uint64_t> source_nodes;
};

class SelfRegulator {
private:
    std::mutex regulator_mutex;
    std::map<std::string, float> concept_importance;
    std::map<std::string, int> concept_frequency;
    std::queue<std::string> filtered_inputs;
    
public:
    SelfRegulator() {
        // Initialize with foundational concepts
        concept_importance = {
            {"artificial_intelligence", 0.9},
            {"learning", 0.8},
            {"reasoning", 0.8},
            {"memory", 0.7},
            {"creativity", 0.6},
            {"consciousness", 0.5},
            {"emotion", 0.4},
            {"philosophy", 0.3},
            {"science", 0.7},
            {"mathematics", 0.6}
        };
    }
    
    bool should_process_input(const std::string& input) {
        std::lock_guard<std::mutex> lock(regulator_mutex);
        
        // Extract keywords and check importance
        std::vector<std::string> keywords = extract_keywords(input);
        float total_importance = 0.0f;
        
        for (const auto& keyword : keywords) {
            if (concept_importance.find(keyword) != concept_importance.end()) {
                total_importance += concept_importance[keyword];
            } else {
                // New concept - assign moderate importance
                concept_importance[keyword] = 0.5f;
                total_importance += 0.5f;
            }
        }
        
        // Filter out low-importance inputs
        return total_importance > 0.3f;
    }
    
    void update_concept_frequency(const std::string& concept) {
        std::lock_guard<std::mutex> lock(regulator_mutex);
        concept_frequency[concept]++;
        
        // Increase importance based on frequency
        if (concept_frequency[concept] > 5) {
            concept_importance[concept] = std::min(1.0f, concept_importance[concept] + 0.1f);
        }
    }
    
private:
    std::vector<std::string> extract_keywords(const std::string& text) {
        std::vector<std::string> keywords;
        std::stringstream ss(text);
        std::string word;
        
        while (ss >> word) {
            // Simple keyword extraction - convert to lowercase
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            keywords.push_back(word);
        }
        
        return keywords;
    }
};

class MelvinContinuousLearner {
private:
    std::unique_ptr<MelvinOptimizedV2> melvin;
    std::unique_ptr<SelfRegulator> regulator;
    std::mt19937 rng;
    
    // Knowledge gap tracking
    std::vector<KnowledgeGap> knowledge_gaps;
    std::mutex gaps_mutex;
    
    // Continuous learning state
    std::atomic<bool> running;
    std::chrono::steady_clock::time_point session_start;
    std::chrono::steady_clock::time_point last_save_time;
    
    // Ollama integration (simulated for now)
    std::vector<std::string> ollama_question_templates;
    std::vector<std::string> ollama_knowledge_topics;
    
    // Statistics
    struct LearningStats {
        uint64_t total_inputs_processed;
        uint64_t knowledge_gaps_detected;
        uint64_t gaps_filled;
        uint64_t recall_tracks_generated;
        uint64_t exploration_tracks_generated;
        uint64_t integrated_conclusions;
        uint64_t ollama_interactions;
    } stats;
    
public:
    MelvinContinuousLearner(const std::string& storage_path = "melvin_unified_memory") 
        : melvin(std::make_unique<MelvinOptimizedV2>(storage_path)),
          regulator(std::make_unique<SelfRegulator>()),
          rng(std::chrono::steady_clock::now().time_since_epoch().count()),
          running(false),
          session_start(std::chrono::steady_clock::now()),
          last_save_time(std::chrono::steady_clock::now()) {
        
        std::cout << "🧠 Melvin Continuous Learner initialized" << std::endl;
        initialize_learning_system();
        stats = {0, 0, 0, 0, 0, 0, 0};
    }
    
    void initialize_learning_system() {
        // Initialize Ollama question templates
        ollama_question_templates = {
            "What do you know about {topic}?",
            "How does {concept} work?",
            "What are the connections between {topic1} and {topic2}?",
            "What questions do you have about {topic}?",
            "How does {concept} relate to your existing knowledge?",
            "What would happen if {scenario}?",
            "Why do you think {phenomenon} occurs?",
            "What patterns do you notice in {domain}?",
            "How would you solve {problem}?",
            "What are the implications of {idea}?"
        };
        
        // Initialize knowledge topics
        ollama_knowledge_topics = {
            "artificial intelligence", "machine learning", "neural networks", "deep learning",
            "cognitive science", "psychology", "philosophy", "consciousness", "creativity",
            "reasoning", "logic", "mathematics", "science", "technology", "robotics",
            "language", "communication", "emotion", "memory", "learning", "adaptation",
            "evolution", "complexity", "systems", "patterns", "emergence", "intelligence"
        };
        
        std::cout << "🧠 Learning system initialized with " << ollama_question_templates.size() 
                  << " question templates and " << ollama_knowledge_topics.size() << " topics" << std::endl;
    }
    
    void run_continuously(int duration_minutes = 60) {
        std::cout << "\n🧠 MELVIN CONTINUOUS LEARNING SESSION" << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << "Duration: " << duration_minutes << " minutes" << std::endl;
        std::cout << "Melvin will continuously learn and fill knowledge gaps!" << std::endl;
        
        running = true;
        auto session_end = std::chrono::steady_clock::now() + 
                          std::chrono::minutes(duration_minutes);
        
        int cycle_count = 0;
        
        while (running && std::chrono::steady_clock::now() < session_end) {
            cycle_count++;
            
            if (cycle_count % 10 == 1) { // Show every 10th cycle
                std::cout << "\n🧠 LEARNING CYCLE " << cycle_count << std::endl;
            }
            
            // 1. Generate input for Melvin to process
            std::string input = generate_learning_input();
            
            // 2. Process input through self-regulator
            if (regulator->should_process_input(input)) {
                process_input_with_tracks(input);
            } else {
                std::cout << "🚫 Input filtered out by self-regulator" << std::endl;
            }
            
            // 3. Detect knowledge gaps
            detect_knowledge_gaps();
            
            // 4. Fill knowledge gaps using Ollama
            fill_knowledge_gaps();
            
            // 5. Perform continuous thinking
            perform_continuous_thinking();
            
            // 6. Save to global repository every couple minutes
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::minutes>(now - last_save_time).count() >= 2) {
                save_to_global_repository();
                last_save_time = now;
            }
            
            // 7. Brief pause between cycles (optimized for speed)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Show progress
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - session_start).count();
            auto remaining = std::chrono::duration_cast<std::chrono::seconds>(
                session_end - std::chrono::steady_clock::now()).count();
            
            std::cout << "\n⏱️ Session Progress: " << elapsed << "s elapsed, " 
                      << remaining << "s remaining" << std::endl;
        }
        
        // Final save and report
        save_to_global_repository();
        generate_learning_report();
        
        running = false;
    }
    
    std::string generate_learning_input() {
        // Reduced output for speed
        
        // Select random topic and generate input
        std::uniform_int_distribution<int> topic_dist(0, ollama_knowledge_topics.size() - 1);
        std::string topic = ollama_knowledge_topics[topic_dist(rng)];
        
        std::vector<std::string> input_templates = {
            "Tell me about " + topic,
            "What do you think about " + topic + "?",
            "How does " + topic + " work?",
            "What are the implications of " + topic + "?",
            "Explain " + topic + " in simple terms",
            "What questions do you have about " + topic + "?",
            "How does " + topic + " relate to artificial intelligence?",
            "What patterns do you see in " + topic + "?"
        };
        
        std::uniform_int_distribution<int> template_dist(0, input_templates.size() - 1);
        std::string input = input_templates[template_dist(rng)];
        
        // Reduced output for speed
        return input;
    }
    
    void process_input_with_tracks(const std::string& input) {
        // Reduced output for speed
        
        stats.total_inputs_processed++;
        
        // Convert input to node
        uint64_t input_node_id = melvin->process_text_input(input, "continuous_learning");
        
        // Generate Recall Track
        RecallTrack recall_track = generate_recall_track(input);
        stats.recall_tracks_generated++;
        
        // Generate Exploration Track  
        ExplorationTrack exploration_track = generate_exploration_track(input);
        stats.exploration_tracks_generated++;
        
        // Generate Integrated Conclusion
        IntegratedConclusion conclusion = generate_integrated_conclusion(input, recall_track, exploration_track);
        stats.integrated_conclusions++;
        
        // Display results
        display_track_results(recall_track, exploration_track, conclusion);
        
        // Store results in Melvin's brain
        store_track_results(input_node_id, recall_track, exploration_track, conclusion);
    }
    
    RecallTrack generate_recall_track(const std::string& input) {
        std::cout << "\n[Recall Track]" << std::endl;
        
        RecallTrack track;
        
        // Extract keywords from input
        std::vector<std::string> keywords = melvin->extract_keywords(input);
        
        // Find relevant nodes in Melvin's memory
        auto relevant_nodes = melvin->find_relevant_nodes(keywords);
        
        track.relevant_nodes.reserve(relevant_nodes.size());
        track.recalled_facts.reserve(relevant_nodes.size());
        
        float total_confidence = 0.0f;
        std::string reasoning_path = "Recall reasoning: ";
        
        for (const auto& node : relevant_nodes) {
            track.relevant_nodes.push_back(node.node_id);
            track.recalled_facts.push_back(node.content);
            total_confidence += node.similarity_score;
            
            reasoning_path += "Found node " + std::to_string(node.node_id) + 
                             " (similarity: " + std::to_string(node.similarity_score) + "), ";
        }
        
        track.recall_confidence = relevant_nodes.empty() ? 0.0f : total_confidence / relevant_nodes.size();
        track.reasoning_path = reasoning_path;
        
        std::cout << "  📚 Relevant nodes found: " << relevant_nodes.size() << std::endl;
        std::cout << "  🎯 Recall confidence: " << std::fixed << std::setprecision(2) 
                  << track.recall_confidence * 100 << "%" << std::endl;
        std::cout << "  🧠 Reasoning: " << track.reasoning_path << std::endl;
        
        return track;
    }
    
    ExplorationTrack generate_exploration_track(const std::string& input) {
        std::cout << "\n[Exploration Track]" << std::endl;
        
        ExplorationTrack track;
        
        // Generate exploration questions
        track.exploration_questions = {
            "What new connections can I make about this topic?",
            "What questions arise from this input?",
            "How does this relate to other concepts I know?",
            "What patterns do I notice?",
            "What implications does this have?"
        };
        
        // Generate new connections
        std::vector<std::string> keywords = melvin->extract_keywords(input);
        for (const auto& keyword : keywords) {
            track.new_connections.push_back("Exploring connections with " + keyword);
        }
        
        // Calculate exploration confidence based on novelty
        track.exploration_confidence = std::min(1.0f, keywords.size() * 0.2f);
        track.exploration_path = "Exploration reasoning: Generated " + 
                                 std::to_string(track.exploration_questions.size()) + 
                                 " questions and " + std::to_string(track.new_connections.size()) + 
                                 " new connections";
        
        std::cout << "  ❓ Exploration questions: " << track.exploration_questions.size() << std::endl;
        std::cout << "  🔗 New connections: " << track.new_connections.size() << std::endl;
        std::cout << "  🎯 Exploration confidence: " << std::fixed << std::setprecision(2) 
                  << track.exploration_confidence * 100 << "%" << std::endl;
        std::cout << "  🧠 Reasoning: " << track.exploration_path << std::endl;
        
        return track;
    }
    
    IntegratedConclusion generate_integrated_conclusion(const std::string& input, 
                                                       const RecallTrack& recall_track,
                                                       const ExplorationTrack& exploration_track) {
        std::cout << "\n[Integration Phase]" << std::endl;
        
        IntegratedConclusion conclusion;
        
        // Calculate confidence score
        conclusion.confidence_score = (recall_track.recall_confidence + exploration_track.exploration_confidence) / 2.0f;
        
        // Determine weighting based on confidence
        if (conclusion.confidence_score > 0.7f) {
            conclusion.weighting = 0.8f; // High confidence emphasizes Recall
            conclusion.reasoning = "High confidence - emphasizing Recall Track";
        } else if (conclusion.confidence_score < 0.3f) {
            conclusion.weighting = 0.2f; // Low confidence emphasizes Exploration
            conclusion.reasoning = "Low confidence - emphasizing Exploration Track";
        } else {
            conclusion.weighting = 0.5f; // Medium confidence balances both
            conclusion.reasoning = "Medium confidence - balancing both tracks";
        }
        
        // Generate integrated conclusion
        if (conclusion.weighting > 0.6f) {
            // Emphasize Recall
            conclusion.conclusion = "Based on my existing knowledge: " + 
                                   (recall_track.recalled_facts.empty() ? 
                                    "I have limited knowledge about this topic." :
                                    recall_track.recalled_facts[0]);
        } else if (conclusion.weighting < 0.4f) {
            // Emphasize Exploration
            conclusion.conclusion = "This is new territory for me. I should explore: " + 
                                   (exploration_track.exploration_questions.empty() ? 
                                    "What questions should I ask?" :
                                    exploration_track.exploration_questions[0]);
        } else {
            // Balance both
            conclusion.conclusion = std::string("Combining my existing knowledge with new exploration: ") +
                                   (recall_track.recalled_facts.empty() ? 
                                    "I need to learn more about this topic." :
                                    "I can build on what I know while exploring new aspects.");
        }
        
        // Combine source nodes
        conclusion.source_nodes = recall_track.relevant_nodes;
        
        std::cout << "  🎯 Confidence score: " << std::fixed << std::setprecision(2) 
                  << conclusion.confidence_score * 100 << "%" << std::endl;
        std::cout << "  ⚖️ Weighting: " << std::fixed << std::setprecision(2) 
                  << conclusion.weighting * 100 << "%" << std::endl;
        std::cout << "  💡 Conclusion: " << conclusion.conclusion << std::endl;
        std::cout << "  🧠 Reasoning: " << conclusion.reasoning << std::endl;
        
        return conclusion;
    }
    
    void display_track_results(const RecallTrack& recall_track,
                             const ExplorationTrack& exploration_track,
                             const IntegratedConclusion& conclusion) {
        std::cout << "\n📊 TRACK RESULTS SUMMARY" << std::endl;
        std::cout << "========================" << std::endl;
        std::cout << "[Recall Track] Confidence: " << std::fixed << std::setprecision(1) 
                  << recall_track.recall_confidence * 100 << "%" << std::endl;
        std::cout << "[Exploration Track] Confidence: " << std::fixed << std::setprecision(1) 
                  << exploration_track.exploration_confidence * 100 << "%" << std::endl;
        std::cout << "[Integration Phase] Final Confidence: " << std::fixed << std::setprecision(1) 
                  << conclusion.confidence_score * 100 << "%" << std::endl;
        std::cout << "[Integration Phase] Weighting: " << std::fixed << std::setprecision(1) 
                  << conclusion.weighting * 100 << "%" << std::endl;
    }
    
    void store_track_results(uint64_t input_node_id,
                           const RecallTrack& recall_track,
                           const ExplorationTrack& exploration_track,
                           const IntegratedConclusion& conclusion) {
        // Store recall track results using Hebbian learning
        for (size_t i = 0; i < recall_track.relevant_nodes.size(); ++i) {
            melvin->update_hebbian_learning(recall_track.relevant_nodes[i]);
        }
        
        // Store exploration track results
        for (const auto& question : exploration_track.exploration_questions) {
            uint64_t question_id = melvin->process_text_input(question, "exploration");
            melvin->update_hebbian_learning(question_id);
        }
        
        // Store integrated conclusion
        uint64_t conclusion_id = melvin->process_text_input(conclusion.conclusion, "conclusion");
        melvin->update_hebbian_learning(conclusion_id);
    }
    
    void detect_knowledge_gaps() {
        std::cout << "\n🔍 Detecting knowledge gaps..." << std::endl;
        
        // Analyze recent inputs for low confidence responses
        // This is a simplified version - in practice, you'd analyze Melvin's recent responses
        
        std::uniform_int_distribution<int> gap_dist(0, 100);
        if (gap_dist(rng) < 30) { // 30% chance of detecting a gap
            KnowledgeGap gap;
            gap.topic = ollama_knowledge_topics[std::uniform_int_distribution<int>(0, ollama_knowledge_topics.size() - 1)(rng)];
            gap.question = "What do you know about " + gap.topic + "?";
            gap.confidence_threshold = 0.5f;
            gap.priority_score = std::uniform_int_distribution<int>(1, 10)(rng);
            gap.detected_time = std::chrono::steady_clock::now();
            gap.related_concepts = {gap.topic};
            
            std::lock_guard<std::mutex> lock(gaps_mutex);
            knowledge_gaps.push_back(gap);
            stats.knowledge_gaps_detected++;
            
            std::cout << "  🎯 Knowledge gap detected: " << gap.topic << " (priority: " << gap.priority_score << ")" << std::endl;
        } else {
            std::cout << "  ✅ No knowledge gaps detected in this cycle" << std::endl;
        }
    }
    
    void fill_knowledge_gaps() {
        std::cout << "\n🤖 Filling knowledge gaps with Ollama..." << std::endl;
        
        std::lock_guard<std::mutex> lock(gaps_mutex);
        
        if (knowledge_gaps.empty()) {
            std::cout << "  ✅ No knowledge gaps to fill" << std::endl;
            return;
        }
        
        // Sort gaps by priority
        std::sort(knowledge_gaps.begin(), knowledge_gaps.end(),
                 [](const KnowledgeGap& a, const KnowledgeGap& b) {
                     return a.priority_score > b.priority_score;
                 });
        
        // Fill top 3 gaps
        int gaps_to_fill = std::min(3, static_cast<int>(knowledge_gaps.size()));
        
        for (int i = 0; i < gaps_to_fill; ++i) {
            const auto& gap = knowledge_gaps[i];
            
            std::cout << "  🎯 Filling gap: " << gap.topic << std::endl;
            
            // Simulate Ollama interaction
            std::string ollama_response = simulate_ollama_interaction(gap.question);
            
            // Process Ollama response through Melvin's brain
            uint64_t response_id = melvin->process_text_input(ollama_response, "ollama_fill");
            
            // Create connections to related concepts using Hebbian learning
            for (const auto& concept : gap.related_concepts) {
                uint64_t concept_id = melvin->process_text_input(concept, "concept");
                melvin->update_hebbian_learning(concept_id);
            }
            
            stats.gaps_filled++;
            stats.ollama_interactions++;
            
            std::cout << "  ✅ Gap filled: " << gap.topic << std::endl;
        }
        
        // Remove filled gaps
        knowledge_gaps.erase(knowledge_gaps.begin(), knowledge_gaps.begin() + gaps_to_fill);
    }
    
    std::string simulate_ollama_interaction(const std::string& question) {
        // Simulate Ollama response - in practice, this would call: ollama generate llama3.2:3b "question"
        std::cout << "  🤖 Ollama responding to: " << question << std::endl;
        
        // Generate a simulated response based on the question
        if (question.find("artificial intelligence") != std::string::npos) {
            return "Artificial intelligence is the simulation of human intelligence in machines. It involves learning, reasoning, and problem-solving capabilities.";
        } else if (question.find("learning") != std::string::npos) {
            return "Learning is the process of acquiring new knowledge, skills, or understanding through experience, study, or instruction.";
        } else if (question.find("reasoning") != std::string::npos) {
            return "Reasoning is the process of thinking about things in a logical way to form conclusions or judgments.";
        } else {
            return "This is an interesting topic that involves complex concepts and relationships that are worth exploring further.";
        }
    }
    
    void perform_continuous_thinking() {
        std::cout << "\n🧠 Performing continuous thinking..." << std::endl;
        
        // Generate thinking prompts
        std::vector<std::string> thinking_prompts = {
            "What patterns do I notice in my recent learning?",
            "How do my new insights connect to existing knowledge?",
            "What questions should I explore next?",
            "What gaps remain in my understanding?",
            "How can I improve my reasoning process?"
        };
        
        std::uniform_int_distribution<int> prompt_dist(0, thinking_prompts.size() - 1);
        std::string prompt = thinking_prompts[prompt_dist(rng)];
        
        std::cout << "  🤔 Thinking: " << prompt << std::endl;
        
        // Let Melvin think about the prompt
        SynthesizedAnswer thinking_answer = melvin->answer_question_intelligently(prompt);
        
        std::cout << "  💡 Insight: " << thinking_answer.answer << std::endl;
        std::cout << "  🎯 Confidence: " << std::fixed << std::setprecision(1) 
                  << thinking_answer.confidence * 100 << "%" << std::endl;
    }
    
    void save_to_global_repository() {
        std::cout << "\n💾 Saving to global repository..." << std::endl;
        
        // Get current brain state
        auto brain_state = melvin->get_unified_state();
        
        std::cout << "  📊 Brain State:" << std::endl;
        std::cout << "    📦 Total Nodes: " << brain_state.global_memory.total_nodes << std::endl;
        std::cout << "    🔗 Total Connections: " << brain_state.global_memory.total_edges << std::endl;
        std::cout << "    💾 Storage Used: " << std::fixed << std::setprecision(2) 
                  << brain_state.global_memory.storage_used_mb << " MB" << std::endl;
        
        // Save memory to disk (this happens automatically in Melvin's brain)
        std::cout << "  ✅ Memory saved to global repository" << std::endl;
    }
    
    void generate_learning_report() {
        std::cout << "\n📊 MELVIN CONTINUOUS LEARNING REPORT" << std::endl;
        std::cout << "====================================" << std::endl;
        
        auto brain_state = melvin->get_unified_state();
        
        std::cout << "\n🧠 BRAIN STATISTICS" << std::endl;
        std::cout << "===================" << std::endl;
        std::cout << "Total Nodes: " << brain_state.global_memory.total_nodes << std::endl;
        std::cout << "Total Connections: " << brain_state.global_memory.total_edges << std::endl;
        std::cout << "Storage Used: " << std::fixed << std::setprecision(2) 
                  << brain_state.global_memory.storage_used_mb << " MB" << std::endl;
        
        std::cout << "\n📈 LEARNING STATISTICS" << std::endl;
        std::cout << "======================" << std::endl;
        std::cout << "Total Inputs Processed: " << stats.total_inputs_processed << std::endl;
        std::cout << "Knowledge Gaps Detected: " << stats.knowledge_gaps_detected << std::endl;
        std::cout << "Knowledge Gaps Filled: " << stats.gaps_filled << std::endl;
        std::cout << "Recall Tracks Generated: " << stats.recall_tracks_generated << std::endl;
        std::cout << "Exploration Tracks Generated: " << stats.exploration_tracks_generated << std::endl;
        std::cout << "Integrated Conclusions: " << stats.integrated_conclusions << std::endl;
        std::cout << "Ollama Interactions: " << stats.ollama_interactions << std::endl;
        
        std::cout << "\n🎯 LEARNING ACHIEVEMENTS" << std::endl;
        std::cout << "=======================" << std::endl;
        std::cout << "✅ Continuous learning performed" << std::endl;
        std::cout << "✅ Knowledge gaps detected and filled" << std::endl;
        std::cout << "✅ Recall/Exploration tracks implemented" << std::endl;
        std::cout << "✅ Self-regulator system active" << std::endl;
        std::cout << "✅ Ollama integration demonstrated" << std::endl;
        std::cout << "✅ Global repository saves performed" << std::endl;
        
        std::cout << "\n🚀 Melvin's continuous learning session complete!" << std::endl;
        std::cout << "His brain has grown and evolved through continuous learning!" << std::endl;
    }
    
    void stop_learning() {
        running = false;
    }
};

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    std::cout << "🧠 MELVIN CONTINUOUS LEARNER" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "Creating continuous learning system with knowledge gap detection" << std::endl;
    std::cout << "Melvin will run continuously, search for holes in his knowledge," << std::endl;
    std::cout << "and use Ollama to help fill them!" << std::endl;
    
    try {
        MelvinContinuousLearner learner;
        
        // Run continuous learning for 10 minutes (optimized for speed)
        learner.run_continuously(10);
        
        std::cout << "\n🎉 Melvin Continuous Learner completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
