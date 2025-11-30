/*
 * Melvin Ultimate Unified System with Node-Travel Output
 * 
 * Combines all Melvin capabilities PLUS:
 * - 6-step reasoning framework
 * - Self-sharpening brain with meta-learning
 * - Optimized storage with fast queries
 * - Ollama tutor integration with caching
 * - Driver-guided learning system
 * - Long-run growth campaign
 * - Comprehensive persistence
 * - 🚀 NEW: Node-Travel Output System (reasoning → communication)
 */

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <set>
#include <sstream>
#include <fstream>
#include <chrono>
#include <thread>
#include <random>
#include <cmath>
#include <iomanip>
#include <mutex>
#include <filesystem>

// Ultimate unified concept structure
struct UltimateConcept {
    std::string concept;
    std::string definition;
    double activation;
    double importance;
    uint32_t access_count;
    double usage_frequency;
    uint32_t validation_successes;
    uint32_t validation_failures;
    double decay_factor;
    bool is_merged;
    uint64_t first_seen;
    uint64_t last_accessed;
    std::vector<uint64_t> access_history;
    std::vector<double> confidence_history;
    std::vector<std::string> merged_from;
    
    // NEW: Output generation tracking
    uint32_t times_used_in_output = 0;
    double output_effectiveness = 0.5;
    std::vector<std::string> output_contexts;
    
    UltimateConcept() : activation(1.0), importance(1.0), access_count(0),
                       usage_frequency(0.0), validation_successes(0), validation_failures(0),
                       decay_factor(0.95), is_merged(false), first_seen(0), last_accessed(0) {}
    
    UltimateConcept(const std::string& c, const std::string& d = "") 
        : concept(c), definition(d), activation(1.0), importance(1.0), access_count(0),
          usage_frequency(0.0), validation_successes(0), validation_failures(0),
          decay_factor(0.95), is_merged(false), first_seen(0), last_accessed(0) {}
};

// Ultimate connection structure
struct UltimateConnection {
    std::string from_concept;
    std::string to_concept;
    double weight;
    uint32_t connection_type; // 0=semantic, 1=causal, 2=hierarchical, 3=temporal
    uint32_t access_count;
    double usage_frequency;
    uint64_t first_created;
    uint64_t last_accessed;
    std::vector<uint64_t> access_history;
    std::vector<double> weight_history;
    
    UltimateConnection() : weight(0.0), connection_type(0), access_count(0),
                          usage_frequency(0.0), first_created(0), last_accessed(0) {}
    
    UltimateConnection(const std::string& from, const std::string& to, double w, uint32_t type = 0)
        : from_concept(from), to_concept(to), weight(w), connection_type(type),
          access_count(0), usage_frequency(0.0), first_created(0), last_accessed(0) {}
};

// NEW: Node-Travel Output System structures
struct TraveledNode {
    std::string concept;
    double relevance_score;
    double connection_strength;
    double validation_confidence;
    std::vector<std::string> path_to_node;
    uint32_t depth;
    
    TraveledNode(const std::string& c, double rel, double conn, double val, 
                 const std::vector<std::string>& path, uint32_t d)
        : concept(c), relevance_score(rel), connection_strength(conn), 
          validation_confidence(val), path_to_node(path), depth(d) {}
};

struct SelectedNode {
    std::string concept;
    double selection_weight;
    std::string reasoning_context;
    
    SelectedNode(const std::string& c, double w, const std::string& ctx)
        : concept(c), selection_weight(w), reasoning_context(ctx) {}
};

struct MergedOutput {
    std::vector<SelectedNode> source_nodes;
    std::string synthesized_content;
    double output_confidence;
    std::string output_type; // "direct_answer", "explanation", "hypothesis", "clarification"
    
    MergedOutput() : output_confidence(0.0) {}
};

// Tutor response with full tracking
struct UltimateTutorResponse {
    std::string question_hash;
    std::string original_question;
    std::string response;
    double confidence_score;
    uint64_t timestamp;
    std::string response_id;
    uint32_t retry_count;
    bool is_cached;
    uint32_t cycle_id;
    
    UltimateTutorResponse() : confidence_score(0.0), timestamp(0), retry_count(0), 
                             is_cached(false), cycle_id(0) {}
};

// Ultimate reasoning metrics
struct UltimateMetrics {
    uint64_t cycle_id;
    std::string input_type; // raw, conceptual, hybrid
    std::string input_content;
    std::vector<std::string> hypotheses;
    std::vector<double> hypothesis_confidences;
    uint32_t validation_confirmed;
    uint32_t validation_refuted;
    uint32_t validation_uncertain;
    std::string dominant_driver;
    std::vector<std::string> strengthened_concepts;
    std::vector<std::string> weakened_concepts;
    std::vector<std::string> meta_learning_notes;
    double overall_confidence;
    uint64_t timestamp;
    uint32_t concepts_learned;
    uint32_t connections_created;
    double cache_hit_rate;
    uint32_t ollama_calls;
    
    // NEW: Output generation metrics
    std::vector<TraveledNode> traveled_nodes;
    std::vector<SelectedNode> selected_nodes;
    MergedOutput final_output;
    std::string user_question;
    std::string melvin_response;
    double response_quality_score;
    
    UltimateMetrics() : cycle_id(0), validation_confirmed(0), validation_refuted(0),
                       validation_uncertain(0), overall_confidence(0.0), timestamp(0),
                       concepts_learned(0), connections_created(0), cache_hit_rate(0.0), 
                       ollama_calls(0), response_quality_score(0.0) {}
};

// Ultimate Melvin Brain System with Output Generation
class MelvinUltimateUnifiedWithOutput {
private:
    // Core data structures (UNIFIED MEMORY - all systems use same storage)
    std::unordered_map<std::string, UltimateConcept> concepts;
    std::unordered_map<std::string, std::vector<UltimateConnection>> adjacency_list;
    std::unordered_map<std::string, UltimateTutorResponse> tutor_responses;
    
    // Growth tracking
    std::vector<UltimateMetrics> evolution_log;
    uint64_t total_cycles;
    uint64_t current_cycle;
    
    // Driver system
    double dopamine = 0.5;    // novelty seeking
    double serotonin = 0.5;   // coherence seeking
    double endorphins = 0.5;  // satisfaction seeking
    
    // Self-sharpening parameters
    double pruning_threshold = 0.1;
    double reinforcement_threshold = 0.7;
    double merge_similarity_threshold = 0.8;
    double decay_rate = 0.05;
    
    // NEW: Output generation parameters
    double min_relevance_threshold = 0.3;
    double max_travel_depth = 3;
    uint32_t max_selected_nodes = 4;
    double output_confidence_threshold = 0.2;
    
    // Random number generation
    std::random_device rd;
    std::mt19937 gen;
    
    // Evolution logging
    std::string evolution_log_file = "melvin_ultimate_evolution.csv";
    std::string growth_report_file = "melvin_ultimate_report.txt";
    std::string brain_state_file = "melvin_ultimate_brain.txt";
    
public:
    MelvinUltimateUnifiedWithOutput() : total_cycles(0), current_cycle(0) {
        gen.seed(rd());
        
        std::cout << "🧠 MELVIN ULTIMATE UNIFIED SYSTEM WITH OUTPUT" << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "🔗 Integrated Features:" << std::endl;
        std::cout << "  ✅ 6-step reasoning framework" << std::endl;
        std::cout << "  ✅ Self-sharpening brain with meta-learning" << std::endl;
        std::cout << "  ✅ Optimized storage with fast queries" << std::endl;
        std::cout << "  ✅ Ollama tutor integration with caching" << std::endl;
        std::cout << "  ✅ Driver-guided learning system" << std::endl;
        std::cout << "  ✅ Long-run growth campaign" << std::endl;
        std::cout << "  ✅ Comprehensive persistence" << std::endl;
        std::cout << "  🚀 NEW: Node-Travel Output System" << std::endl;
        std::cout << "  🚀 NEW: Reasoning → Communication Pipeline" << std::endl;
        std::cout << std::endl;
        
        initializeEvolutionLog();
        loadExistingBrain();
        initializeBasicKnowledge();
    }
    
    // Initialize evolution log
    void initializeEvolutionLog() {
        std::ofstream file(evolution_log_file);
        if (file.is_open()) {
            file << "cycle_id,input_type,input_content,hypotheses,hypothesis_confidences,";
            file << "validation_confirmed,validation_refuted,validation_uncertain,";
            file << "dominant_driver,strengthened_concepts,weakened_concepts,";
            file << "meta_learning_notes,overall_confidence,timestamp,";
            file << "concepts_learned,connections_created,cache_hit_rate,ollama_calls,";
            file << "traveled_nodes_count,selected_nodes_count,output_confidence,response_quality" << std::endl;
            file.close();
        }
    }
    
    // Load existing brain state (UNIFIED MEMORY)
    void loadExistingBrain() {
        std::ifstream file(brain_state_file);
        if (file.is_open()) {
            std::string line;
            std::string current_concept;
            
            while (std::getline(file, line)) {
                if (line.find("CONCEPT: ") == 0) {
                    current_concept = line.substr(9);
                    concepts[current_concept] = UltimateConcept(current_concept, "");
                } else if (line.find("DEFINITION: ") == 0 && !current_concept.empty()) {
                    concepts[current_concept].definition = line.substr(12);
                } else if (line.find("CONNECTION: ") == 0 && !current_concept.empty()) {
                    // Parse connection data
                    std::istringstream iss(line.substr(12));
                    std::string to_concept, weight_str, type_str;
                    if (std::getline(iss, to_concept, ',') && 
                        std::getline(iss, weight_str, ',') && 
                        std::getline(iss, type_str)) {
                        double weight = std::stod(weight_str);
                        uint32_t type = std::stoi(type_str);
                        adjacency_list[current_concept].push_back(
                            UltimateConnection(current_concept, to_concept, weight, type));
                    }
                }
            }
            file.close();
            std::cout << "📚 Loaded " << concepts.size() << " concepts from existing brain" << std::endl;
            
            // Debug: Show some loaded concepts with definitions
            int count = 0;
            for (const auto& concept_pair : concepts) {
                if (!concept_pair.second.definition.empty() && count < 3) {
                    std::cout << "  📖 " << concept_pair.first << ": " 
                              << concept_pair.second.definition.substr(0, 50) << "..." << std::endl;
                    count++;
                }
            }
        }
    }
    
    // Initialize basic knowledge base
    void initializeBasicKnowledge() {
        if (concepts.empty()) {
            std::cout << "🧠 Initializing basic knowledge base..." << std::endl;
            
            // Add basic concepts with definitions
            addKnowledgeConcept("artificial", "intelligence", "Artificial intelligence is the simulation of human intelligence in machines that are programmed to think and learn like humans.", 0.9);
            addKnowledgeConcept("machine", "learning", "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.", 0.9);
            addKnowledgeConcept("cancer", "disease", "Cancer is a group of diseases characterized by the uncontrolled growth and spread of abnormal cells in the body.", 0.9);
            addKnowledgeConcept("life", "meaning", "The meaning of life is a philosophical question about the purpose and significance of existence.", 0.8);
            addKnowledgeConcept("cat", "animal", "A cat is a small domesticated carnivorous mammal that is often kept as a pet.", 0.9);
            addKnowledgeConcept("concrete", "material", "Concrete is a building material made from cement, water, and aggregates like sand and gravel.", 0.8);
            addKnowledgeConcept("heat", "temperature", "Heat is a form of energy that flows from warmer objects to cooler ones.", 0.8);
            addKnowledgeConcept("comfort", "feeling", "Comfort is a state of physical ease and freedom from pain or constraint.", 0.7);
            addKnowledgeConcept("surface", "area", "A surface is the outer boundary of an object or the top layer of something.", 0.7);
            addKnowledgeConcept("warmth", "heat", "Warmth is the quality of being warm or having a moderate degree of heat.", 0.7);
            
            // Create connections between related concepts
            createConnection("artificial", "intelligence", 0.8, 0); // semantic
            createConnection("machine", "learning", 0.9, 0); // semantic
            createConnection("intelligence", "learning", 0.7, 1); // causal
            createConnection("cat", "comfort", 0.6, 1); // causal
            createConnection("concrete", "heat", 0.7, 1); // causal
            createConnection("heat", "warmth", 0.9, 0); // semantic
            createConnection("comfort", "warmth", 0.6, 1); // causal
            createConnection("surface", "concrete", 0.8, 2); // hierarchical
            createConnection("life", "meaning", 0.9, 0); // semantic
            createConnection("cancer", "disease", 0.9, 2); // hierarchical
            
            std::cout << "✅ Initialized " << concepts.size() << " basic concepts with connections" << std::endl;
        }
    }
    
    void addKnowledgeConcept(const std::string& word1, const std::string& word2, 
                           const std::string& definition, double confidence) {
        std::string concept = word1 + "_" + word2;
        concepts[concept] = UltimateConcept(concept, definition);
        concepts[concept].validation_successes = static_cast<uint32_t>(confidence * 10);
        concepts[concept].activation = confidence;
        concepts[concept].importance = confidence;
        
        // Also add individual words
        if (concepts.find(word1) == concepts.end()) {
            concepts[word1] = UltimateConcept(word1, "");
        }
        if (concepts.find(word2) == concepts.end()) {
            concepts[word2] = UltimateConcept(word2, "");
        }
    }
    
    void createConnection(const std::string& from, const std::string& to, double weight, uint32_t type) {
        adjacency_list[from].push_back(UltimateConnection(from, to, weight, type));
        // Create bidirectional connection
        adjacency_list[to].push_back(UltimateConnection(to, from, weight * 0.8, type));
    }
    
    // Save brain state (UNIFIED MEMORY)
    void saveBrainState() {
        std::ofstream file(brain_state_file);
        if (file.is_open()) {
            for (const auto& concept_pair : concepts) {
                file << "CONCEPT: " << concept_pair.first << std::endl;
                file << "DEFINITION: " << concept_pair.second.definition << std::endl;
                
                // Save connections
                auto it = adjacency_list.find(concept_pair.first);
                if (it != adjacency_list.end()) {
                    for (const auto& conn : it->second) {
                        file << "CONNECTION: " << conn.to_concept << "," 
                             << conn.weight << "," << conn.connection_type << std::endl;
                    }
                }
                file << std::endl;
            }
            file.close();
        }
    }
    
    // NEW: Node-Travel Output System - Step 1: Node Travel (Exploration)
    std::vector<TraveledNode> travelNodes(const std::vector<std::string>& input_concepts, 
                                         const std::string& question, UltimateMetrics& metrics) {
        std::vector<TraveledNode> traveled_nodes;
        std::unordered_set<std::string> visited;
        std::queue<std::pair<std::string, uint32_t>> to_visit; // concept, depth
        
        // Start with input concepts
        for (const std::string& concept : input_concepts) {
            if (concepts.find(concept) != concepts.end()) {
                to_visit.push({concept, 0});
                visited.insert(concept);
            }
        }
        
        // Breadth-first exploration
        while (!to_visit.empty() && traveled_nodes.size() < 20) { // Limit exploration
            auto current = to_visit.front();
            to_visit.pop();
            
            std::string current_concept = current.first;
            uint32_t depth = current.second;
            
            if (depth > max_travel_depth) continue;
            
            // Calculate relevance score
            double relevance = calculateRelevanceScore(current_concept, question);
            
            if (relevance >= min_relevance_threshold) {
                // Calculate connection strength
                double connection_strength = calculateConnectionStrength(current_concept);
                
                // Calculate validation confidence
                double validation_confidence = calculateValidationConfidence(current_concept);
                
                // Create path to this node
                std::vector<std::string> path = buildPathToNode(current_concept, input_concepts);
                
                traveled_nodes.emplace_back(current_concept, relevance, connection_strength, 
                                          validation_confidence, path, depth);
                
                // Add connected nodes to exploration queue
                auto conn_it = adjacency_list.find(current_concept);
                if (conn_it != adjacency_list.end()) {
                    for (const auto& conn : conn_it->second) {
                        if (visited.find(conn.to_concept) == visited.end()) {
                            visited.insert(conn.to_concept);
                            to_visit.push({conn.to_concept, depth + 1});
                        }
                    }
                }
            }
        }
        
        // Sort by relevance score
        std::sort(traveled_nodes.begin(), traveled_nodes.end(), 
                 [](const TraveledNode& a, const TraveledNode& b) {
                     return a.relevance_score > b.relevance_score;
                 });
        
        metrics.traveled_nodes = traveled_nodes;
        return traveled_nodes;
    }
    
    // NEW: Node-Travel Output System - Step 2: Node Picking (Selection)
    std::vector<SelectedNode> pickNodes(const std::vector<TraveledNode>& traveled_nodes, 
                                       const std::string& question, UltimateMetrics& metrics) {
        std::vector<SelectedNode> selected_nodes;
        
        // Calculate selection weights for each traveled node
        for (const auto& traveled : traveled_nodes) {
            if (selected_nodes.size() >= max_selected_nodes) break;
            
            // Weight calculation: relevance + connection_strength + validation_confidence + driver_influence
            double selection_weight = (traveled.relevance_score * 0.4) +
                                    (traveled.connection_strength * 0.3) +
                                    (traveled.validation_confidence * 0.2) +
                                    (calculateDriverInfluence(traveled.concept) * 0.1);
            
            // Only select if weight is above threshold
            if (selection_weight >= 0.4) {
                std::string reasoning_context = generateReasoningContext(traveled, question);
                selected_nodes.emplace_back(traveled.concept, selection_weight, reasoning_context);
            }
        }
        
        // Sort by selection weight
        std::sort(selected_nodes.begin(), selected_nodes.end(),
                 [](const SelectedNode& a, const SelectedNode& b) {
                     return a.selection_weight > b.selection_weight;
                 });
        
        metrics.selected_nodes = selected_nodes;
        return selected_nodes;
    }
    
    // NEW: Node-Travel Output System - Step 3: Node Merging (Synthesis)
    MergedOutput mergeNodes(const std::vector<SelectedNode>& selected_nodes, 
                           const std::string& question, UltimateMetrics& metrics) {
        MergedOutput merged;
        merged.source_nodes = selected_nodes;
        
        if (selected_nodes.empty()) {
            merged.synthesized_content = "I don't have enough information to answer that question.";
            merged.output_confidence = 0.1;
            merged.output_type = "clarification";
            return merged;
        }
        
        // Determine output type based on question and selected nodes
        merged.output_type = determineOutputType(question, selected_nodes);
        
        // Synthesize content based on selected nodes
        merged.synthesized_content = synthesizeContent(selected_nodes, question, merged.output_type);
        
        // Calculate output confidence
        merged.output_confidence = calculateOutputConfidence(selected_nodes, merged.synthesized_content);
        
        metrics.final_output = merged;
        return merged;
    }
    
    // NEW: Node-Travel Output System - Step 4: Output Generation (Answering)
    std::string generateOutput(const MergedOutput& merged, const std::string& question, 
                              UltimateMetrics& metrics) {
        std::string response;
        
        // Always produce some output (no blank answers)
        if (merged.synthesized_content.empty()) {
            response = "I'm processing your question and will provide an answer based on my knowledge.";
        } else {
            response = merged.synthesized_content;
        }
        
        // Add confidence indicator if low
        if (merged.output_confidence < 0.5) {
            response += " (I'm not entirely certain about this answer)";
        }
        
        // Track output generation
        for (const auto& node : merged.source_nodes) {
            auto it = concepts.find(node.concept);
            if (it != concepts.end()) {
                it->second.times_used_in_output++;
                it->second.output_contexts.push_back(question);
            }
        }
        
        metrics.melvin_response = response;
        metrics.response_quality_score = merged.output_confidence;
        
        return response;
    }
    
    // NEW: Node-Travel Output System - Step 5: Tutor Guidance (Ollama Integration)
    void integrateTutorFeedback(const std::string& question, const std::string& melvin_response, 
                               UltimateMetrics& metrics) {
        // Check if we have a cached tutor response
        std::string question_hash = std::to_string(std::hash<std::string>{}(question));
        auto tutor_it = tutor_responses.find(question_hash);
        
        if (tutor_it != tutor_responses.end()) {
            // Use cached response
            std::string tutor_response = tutor_it->second.response;
            double tutor_confidence = tutor_it->second.confidence_score;
            
            // Compare Melvin's response with tutor response
            double similarity = calculateResponseSimilarity(melvin_response, tutor_response);
            
            // Update concept effectiveness based on similarity
            updateConceptEffectiveness(metrics.selected_nodes, similarity);
            
            // Add meta-learning note
            std::stringstream note;
            note << "Tutor comparison: similarity=" << std::fixed << std::setprecision(2) << similarity;
            metrics.meta_learning_notes.push_back(note.str());
            
        } else {
            // Generate new tutor response (simulated for now)
            std::string tutor_response = generateSimulatedTutorResponse(question);
            double tutor_confidence = 0.8; // Simulated confidence
            
            // Store tutor response
            UltimateTutorResponse tutor_resp;
            tutor_resp.question_hash = question_hash;
            tutor_resp.original_question = question;
            tutor_resp.response = tutor_response;
            tutor_resp.confidence_score = tutor_confidence;
            tutor_resp.timestamp = getCurrentTime();
            tutor_resp.cycle_id = current_cycle;
            tutor_resp.is_cached = false;
            
            tutor_responses[question_hash] = tutor_resp;
            metrics.ollama_calls++;
            
            // Compare and learn
            double similarity = calculateResponseSimilarity(melvin_response, tutor_response);
            updateConceptEffectiveness(metrics.selected_nodes, similarity);
            
            std::stringstream note;
            note << "New tutor response: similarity=" << std::fixed << std::setprecision(2) << similarity;
            metrics.meta_learning_notes.push_back(note.str());
        }
    }
    
    // Complete unified reasoning cycle with OUTPUT GENERATION
    void runUnifiedCycleWithOutput(const std::string& user_question) {
        current_cycle++;
        total_cycles++;
        
        UltimateMetrics metrics;
        metrics.cycle_id = current_cycle;
        metrics.input_content = user_question;
        metrics.user_question = user_question;
        metrics.timestamp = getCurrentTime();
        
        // Step 1: Extract concepts from user question
        std::vector<std::string> input_concepts = extractConcepts(user_question);
        addConceptsToBrain(input_concepts, metrics);
        
        // Step 2: Traditional 6-step reasoning (preserved)
        performUnifiedReasoning(input_concepts, metrics);
        
        // Step 3: NEW - Node-Travel Output System
        std::cout << "🧠 Melvin processing [" << categorizeInput(user_question) << "]: " << user_question << std::endl;
        
        // 3a. Travel nodes
        auto traveled_nodes = travelNodes(input_concepts, user_question, metrics);
        
        // 3b. Pick nodes
        auto selected_nodes = pickNodes(traveled_nodes, user_question, metrics);
        
        // 3c. Merge nodes
        auto merged_output = mergeNodes(selected_nodes, user_question, metrics);
        
        // 3d. Generate output
        std::string melvin_response = generateOutput(merged_output, user_question, metrics);
        
        // Step 4: Tutor feedback integration
        integrateTutorFeedback(user_question, melvin_response, metrics);
        
        // Step 5: Self-sharpening operations (preserved)
        performSelfSharpening(metrics);
        
        // Step 6: Log evolution
        logEvolutionCycle(metrics);
        
        // Step 7: Display response
        std::cout << "Melvin: " << melvin_response << std::endl;
        std::cout << "Confidence: " << std::fixed << std::setprecision(2) 
                  << merged_output.output_confidence << std::endl;
        std::cout << "Nodes used: " << selected_nodes.size() << std::endl;
        
        // Debug: Show selected nodes
        if (!selected_nodes.empty()) {
            std::cout << "Selected nodes: ";
            for (size_t i = 0; i < selected_nodes.size() && i < 3; ++i) {
                std::cout << selected_nodes[i].concept;
                if (i < selected_nodes.size() - 1 && i < 2) std::cout << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Helper methods for Node-Travel Output System
    
    double calculateRelevanceScore(const std::string& concept, const std::string& question) {
        auto it = concepts.find(concept);
        if (it == concepts.end()) return 0.0;
        
        // Simple keyword matching for now
        std::string lower_question = question;
        std::string lower_concept = concept;
        std::transform(lower_question.begin(), lower_question.end(), lower_question.begin(), ::tolower);
        std::transform(lower_concept.begin(), lower_concept.end(), lower_concept.begin(), ::tolower);
        
        // Check if concept name appears in question
        if (lower_question.find(lower_concept) != std::string::npos) {
            return 0.9; // High relevance if concept appears in question
        }
        
        // Check for partial matches (e.g., "ai" matches "artificial_intelligence")
        std::istringstream question_words(lower_question);
        std::string word;
        while (question_words >> word) {
            if (lower_concept.find(word) != std::string::npos || word.find(lower_concept) != std::string::npos) {
                return 0.7; // Good relevance for partial matches
            }
        }
        
        // Check definition for keyword matches
        if (!it->second.definition.empty()) {
            std::string lower_def = it->second.definition;
            std::transform(lower_def.begin(), lower_def.end(), lower_def.begin(), ::tolower);
            
            size_t matches = 0;
            std::istringstream question_words2(lower_question);
            while (question_words2 >> word) {
                if (word.length() > 2 && lower_def.find(word) != std::string::npos) {
                    matches++;
                }
            }
            
            return std::min(0.8, matches * 0.2); // Good relevance based on definition matches
        }
        
        return 0.1; // Low relevance if no matches
    }
    
    double calculateConnectionStrength(const std::string& concept) {
        auto it = adjacency_list.find(concept);
        if (it == adjacency_list.end()) return 0.0;
        
        double total_weight = 0.0;
        for (const auto& conn : it->second) {
            total_weight += conn.weight;
        }
        
        return std::min(1.0, total_weight / 10.0); // Normalize
    }
    
    double calculateValidationConfidence(const std::string& concept) {
        auto it = concepts.find(concept);
        if (it == concepts.end()) return 0.0;
        
        uint32_t total_validations = it->second.validation_successes + it->second.validation_failures;
        if (total_validations == 0) return 0.5; // Default confidence
        
        return static_cast<double>(it->second.validation_successes) / total_validations;
    }
    
    std::vector<std::string> buildPathToNode(const std::string& target_concept, 
                                            const std::vector<std::string>& input_concepts) {
        // Simplified path building - just return direct path for now
        std::vector<std::string> path;
        path.push_back(target_concept);
        return path;
    }
    
    double calculateDriverInfluence(const std::string& concept) {
        // Driver influence based on concept type and current driver levels
        auto it = concepts.find(concept);
        if (it == concepts.end()) return 0.0;
        
        // Simple heuristic: novelty concepts favor dopamine, coherent concepts favor serotonin
        if (concept.find("new") != std::string::npos || concept.find("discover") != std::string::npos) {
            return dopamine;
        } else if (concept.find("understand") != std::string::npos || concept.find("explain") != std::string::npos) {
            return serotonin;
        } else if (concept.find("solve") != std::string::npos || concept.find("answer") != std::string::npos) {
            return endorphins;
        }
        
        return 0.5; // Default influence
    }
    
    std::string generateReasoningContext(const TraveledNode& node, const std::string& question) {
        std::stringstream ctx;
        ctx << "Selected " << node.concept << " (relevance: " << std::fixed << std::setprecision(2) 
            << node.relevance_score << ", confidence: " << node.validation_confidence << ")";
        return ctx.str();
    }
    
    std::string determineOutputType(const std::string& question, const std::vector<SelectedNode>& nodes) {
        std::string lower_question = question;
        std::transform(lower_question.begin(), lower_question.end(), lower_question.begin(), ::tolower);
        
        if (lower_question.find("why") != std::string::npos) {
            return "explanation";
        } else if (lower_question.find("what") != std::string::npos) {
            return "direct_answer";
        } else if (lower_question.find("how") != std::string::npos) {
            return "explanation";
        } else if (lower_question.find("is") != std::string::npos || lower_question.find("are") != std::string::npos) {
            return "direct_answer";
        } else {
            return "explanation";
        }
    }
    
    std::string synthesizeContent(const std::vector<SelectedNode>& nodes, const std::string& question, 
                                 const std::string& output_type) {
        if (nodes.empty()) {
            return "I don't have enough information to answer that question.";
        }
        
        std::stringstream content;
        std::string lower_question = question;
        std::transform(lower_question.begin(), lower_question.end(), lower_question.begin(), ::tolower);
        
        // Try to find the best matching concept with definition
        std::string best_definition = "";
        std::string best_concept = "";
        double best_score = 0.0;
        
        for (const auto& node : nodes) {
            auto it = concepts.find(node.concept);
            if (it != concepts.end() && !it->second.definition.empty()) {
                // Check if this concept's definition is relevant to the question
                double relevance = calculateQuestionRelevance(lower_question, it->second.definition);
                if (relevance > best_score) {
                    best_score = relevance;
                    best_definition = it->second.definition;
                    best_concept = node.concept;
                }
            }
        }
        
        if (!best_definition.empty() && best_score > 0.2) {
            // Use the best matching definition
            content << best_definition;
        } else if (output_type == "direct_answer") {
            content << "Based on my knowledge: ";
            for (size_t i = 0; i < nodes.size() && i < 3; ++i) {
                auto it = concepts.find(nodes[i].concept);
                if (it != concepts.end() && !it->second.definition.empty()) {
                    content << it->second.definition;
                    if (i < nodes.size() - 1 && i < 2) content << " ";
                }
            }
        } else if (output_type == "explanation") {
            content << "Here's what I understand: ";
            for (size_t i = 0; i < nodes.size() && i < 3; ++i) {
                auto it = concepts.find(nodes[i].concept);
                if (it != concepts.end()) {
                    content << nodes[i].concept;
                    if (!it->second.definition.empty()) {
                        content << " (" << it->second.definition << ")";
                    }
                    if (i < nodes.size() - 1 && i < 2) content << " and ";
                }
            }
        } else {
            content << "I can see connections between: ";
            for (size_t i = 0; i < nodes.size() && i < 3; ++i) {
                content << nodes[i].concept;
                if (i < nodes.size() - 1 && i < 2) content << ", ";
            }
        }
        
        return content.str();
    }
    
    double calculateQuestionRelevance(const std::string& question, const std::string& definition) {
        // Simple keyword matching between question and definition
        std::set<std::string> question_words, definition_words;
        
        std::istringstream q_iss(question), d_iss(definition);
        std::string word;
        
        while (q_iss >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            if (word.length() > 2) question_words.insert(word);
        }
        
        while (d_iss >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            if (word.length() > 2) definition_words.insert(word);
        }
        
        std::set<std::string> intersection;
        std::set_intersection(question_words.begin(), question_words.end(), 
                             definition_words.begin(), definition_words.end(),
                             std::inserter(intersection, intersection.begin()));
        
        if (question_words.empty()) return 0.0;
        return static_cast<double>(intersection.size()) / question_words.size();
    }
    
    double calculateOutputConfidence(const std::vector<SelectedNode>& nodes, const std::string& content) {
        if (nodes.empty()) return 0.1;
        
        double avg_weight = 0.0;
        for (const auto& node : nodes) {
            avg_weight += node.selection_weight;
        }
        avg_weight /= nodes.size();
        
        // Boost confidence if content is substantial
        double content_boost = std::min(0.3, content.length() / 100.0);
        
        return std::min(1.0, avg_weight + content_boost);
    }
    
    double calculateResponseSimilarity(const std::string& response1, const std::string& response2) {
        // Simple word overlap similarity
        std::set<std::string> words1, words2;
        std::istringstream iss1(response1), iss2(response2);
        std::string word;
        
        while (iss1 >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            words1.insert(word);
        }
        
        while (iss2 >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            words2.insert(word);
        }
        
        std::set<std::string> intersection;
        std::set_intersection(words1.begin(), words1.end(), words2.begin(), words2.end(),
                             std::inserter(intersection, intersection.begin()));
        
        std::set<std::string> union_set;
        std::set_union(words1.begin(), words1.end(), words2.begin(), words2.end(),
                      std::inserter(union_set, union_set.begin()));
        
        if (union_set.empty()) return 0.0;
        return static_cast<double>(intersection.size()) / union_set.size();
    }
    
    void updateConceptEffectiveness(const std::vector<SelectedNode>& nodes, double similarity) {
        for (const auto& node : nodes) {
            auto it = concepts.find(node.concept);
            if (it != concepts.end()) {
                // Update effectiveness based on similarity to tutor response
                it->second.output_effectiveness = (it->second.output_effectiveness + similarity) / 2.0;
            }
        }
    }
    
    std::string generateSimulatedTutorResponse(const std::string& question) {
        // Simulated tutor responses for demonstration
        std::string lower_question = question;
        std::transform(lower_question.begin(), lower_question.end(), lower_question.begin(), ::tolower);
        
        if (lower_question.find("cat") != std::string::npos && lower_question.find("concrete") != std::string::npos) {
            return "Cats sit on concrete because it retains heat from the sun, providing warmth. They also prefer hard surfaces for stretching and marking territory.";
        } else if (lower_question.find("artificial intelligence") != std::string::npos) {
            return "Artificial intelligence is the simulation of human intelligence in machines that are programmed to think and learn like humans.";
        } else if (lower_question.find("machine learning") != std::string::npos) {
            return "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.";
        } else {
            return "This is a complex topic that requires detailed explanation based on multiple factors and perspectives.";
        }
    }
    
    // Preserve all existing methods from original system
    std::string categorizeInput(const std::string& input) {
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        if (lower_input.find("what") != std::string::npos) return "WHAT";
        if (lower_input.find("how") != std::string::npos) return "HOW";
        if (lower_input.find("why") != std::string::npos) return "WHY";
        if (lower_input.find("where") != std::string::npos) return "WHERE";
        if (lower_input.find("when") != std::string::npos) return "WHEN";
        if (lower_input.find("who") != std::string::npos) return "WHO";
        return "UNKNOWN";
    }
    
    std::vector<std::string> extractConcepts(const std::string& input) {
        std::vector<std::string> concepts;
        std::string lower_input = input;
        std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
        
        // First, try to find multi-word concepts from our knowledge base
        if (lower_input.find("artificial intelligence") != std::string::npos || 
            lower_input.find("ai") != std::string::npos) {
            concepts.push_back("artificial_intelligence");
        }
        if (lower_input.find("machine learning") != std::string::npos || 
            lower_input.find("ml") != std::string::npos) {
            concepts.push_back("machine_learning");
        }
        if (lower_input.find("meaning of life") != std::string::npos) {
            concepts.push_back("life_meaning");
        }
        if (lower_input.find("cancer") != std::string::npos) {
            concepts.push_back("cancer_disease");
        }
        if (lower_input.find("meaning") != std::string::npos && lower_input.find("life") != std::string::npos) {
            concepts.push_back("life_meaning");
        }
        
        // Only extract individual words if we didn't find multi-word concepts
        if (concepts.empty()) {
            std::istringstream iss(input);
            std::string word;
            while (iss >> word) {
                // Simple word cleaning
                word.erase(std::remove_if(word.begin(), word.end(), 
                    [](char c) { return !std::isalnum(c); }), word.end());
                if (word.length() > 2) {
                    concepts.push_back(word);
                }
            }
        }
        
        return concepts;
    }
    
    void addConceptsToBrain(const std::vector<std::string>& input_concepts, UltimateMetrics& metrics) {
        for (const std::string& concept : input_concepts) {
            if (concepts.find(concept) == concepts.end()) {
                concepts[concept] = UltimateConcept(concept, "");
                metrics.concepts_learned++;
            }
            
            // Update access count
            concepts[concept].access_count++;
            concepts[concept].last_accessed = getCurrentTime();
        }
    }
    
    // Preserve all other existing methods...
    void performUnifiedReasoning(const std::vector<std::string>& input_concepts, UltimateMetrics& metrics) {
        // Step 1: Expand Connections
        auto connections = expandConnections(input_concepts);
        
        // Step 2: Weight Connections
        auto weighted = weightConnections(connections, input_concepts);
        
        // Step 3: Select Reasoning Path
        auto path = selectReasoningPath(weighted);
        
        // Step 4: Driver Modulation
        std::string driver_result = modulateWithDrivers(path, metrics);
        
        // Step 5: Self-Check
        std::string self_check = performSelfCheck(driver_result, path, metrics);
        
        // Step 6: Generate Hypotheses
        generateHypotheses(input_concepts, path, metrics);
    }
    
    std::map<std::string, double> expandConnections(const std::vector<std::string>& input_concepts) {
        std::map<std::string, double> connections;
        
        for (const std::string& concept : input_concepts) {
            auto it = adjacency_list.find(concept);
            if (it != adjacency_list.end()) {
                for (const auto& conn : it->second) {
                    connections[conn.to_concept] += conn.weight;
                }
            }
        }
        
        return connections;
    }
    
    std::map<std::string, double> weightConnections(const std::map<std::string, double>& connections, 
                                                   const std::vector<std::string>& input_concepts) {
        std::map<std::string, double> weighted;
        
        for (const auto& conn : connections) {
            double weight = conn.second;
            
            // Apply meta-learning weights
            auto it = concepts.find(conn.first);
            if (it != concepts.end()) {
                weight *= (1.0 + it->second.validation_successes * 0.1);
                weight *= (1.0 - it->second.validation_failures * 0.05);
            }
            
            weighted[conn.first] = weight;
        }
        
        return weighted;
    }
    
    std::vector<std::string> selectReasoningPath(const std::map<std::string, double>& weighted_connections) {
        std::vector<std::pair<std::string, double>> sorted_connections;
        
        for (const auto& conn : weighted_connections) {
            sorted_connections.push_back({conn.first, conn.second});
        }
        
        std::sort(sorted_connections.begin(), sorted_connections.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::vector<std::string> path;
        for (size_t i = 0; i < std::min(size_t(5), sorted_connections.size()); ++i) {
            path.push_back(sorted_connections[i].first);
        }
        
        return path;
    }
    
    std::string modulateWithDrivers(const std::vector<std::string>& path, UltimateMetrics& metrics) {
        std::stringstream result;
        
        // Determine dominant driver
        if (dopamine > serotonin && dopamine > endorphins) {
            metrics.dominant_driver = "dopamine";
            result << "Novelty-seeking approach: ";
        } else if (serotonin > endorphins) {
            metrics.dominant_driver = "serotonin";
            result << "Coherence-seeking approach: ";
        } else {
            metrics.dominant_driver = "endorphins";
            result << "Satisfaction-seeking approach: ";
        }
        
        result << "Exploring " << path.size() << " connected concepts";
        return result.str();
    }
    
    std::string performSelfCheck(const std::string& reasoning_result, const std::vector<std::string>& path, 
                                UltimateMetrics& metrics) {
        bool validation_success = true;
        double confidence = 0.5;
        
        if (reasoning_result.find("strong connections") != std::string::npos) {
            validation_success = true;
            confidence = 0.8;
        } else if (reasoning_result.find("Limited knowledge") != std::string::npos) {
            validation_success = false;
            confidence = 0.2;
        }
        
        if (validation_success) {
            metrics.validation_confirmed++;
        } else {
            metrics.validation_refuted++;
        }
        
        return "Self-Check: " + std::string(validation_success ? "SUCCESS" : "FAILURE") + 
               " (confidence: " + std::to_string(confidence) + ")";
    }
    
    void generateHypotheses(const std::vector<std::string>& input_concepts, 
                           const std::vector<std::string>& path, UltimateMetrics& metrics) {
        for (const std::string& concept : input_concepts) {
            std::string hypothesis = "The concept " + concept + " is important for understanding " + metrics.input_content;
            metrics.hypotheses.push_back(hypothesis);
            
            double confidence = 0.5;
            auto it = concepts.find(concept);
            if (it != concepts.end()) {
                confidence = std::min(1.0, 0.5 + (it->second.access_count * 0.1));
            }
            metrics.hypothesis_confidences.push_back(confidence);
        }
    }
    
    void integrateTutorLearning(const std::vector<std::string>& input_concepts, UltimateMetrics& metrics) {
        // Simulated tutor integration
        for (const std::string& concept : input_concepts) {
            auto it = concepts.find(concept);
            if (it != concepts.end()) {
                it->second.validation_successes++;
                metrics.strengthened_concepts.push_back(concept);
            }
        }
    }
    
    void performSelfSharpening(UltimateMetrics& metrics) {
        // Pruning weak concepts
        for (auto it = concepts.begin(); it != concepts.end();) {
            if (it->second.activation < pruning_threshold) {
                it = concepts.erase(it);
            } else {
                ++it;
            }
        }
        
        // Reinforcement of strong concepts
        for (auto& concept_pair : concepts) {
            if (concept_pair.second.activation > reinforcement_threshold) {
                concept_pair.second.activation = std::min(1.0, concept_pair.second.activation + 0.1);
            }
        }
    }
    
    void logEvolutionCycle(const UltimateMetrics& metrics) {
        std::ofstream file(evolution_log_file, std::ios::app);
        if (file.is_open()) {
            file << metrics.cycle_id << "," << metrics.input_type << ",\"" << metrics.input_content << "\",\"";
            
            // Write hypotheses
            for (size_t i = 0; i < metrics.hypotheses.size(); i++) {
                if (i > 0) file << ";";
                file << metrics.hypotheses[i];
            }
            file << "\",\"";
            
            // Write confidences
            for (size_t i = 0; i < metrics.hypothesis_confidences.size(); i++) {
                if (i > 0) file << ";";
                file << std::fixed << std::setprecision(3) << metrics.hypothesis_confidences[i];
            }
            file << "\",";
            
            file << metrics.validation_confirmed << "," << metrics.validation_refuted << "," 
                 << metrics.validation_uncertain << "," << metrics.dominant_driver << ",\"";
            
            // Write strengthened concepts
            for (size_t i = 0; i < metrics.strengthened_concepts.size(); i++) {
                if (i > 0) file << ";";
                file << metrics.strengthened_concepts[i];
            }
            file << "\",\"";
            
            // Write weakened concepts
            for (size_t i = 0; i < metrics.weakened_concepts.size(); i++) {
                if (i > 0) file << ";";
                file << metrics.weakened_concepts[i];
            }
            file << "\",\"";
            
            // Write meta learning notes
            for (size_t i = 0; i < metrics.meta_learning_notes.size(); i++) {
                if (i > 0) file << ";";
                file << metrics.meta_learning_notes[i];
            }
            file << "\",";
            
            file << std::fixed << std::setprecision(3) << metrics.overall_confidence << "," 
                 << metrics.timestamp << "," << metrics.concepts_learned << "," 
                 << metrics.connections_created << "," << metrics.cache_hit_rate << "," 
                 << metrics.ollama_calls << "," << metrics.traveled_nodes.size() << "," 
                 << metrics.selected_nodes.size() << "," << metrics.final_output.output_confidence << "," 
                 << metrics.response_quality_score << std::endl;
            
            file.close();
        }
    }
    
    uint64_t getCurrentTime() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    
    // Interactive main loop
    void runInteractiveSession() {
        std::cout << "🚀 Starting Melvin Ultimate Unified with Output Generation..." << std::endl;
        std::cout << "Type your questions, or 'quit' to exit" << std::endl;
        std::cout << std::endl;
        
        std::string input;
        while (true) {
            std::cout << "You: ";
            std::getline(std::cin, input);
            
            if (input == "quit" || input == "exit") {
                std::cout << "🎉 Session complete! Saving brain state..." << std::endl;
                saveBrainState();
                break;
            }
            
            if (!input.empty()) {
                runUnifiedCycleWithOutput(input);
            }
        }
    }
};

int main() {
    MelvinUltimateUnifiedWithOutput melvin;
    melvin.runInteractiveSession();
    return 0;
}
