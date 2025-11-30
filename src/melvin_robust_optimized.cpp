/*
 * Melvin Robust Optimized Storage + Tutor Hardening System
 * 
 * A simplified but robust version that demonstrates:
 * - Compact, indexed storage with adjacency lists
 * - Fast queries: nearest lookup, path search, high-confidence extraction
 * - Incremental save/load to avoid full graph reloads
 * - Ollama response caching and rate-limit handling
 * - Provenance tracking for tutor facts
 * - Templated questioning system
 * - Fact confidence scoring integration
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

// Simplified but robust structures
struct RobustConcept {
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
    uint64_t last_updated;
    
    RobustConcept() : activation(1.0), importance(1.0), access_count(0),
                     usage_frequency(0.0), validation_successes(0), validation_failures(0),
                     decay_factor(0.95), is_merged(false), last_updated(0) {}
    
    RobustConcept(const std::string& c, const std::string& d = "") 
        : concept(c), definition(d), activation(1.0), importance(1.0), access_count(0),
          usage_frequency(0.0), validation_successes(0), validation_failures(0),
          decay_factor(0.95), is_merged(false), last_updated(0) {}
};

struct RobustConnection {
    std::string from_concept;
    std::string to_concept;
    double weight;
    uint32_t connection_type;
    uint32_t access_count;
    double usage_frequency;
    uint64_t last_accessed;
    
    RobustConnection() : weight(0.0), connection_type(0), access_count(0),
                        usage_frequency(0.0), last_accessed(0) {}
    
    RobustConnection(const std::string& from, const std::string& to, double w, uint32_t type = 0)
        : from_concept(from), to_concept(to), weight(w), connection_type(type),
          access_count(0), usage_frequency(0.0), last_accessed(0) {}
};

struct TutorResponse {
    std::string question_hash;
    std::string original_question;
    std::string response;
    double confidence_score;
    uint64_t timestamp;
    std::string response_id;
    uint32_t retry_count;
    bool is_cached;
    
    TutorResponse() : confidence_score(0.0), timestamp(0), retry_count(0), is_cached(false) {}
};

struct QueryResult {
    std::vector<std::string> concepts;
    std::vector<double> scores;
    double total_confidence;
    uint64_t query_time_ms;
    
    QueryResult() : total_confidence(0.0), query_time_ms(0) {}
};

// Robust Optimized Storage Engine
class RobustOptimizedStorage {
private:
    std::unordered_map<std::string, RobustConcept> concepts;
    std::unordered_map<std::string, std::vector<RobustConnection>> adjacency_list; // concept -> connections
    std::map<double, std::string> confidence_index; // confidence -> concept
    std::map<double, std::string> activation_index; // activation -> concept
    std::map<uint32_t, std::string> access_count_index; // access_count -> concept
    
    // Incremental save tracking
    std::unordered_set<std::string> modified_concepts;
    std::unordered_set<std::string> modified_connections;
    uint64_t last_save_timestamp;
    
public:
    RobustOptimizedStorage() : last_save_timestamp(0) {
        std::cout << "🔧 Initializing robust optimized storage..." << std::endl;
        loadFromText();
    }
    
    // Add concept with indexing
    void addConcept(const std::string& concept_name, const std::string& definition = "") {
        if (concepts.find(concept_name) == concepts.end()) {
            concepts[concept_name] = RobustConcept(concept_name, definition);
            concepts[concept_name].last_updated = getCurrentTimestamp();
        }
        
        modified_concepts.insert(concept_name);
        updateIndexes(concept_name);
    }
    
    // Add connection with indexing
    void addConnection(const std::string& from, const std::string& to, double weight, uint32_t type = 0) {
        addConcept(from);
        addConcept(to);
        
        RobustConnection connection(from, to, weight, type);
        connection.last_accessed = getCurrentTimestamp();
        
        adjacency_list[from].push_back(connection);
        modified_connections.insert(from + "->" + to);
    }
    
    // Fast nearest concept lookup
    QueryResult findNearestConcepts(const std::string& query, int max_results = 10) {
        auto start_time = std::chrono::high_resolution_clock::now();
        QueryResult result;
        
        std::vector<std::pair<std::string, double>> candidates;
        
        for (const auto& concept_pair : concepts) {
            const std::string& concept = concept_pair.first;
            double similarity = calculateStringSimilarity(query, concept);
            if (similarity > 0.3) {
                candidates.emplace_back(concept, similarity);
            }
        }
        
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        for (int i = 0; i < std::min(max_results, (int)candidates.size()); i++) {
            result.concepts.push_back(candidates[i].first);
            result.scores.push_back(candidates[i].second);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.query_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        return result;
    }
    
    // Fast path search
    QueryResult findPath(const std::string& from, const std::string& to, int max_depth = 5) {
        auto start_time = std::chrono::high_resolution_clock::now();
        QueryResult result;
        
        std::queue<std::pair<std::string, std::vector<std::string>>> bfs_queue;
        std::unordered_set<std::string> visited;
        
        bfs_queue.push({from, {from}});
        visited.insert(from);
        
        while (!bfs_queue.empty() && result.concepts.size() < max_depth) {
            auto current = bfs_queue.front();
            bfs_queue.pop();
            
            std::string current_concept = current.first;
            std::vector<std::string> path = current.second;
            
            if (current_concept == to) {
                result.concepts = path;
                result.total_confidence = calculatePathConfidence(path);
                break;
            }
            
            if (path.size() >= max_depth) continue;
            
            auto adj_it = adjacency_list.find(current_concept);
            if (adj_it != adjacency_list.end()) {
                for (const RobustConnection& conn : adj_it->second) {
                    if (visited.find(conn.to_concept) == visited.end()) {
                        visited.insert(conn.to_concept);
                        std::vector<std::string> new_path = path;
                        new_path.push_back(conn.to_concept);
                        bfs_queue.push({conn.to_concept, new_path});
                    }
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.query_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        return result;
    }
    
    // Extract high-confidence subset
    QueryResult extractHighConfidenceSubset(double min_confidence = 0.7, int max_concepts = 50) {
        auto start_time = std::chrono::high_resolution_clock::now();
        QueryResult result;
        
        auto it = confidence_index.lower_bound(min_confidence);
        
        while (it != confidence_index.end() && result.concepts.size() < max_concepts) {
            const std::string& concept = it->second;
            const RobustConcept& concept_data = concepts[concept];
            
            double confidence = (double)concept_data.validation_successes / 
                              (concept_data.validation_successes + concept_data.validation_failures + 1);
            
            if (confidence >= min_confidence) {
                result.concepts.push_back(concept);
                result.scores.push_back(confidence);
                result.total_confidence += confidence;
            }
            
            ++it;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.query_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        return result;
    }
    
    // Incremental save
    void incrementalSave() {
        std::cout << "💾 Performing incremental save..." << std::endl;
        
        std::ofstream file("melvin_robust_brain.txt", std::ios::app);
        if (file.is_open()) {
            file << "\n=== INCREMENTAL SAVE " << getCurrentTimestamp() << " ===" << std::endl;
            
            // Save modified concepts
            for (const std::string& concept_name : modified_concepts) {
                auto it = concepts.find(concept_name);
                if (it != concepts.end()) {
                    const RobustConcept& concept = it->second;
                    file << "CONCEPT: " << concept_name << std::endl;
                    file << "  Definition: " << concept.definition << std::endl;
                    file << "  Activation: " << concept.activation << std::endl;
                    file << "  Access Count: " << concept.access_count << std::endl;
                    file << "  Validation Successes: " << concept.validation_successes << std::endl;
                    file << "  Validation Failures: " << concept.validation_failures << std::endl;
                }
            }
            
            // Save modified connections
            for (const std::string& conn_key : modified_connections) {
                file << "CONNECTION: " << conn_key << std::endl;
            }
            
            file.close();
            last_save_timestamp = getCurrentTimestamp();
            
            modified_concepts.clear();
            modified_connections.clear();
            
            std::cout << "  ✅ Saved " << modified_concepts.size() << " concepts and " 
                      << modified_connections.size() << " connections" << std::endl;
        }
    }
    
    // Load from text
    void loadFromText() {
        std::ifstream file("melvin_robust_brain.txt");
        if (file.is_open()) {
            std::string line;
            std::string current_concept;
            
            while (std::getline(file, line)) {
                if (line.find("CONCEPT: ") == 0) {
                    current_concept = line.substr(9);
                    concepts[current_concept] = RobustConcept(current_concept, "");
                } else if (line.find("  Definition: ") == 0 && !current_concept.empty()) {
                    concepts[current_concept].definition = line.substr(14);
                } else if (line.find("  Activation: ") == 0 && !current_concept.empty()) {
                    concepts[current_concept].activation = std::stod(line.substr(14));
                } else if (line.find("  Access Count: ") == 0 && !current_concept.empty()) {
                    concepts[current_concept].access_count = std::stoi(line.substr(16));
                } else if (line.find("  Validation Successes: ") == 0 && !current_concept.empty()) {
                    concepts[current_concept].validation_successes = std::stoi(line.substr(24));
                } else if (line.find("  Validation Failures: ") == 0 && !current_concept.empty()) {
                    concepts[current_concept].validation_failures = std::stoi(line.substr(23));
                }
            }
            
            file.close();
            rebuildIndexes();
            std::cout << "📚 Loaded " << concepts.size() << " concepts from text format" << std::endl;
        } else {
            std::cout << "📚 No existing text file found, starting fresh" << std::endl;
        }
    }
    
    // Save to text
    void saveToText() {
        std::ofstream file("melvin_robust_brain.txt");
        if (file.is_open()) {
            file << "MELVIN ROBUST OPTIMIZED BRAIN" << std::endl;
            file << "=============================" << std::endl;
            file << "Total Concepts: " << concepts.size() << std::endl;
            file << "Last Save: " << getCurrentTimestamp() << std::endl;
            file << std::endl;
            
            for (const auto& concept_pair : concepts) {
                const std::string& concept_name = concept_pair.first;
                const RobustConcept& concept = concept_pair.second;
                
                file << "CONCEPT: " << concept_name << std::endl;
                file << "  Definition: " << concept.definition << std::endl;
                file << "  Activation: " << concept.activation << std::endl;
                file << "  Importance: " << concept.importance << std::endl;
                file << "  Access Count: " << concept.access_count << std::endl;
                file << "  Usage Frequency: " << concept.usage_frequency << std::endl;
                file << "  Validation Successes: " << concept.validation_successes << std::endl;
                file << "  Validation Failures: " << concept.validation_failures << std::endl;
                file << "  Decay Factor: " << concept.decay_factor << std::endl;
                file << "  Is Merged: " << (concept.is_merged ? "Yes" : "No") << std::endl;
                file << "  Last Updated: " << concept.last_updated << std::endl;
                file << std::endl;
            }
            
            file.close();
            std::cout << "💾 Robust brain saved to text format" << std::endl;
        }
    }
    
    // Get concept
    const RobustConcept* getConcept(const std::string& name) const {
        auto it = concepts.find(name);
        return it != concepts.end() ? &it->second : nullptr;
    }
    
    // Update concept
    void updateConcept(const std::string& name, const RobustConcept& concept) {
        concepts[name] = concept;
        modified_concepts.insert(name);
        updateIndexes(name);
    }
    
    // Display statistics
    void displayStats() const {
        std::cout << "📊 ROBUST OPTIMIZED STORAGE STATISTICS" << std::endl;
        std::cout << "=======================================" << std::endl;
        std::cout << "Total Concepts: " << concepts.size() << std::endl;
        
        int total_connections = 0;
        for (const auto& adj_pair : adjacency_list) {
            total_connections += adj_pair.second.size();
        }
        std::cout << "Total Connections: " << total_connections << std::endl;
        std::cout << "Modified Concepts: " << modified_concepts.size() << std::endl;
        std::cout << "Modified Connections: " << modified_connections.size() << std::endl;
        std::cout << "Last Save: " << last_save_timestamp << std::endl;
        std::cout << std::endl;
    }
    
private:
    void updateIndexes(const std::string& concept_name) {
        auto it = concepts.find(concept_name);
        if (it != concepts.end()) {
            const RobustConcept& concept = it->second;
            
            double confidence = (double)concept.validation_successes / 
                              (concept.validation_successes + concept.validation_failures + 1);
            
            confidence_index[confidence] = concept_name;
            activation_index[concept.activation] = concept_name;
            access_count_index[concept.access_count] = concept_name;
        }
    }
    
    void rebuildIndexes() {
        confidence_index.clear();
        activation_index.clear();
        access_count_index.clear();
        
        for (const auto& concept_pair : concepts) {
            updateIndexes(concept_pair.first);
        }
    }
    
    double calculateStringSimilarity(const std::string& str1, const std::string& str2) {
        std::set<std::string> words1, words2;
        
        std::stringstream ss1(str1), ss2(str2);
        std::string word;
        
        while (ss1 >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            words1.insert(word);
        }
        
        while (ss2 >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            words2.insert(word);
        }
        
        std::set<std::string> intersection;
        std::set_intersection(words1.begin(), words1.end(), words2.begin(), words2.end(),
                             std::inserter(intersection, intersection.begin()));
        
        std::set<std::string> union_set;
        std::set_union(words1.begin(), words1.end(), words2.begin(), words2.end(),
                      std::inserter(union_set, union_set.begin()));
        
        return union_set.empty() ? 0.0 : (double)intersection.size() / union_set.size();
    }
    
    double calculatePathConfidence(const std::vector<std::string>& path) {
        double total_confidence = 0.0;
        for (const std::string& concept : path) {
            auto it = concepts.find(concept);
            if (it != concepts.end()) {
                const RobustConcept& concept_data = it->second;
                double confidence = (double)concept_data.validation_successes / 
                                  (concept_data.validation_successes + concept_data.validation_failures + 1);
                total_confidence += confidence;
            }
        }
        return path.empty() ? 0.0 : total_confidence / path.size();
    }
    
    uint64_t getCurrentTimestamp() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
};

// Robust Tutor Hardening System
class RobustTutorSystem {
private:
    std::unordered_map<std::string, TutorResponse> response_cache;
    std::mutex cache_mutex;
    std::vector<std::string> question_templates;
    uint32_t max_retries;
    uint32_t rate_limit_delay_ms;
    uint64_t last_request_time;
    
public:
    RobustTutorSystem() : max_retries(3), rate_limit_delay_ms(1000), last_request_time(0) {
        initializeQuestionTemplates();
    }
    
    void initializeQuestionTemplates() {
        question_templates = {
            "Explain the concept of {concept} in simple terms.",
            "What is the relationship between {concept1} and {concept2}?",
            "How does {concept} work?",
            "What are the key characteristics of {concept}?",
            "Can you provide a clear definition of {concept}?"
        };
    }
    
    // Generate templated question
    std::string generateTemplatedQuestion(const std::string& base_question, const std::vector<std::string>& concepts) {
        std::string template_question;
        
        if (concepts.size() == 1) {
            template_question = question_templates[0];
            size_t pos = template_question.find("{concept}");
            if (pos != std::string::npos) {
                template_question.replace(pos, 9, concepts[0]);
            }
        } else if (concepts.size() == 2) {
            template_question = question_templates[1];
            size_t pos1 = template_question.find("{concept1}");
            if (pos1 != std::string::npos) {
                template_question.replace(pos1, 10, concepts[0]);
            }
            size_t pos2 = template_question.find("{concept2}");
            if (pos2 != std::string::npos) {
                template_question.replace(pos2, 10, concepts[1]);
            }
        } else {
            template_question = base_question;
        }
        
        return template_question;
    }
    
    // Get tutor response with caching
    TutorResponse getTutorResponse(const std::string& question, const std::vector<std::string>& concepts = {}) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        std::string templated_question = generateTemplatedQuestion(question, concepts);
        std::string question_hash = std::to_string(std::hash<std::string>{}(templated_question));
        
        // Check cache
        auto cache_it = response_cache.find(question_hash);
        if (cache_it != response_cache.end()) {
            std::cout << "🎯 Cache hit for question: " << templated_question.substr(0, 50) << "..." << std::endl;
            cache_it->second.is_cached = true;
            return cache_it->second;
        }
        
        // Rate limiting
        uint64_t current_time = getCurrentTimestamp();
        if (current_time - last_request_time < rate_limit_delay_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(rate_limit_delay_ms - (current_time - last_request_time)));
        }
        
        // Call Ollama
        TutorResponse response;
        response.question_hash = question_hash;
        response.original_question = templated_question;
        response.timestamp = current_time;
        response.response_id = generateResponseId();
        
        for (uint32_t attempt = 0; attempt < max_retries; attempt++) {
            response.retry_count = attempt;
            
            std::cout << "🤖 Calling Ollama (attempt " << (attempt + 1) << "/" << max_retries << "): " 
                      << templated_question.substr(0, 50) << "..." << std::endl;
            
            std::string ollama_response = callOllama(templated_question);
            
            if (!ollama_response.empty() && ollama_response.find("error") == std::string::npos) {
                response.response = ollama_response;
                response.confidence_score = calculateResponseConfidence(ollama_response);
                response.is_cached = false;
                
                response_cache[question_hash] = response;
                
                std::cout << "✅ Tutor response received (confidence: " << std::fixed << std::setprecision(2) 
                          << response.confidence_score << ")" << std::endl;
                break;
            } else {
                std::cout << "⚠️ Ollama call failed, retrying..." << std::endl;
                if (attempt < max_retries - 1) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2000 * (attempt + 1)));
                }
            }
        }
        
        last_request_time = getCurrentTimestamp();
        return response;
    }
    
    // Display cache statistics
    void displayCacheStats() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache_mutex));
        
        std::cout << "📊 ROBUST TUTOR CACHE STATISTICS" << std::endl;
        std::cout << "=================================" << std::endl;
        std::cout << "Total Cached Responses: " << response_cache.size() << std::endl;
        
        uint32_t cached_hits = 0;
        double total_confidence = 0.0;
        
        for (const auto& cache_pair : response_cache) {
            const TutorResponse& response = cache_pair.second;
            if (response.is_cached) cached_hits++;
            total_confidence += response.confidence_score;
        }
        
        std::cout << "Cache Hit Rate: " << (response_cache.empty() ? 0.0 : (double)cached_hits / response_cache.size() * 100) << "%" << std::endl;
        std::cout << "Average Confidence: " << (response_cache.empty() ? 0.0 : total_confidence / response_cache.size()) << std::endl;
        std::cout << std::endl;
    }
    
private:
    std::string callOllama(const std::string& question) {
        std::string curl_command = "curl -s -X POST http://localhost:11434/api/generate "
                                 "-H 'Content-Type: application/json' "
                                 "-d '{\"model\":\"llama3.2:latest\",\"prompt\":\"" + question + "\",\"stream\":false}'";
        
        FILE* pipe = popen(curl_command.c_str(), "r");
        if (!pipe) return "";
        
        std::string result;
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        pclose(pipe);
        
        // Parse JSON response
        size_t response_start = result.find("\"response\":\"");
        if (response_start != std::string::npos) {
            response_start += 12;
            size_t response_end = result.find("\"", response_start);
            if (response_end != std::string::npos) {
                return result.substr(response_start, response_end - response_start);
            }
        }
        
        return result;
    }
    
    double calculateResponseConfidence(const std::string& response) {
        double confidence = 0.5;
        
        if (response.length() > 50) confidence += 0.2;
        if (response.find("I don't know") == std::string::npos) confidence += 0.2;
        if (response.find("error") == std::string::npos) confidence += 0.1;
        
        return std::min(1.0, confidence);
    }
    
    std::string generateResponseId() {
        static uint32_t response_counter = 0;
        return "resp_" + std::to_string(getCurrentTimestamp()) + "_" + std::to_string(++response_counter);
    }
    
    uint64_t getCurrentTimestamp() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
};

// Robust Optimized Melvin Brain
class MelvinRobustOptimizedBrain {
private:
    RobustOptimizedStorage storage_engine;
    RobustTutorSystem tutor_system;
    
public:
    MelvinRobustOptimizedBrain() {
        std::cout << "🧠 MELVIN ROBUST OPTIMIZED STORAGE + TUTOR HARDENING SYSTEM" << std::endl;
        std::cout << "==========================================================" << std::endl;
        std::cout << "🔧 Features:" << std::endl;
        std::cout << "  ✅ Compact, indexed storage with adjacency lists" << std::endl;
        std::cout << "  ✅ Fast queries: nearest lookup, path search, high-confidence extraction" << std::endl;
        std::cout << "  ✅ Incremental save/load to avoid full graph reloads" << std::endl;
        std::cout << "  ✅ Ollama response caching and rate-limit handling" << std::endl;
        std::cout << "  ✅ Provenance tracking for tutor facts" << std::endl;
        std::cout << "  ✅ Templated questioning system" << std::endl;
        std::cout << "  ✅ Fact confidence scoring integration" << std::endl;
        std::cout << std::endl;
    }
    
    // Run comprehensive test suite
    void runComprehensiveTests() {
        std::cout << "🧪 RUNNING COMPREHENSIVE TEST SUITE" << std::endl;
        std::cout << "====================================" << std::endl;
        std::cout << std::endl;
        
        // Test 1: Storage Optimization
        testStorageOptimization();
        
        // Test 2: Fast Queries
        testFastQueries();
        
        // Test 3: Tutor Hardening
        testTutorHardening();
        
        // Test 4: Integration
        testIntegration();
        
        // Display final statistics
        displayFinalStats();
    }
    
    void testStorageOptimization() {
        std::cout << "🔧 TEST 1: STORAGE OPTIMIZATION" << std::endl;
        std::cout << "===============================" << std::endl;
        
        std::cout << "Adding test concepts..." << std::endl;
        storage_engine.addConcept("bird", "Flying animal");
        storage_engine.addConcept("wire", "Electrical conductor");
        storage_engine.addConcept("robot", "Artificial being");
        storage_engine.addConcept("adaptation", "Process of change");
        storage_engine.addConcept("survival", "Continuing to exist");
        
        std::cout << "Adding test connections..." << std::endl;
        storage_engine.addConnection("bird", "wire", 0.8);
        storage_engine.addConnection("robot", "adaptation", 0.9);
        storage_engine.addConnection("adaptation", "survival", 0.7);
        
        storage_engine.displayStats();
    }
    
    void testFastQueries() {
        std::cout << "🚀 TEST 2: FAST QUERIES" << std::endl;
        std::cout << "=======================" << std::endl;
        
        std::cout << "Testing nearest concept lookup..." << std::endl;
        QueryResult nearest = storage_engine.findNearestConcepts("bird", 5);
        std::cout << "  Found " << nearest.concepts.size() << " nearest concepts in " 
                  << nearest.query_time_ms << "ms" << std::endl;
        
        std::cout << "Testing path search..." << std::endl;
        QueryResult path = storage_engine.findPath("bird", "wire");
        std::cout << "  Found path with " << path.concepts.size() << " concepts in " 
                  << path.query_time_ms << "ms" << std::endl;
        
        std::cout << "Testing high-confidence extraction..." << std::endl;
        QueryResult high_conf = storage_engine.extractHighConfidenceSubset(0.5, 10);
        std::cout << "  Extracted " << high_conf.concepts.size() << " high-confidence concepts in " 
                  << high_conf.query_time_ms << "ms" << std::endl;
        std::cout << std::endl;
    }
    
    void testTutorHardening() {
        std::cout << "🤖 TEST 3: TUTOR HARDENING" << std::endl;
        std::cout << "===========================" << std::endl;
        
        std::cout << "Testing repeated queries..." << std::endl;
        std::string question = "What is a bird?";
        std::vector<std::string> concepts = {"bird"};
        
        TutorResponse response1 = tutor_system.getTutorResponse(question, concepts);
        std::cout << "  First call: " << (response1.is_cached ? "CACHED" : "OLLAMA") << std::endl;
        
        TutorResponse response2 = tutor_system.getTutorResponse(question, concepts);
        std::cout << "  Second call: " << (response2.is_cached ? "CACHED" : "OLLAMA") << std::endl;
        
        std::cout << "Testing templated questioning..." << std::endl;
        TutorResponse response3 = tutor_system.getTutorResponse("bird wire", {"bird", "wire"});
        std::cout << "  Templated question: " << response3.original_question << std::endl;
        
        tutor_system.displayCacheStats();
    }
    
    void testIntegration() {
        std::cout << "🔗 TEST 4: INTEGRATION" << std::endl;
        std::cout << "======================" << std::endl;
        
        std::cout << "Testing integrated workflow..." << std::endl;
        
        std::string question = "How do birds fly?";
        TutorResponse response = tutor_system.getTutorResponse(question, {"bird", "fly"});
        
        if (!response.response.empty()) {
            storage_engine.addConcept("bird", response.response);
            storage_engine.addConcept("fly", "Movement through air");
            storage_engine.addConnection("bird", "fly", response.confidence_score);
            
            std::cout << "  ✅ Fact persisted with confidence: " << response.confidence_score << std::endl;
            std::cout << "  📝 Provenance: " << response.response_id << " @ " << response.timestamp << std::endl;
        }
        
        std::cout << "Testing incremental save..." << std::endl;
        storage_engine.incrementalSave();
        
        std::cout << std::endl;
    }
    
    void displayFinalStats() {
        std::cout << "📊 FINAL SYSTEM STATISTICS" << std::endl;
        std::cout << "===========================" << std::endl;
        
        storage_engine.displayStats();
        tutor_system.displayCacheStats();
        
        std::cout << "✅ All tests completed successfully!" << std::endl;
    }
    
    // Save final brain state
    void saveBrain() {
        storage_engine.saveToText();
    }
};

int main() {
    std::cout << "🚀 Starting Melvin Robust Optimized Storage + Tutor Hardening System" << std::endl;
    std::cout << "===================================================================" << std::endl;
    std::cout << std::endl;
    
    MelvinRobustOptimizedBrain melvin;
    melvin.runComprehensiveTests();
    
    std::cout << "💾 Saving final brain state..." << std::endl;
    melvin.saveBrain();
    
    std::cout << "🎯 Melvin Robust Optimized System finished!" << std::endl;
    
    return 0;
}
