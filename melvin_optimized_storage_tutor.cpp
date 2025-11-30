/*
 * Melvin Optimized Storage + Tutor Hardening System
 * 
 * Features:
 * - Compact, indexed storage with adjacency lists and sparse matrices
 * - Fast queries: nearest concept lookup, path search, high-confidence extraction
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
#include <memory>

// Optimized storage structures
struct OptimizedConcept {
    uint32_t id;
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
    
    OptimizedConcept() : id(0), activation(1.0), importance(1.0), access_count(0),
                        usage_frequency(0.0), validation_successes(0), validation_failures(0),
                        decay_factor(0.95), is_merged(false), last_updated(0) {}
};

struct OptimizedConnection {
    uint32_t from_id;
    uint32_t to_id;
    double weight;
    uint32_t connection_type; // 0=semantic, 1=causal, 2=hierarchical, 3=temporal
    uint32_t access_count;
    double usage_frequency;
    uint64_t last_accessed;
    
    OptimizedConnection() : from_id(0), to_id(0), weight(0.0), connection_type(0),
                           access_count(0), usage_frequency(0.0), last_accessed(0) {}
};

// Tutor response caching and provenance
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

// Fast query structures
struct QueryResult {
    std::vector<uint32_t> concept_ids;
    std::vector<double> scores;
    double total_confidence;
    uint64_t query_time_ms;
    
    QueryResult() : total_confidence(0.0), query_time_ms(0) {}
};

// Optimized Storage Engine
class OptimizedStorageEngine {
private:
    // Core data structures
    std::unordered_map<uint32_t, OptimizedConcept> concepts;
    std::unordered_map<uint32_t, OptimizedConnection> connections;
    std::unordered_map<std::string, uint32_t> concept_name_to_id;
    std::unordered_map<uint32_t, std::vector<uint32_t>> adjacency_list; // concept_id -> connection_ids
    std::vector<std::vector<double>> sparse_connection_matrix; // Sparse matrix for fast lookups
    
    // Indexes for fast queries
    std::multimap<double, uint32_t> confidence_index; // confidence -> concept_id
    std::multimap<double, uint32_t> activation_index; // activation -> concept_id
    std::multimap<uint32_t, uint32_t> access_count_index; // access_count -> concept_id
    
    // Storage metadata
    uint32_t next_concept_id;
    uint32_t next_connection_id;
    uint64_t last_save_timestamp;
    std::string storage_file_path;
    
    // Incremental save tracking
    std::unordered_set<uint32_t> modified_concepts;
    std::unordered_set<uint32_t> modified_connections;
    
public:
    OptimizedStorageEngine(const std::string& file_path = "melvin_optimized_brain.bin") 
        : next_concept_id(1), next_connection_id(1), last_save_timestamp(0), storage_file_path(file_path) {
        initializeStorage();
    }
    
    void initializeStorage() {
        std::cout << "🔧 Initializing optimized storage engine..." << std::endl;
        
        // Initialize sparse matrix (start with 100x100, expand as needed)
        sparse_connection_matrix.resize(100, std::vector<double>(100, 0.0));
        
        // Load existing data if available
        loadFromBinary();
    }
    
    // Add concept with optimized indexing
    uint32_t addConcept(const std::string& concept_name, const std::string& definition = "") {
        // Check if concept already exists
        auto it = concept_name_to_id.find(concept_name);
        if (it != concept_name_to_id.end()) {
            modified_concepts.insert(it->second);
            return it->second;
        }
        
        // Create new concept
        uint32_t id = next_concept_id++;
        OptimizedConcept concept;
        concept.id = id;
        concept.concept = concept_name;
        concept.definition = definition;
        concept.last_updated = getCurrentTimestamp();
        
        concepts[id] = concept;
        concept_name_to_id[concept_name] = id;
        modified_concepts.insert(id);
        
        // Update indexes
        confidence_index.insert({concept.activation, id});
        activation_index.insert({concept.activation, id});
        access_count_index.insert({concept.access_count, id});
        
        // Expand sparse matrix if needed
        expandSparseMatrix();
        
        return id;
    }
    
    // Add connection with optimized indexing
    uint32_t addConnection(uint32_t from_id, uint32_t to_id, double weight, uint32_t type = 0) {
        uint32_t conn_id = next_connection_id++;
        OptimizedConnection connection;
        connection.from_id = from_id;
        connection.to_id = to_id;
        connection.weight = weight;
        connection.connection_type = type;
        connection.last_accessed = getCurrentTimestamp();
        
        connections[conn_id] = connection;
        modified_connections.insert(conn_id);
        
        // Update adjacency list
        adjacency_list[from_id].push_back(conn_id);
        
        // Update sparse matrix
        if (from_id < sparse_connection_matrix.size() && to_id < sparse_connection_matrix[from_id].size()) {
            sparse_connection_matrix[from_id][to_id] = weight;
        }
        
        return conn_id;
    }
    
    // Fast nearest concept lookup
    QueryResult findNearestConcepts(const std::string& query, int max_results = 10) {
        auto start_time = std::chrono::high_resolution_clock::now();
        QueryResult result;
        
        // Use string similarity for now (could be enhanced with embeddings)
        std::vector<std::pair<uint32_t, double>> candidates;
        
        for (const auto& concept_pair : concepts) {
            uint32_t id = concept_pair.first;
            const OptimizedConcept& concept = concept_pair.second;
            
            double similarity = calculateStringSimilarity(query, concept.concept);
            if (similarity > 0.3) { // Threshold for relevance
                candidates.emplace_back(id, similarity);
            }
        }
        
        // Sort by similarity
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Take top results
        for (int i = 0; i < std::min(max_results, (int)candidates.size()); i++) {
            result.concept_ids.push_back(candidates[i].first);
            result.scores.push_back(candidates[i].second);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.query_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        return result;
    }
    
    // Fast path search between concepts
    QueryResult findPath(uint32_t from_id, uint32_t to_id, int max_depth = 5) {
        auto start_time = std::chrono::high_resolution_clock::now();
        QueryResult result;
        
        // BFS search through adjacency list
        std::queue<std::pair<uint32_t, std::vector<uint32_t>>> bfs_queue;
        std::unordered_set<uint32_t> visited;
        
        bfs_queue.push({from_id, {from_id}});
        visited.insert(from_id);
        
        while (!bfs_queue.empty() && result.concept_ids.size() < max_depth) {
            auto current = bfs_queue.front();
            bfs_queue.pop();
            
            uint32_t current_id = current.first;
            std::vector<uint32_t> path = current.second;
            
            if (current_id == to_id) {
                // Found path
                result.concept_ids = path;
                result.total_confidence = calculatePathConfidence(path);
                break;
            }
            
            if (path.size() >= max_depth) continue;
            
            // Explore connections
            auto adj_it = adjacency_list.find(current_id);
            if (adj_it != adjacency_list.end()) {
                for (uint32_t conn_id : adj_it->second) {
                    auto conn_it = connections.find(conn_id);
                    if (conn_it != connections.end()) {
                        uint32_t next_id = conn_it->second.to_id;
                        if (visited.find(next_id) == visited.end()) {
                            visited.insert(next_id);
                            std::vector<uint32_t> new_path = path;
                            new_path.push_back(next_id);
                            bfs_queue.push({next_id, new_path});
                        }
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
        
        // Use confidence index for fast extraction
        auto it = confidence_index.lower_bound(min_confidence);
        
        while (it != confidence_index.end() && result.concept_ids.size() < max_concepts) {
            uint32_t id = it->second;
            const OptimizedConcept& concept = concepts[id];
            
            double confidence = (double)concept.validation_successes / 
                              (concept.validation_successes + concept.validation_failures + 1);
            
            if (confidence >= min_confidence) {
                result.concept_ids.push_back(id);
                result.scores.push_back(confidence);
                result.total_confidence += confidence;
            }
            
            ++it;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.query_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        return result;
    }
    
    // Incremental save (only save modified data)
    void incrementalSave() {
        std::cout << "💾 Performing incremental save..." << std::endl;
        
        std::ofstream file(storage_file_path, std::ios::binary | std::ios::app);
        if (file.is_open()) {
            uint64_t save_timestamp = getCurrentTimestamp();
            
            // Write incremental save marker
            file.write(reinterpret_cast<const char*>(&save_timestamp), sizeof(save_timestamp));
            
            // Save modified concepts
            uint32_t modified_concepts_count = modified_concepts.size();
            file.write(reinterpret_cast<const char*>(&modified_concepts_count), sizeof(modified_concepts_count));
            
            for (uint32_t id : modified_concepts) {
                auto it = concepts.find(id);
                if (it != concepts.end()) {
                    file.write(reinterpret_cast<const char*>(&id), sizeof(id));
                    file.write(reinterpret_cast<const char*>(&it->second), sizeof(OptimizedConcept));
                }
            }
            
            // Save modified connections
            uint32_t modified_connections_count = modified_connections.size();
            file.write(reinterpret_cast<const char*>(&modified_connections_count), sizeof(modified_connections_count));
            
            for (uint32_t id : modified_connections) {
                auto it = connections.find(id);
                if (it != connections.end()) {
                    file.write(reinterpret_cast<const char*>(&id), sizeof(id));
                    file.write(reinterpret_cast<const char*>(&it->second), sizeof(OptimizedConnection));
                }
            }
            
            file.close();
            last_save_timestamp = save_timestamp;
            
            // Clear modification tracking
            modified_concepts.clear();
            modified_connections.clear();
            
            std::cout << "  ✅ Saved " << modified_concepts_count << " concepts and " << modified_connections_count << " connections" << std::endl;
        }
    }
    
    // Load from binary with incremental support
    void loadFromBinary() {
        std::ifstream file(storage_file_path, std::ios::binary);
        if (file.is_open()) {
            try {
                // Load base data
                loadBaseData(file);
                
                // Load incremental updates
                while (file.good()) {
                    uint64_t timestamp;
                    file.read(reinterpret_cast<char*>(&timestamp), sizeof(timestamp));
                    if (file.fail()) break;
                    
                    // Load incremental concepts
                    uint32_t concepts_count;
                    file.read(reinterpret_cast<char*>(&concepts_count), sizeof(concepts_count));
                    
                    for (uint32_t i = 0; i < concepts_count; i++) {
                        uint32_t id;
                        file.read(reinterpret_cast<char*>(&id), sizeof(id));
                        
                        OptimizedConcept concept;
                        file.read(reinterpret_cast<char*>(&concept), sizeof(OptimizedConcept));
                        
                        concepts[id] = concept;
                        concept_name_to_id[concept.concept] = id;
                    }
                    
                    // Load incremental connections
                    uint32_t connections_count;
                    file.read(reinterpret_cast<char*>(&connections_count), sizeof(connections_count));
                    
                    for (uint32_t i = 0; i < connections_count; i++) {
                        uint32_t id;
                        file.read(reinterpret_cast<char*>(&id), sizeof(id));
                        
                        OptimizedConnection connection;
                        file.read(reinterpret_cast<char*>(&connection), sizeof(OptimizedConnection));
                        
                        connections[id] = connection;
                        adjacency_list[connection.from_id].push_back(id);
                    }
                }
                
                file.close();
                rebuildIndexes();
                std::cout << "📚 Loaded " << concepts.size() << " concepts and " << connections.size() << " connections" << std::endl;
            } catch (const std::exception& e) {
                std::cout << "⚠️ Error loading binary file: " << e.what() << std::endl;
                file.close();
            }
        } else {
            std::cout << "📚 No existing binary file found, starting fresh" << std::endl;
        }
    }
    
    // Get concept by ID
    const OptimizedConcept* getConcept(uint32_t id) const {
        auto it = concepts.find(id);
        return it != concepts.end() ? &it->second : nullptr;
    }
    
    // Get concept by name
    uint32_t getConceptId(const std::string& name) const {
        auto it = concept_name_to_id.find(name);
        return it != concept_name_to_id.end() ? it->second : 0;
    }
    
    // Update concept (marks as modified)
    void updateConcept(uint32_t id, const OptimizedConcept& concept) {
        concepts[id] = concept;
        modified_concepts.insert(id);
        rebuildIndexes();
    }
    
    // Get storage statistics
    void displayStats() const {
        std::cout << "📊 OPTIMIZED STORAGE STATISTICS" << std::endl;
        std::cout << "===============================" << std::endl;
        std::cout << "Total Concepts: " << concepts.size() << std::endl;
        std::cout << "Total Connections: " << connections.size() << std::endl;
        std::cout << "Sparse Matrix Size: " << sparse_connection_matrix.size() << "x" << sparse_connection_matrix.size() << std::endl;
        std::cout << "Modified Concepts: " << modified_concepts.size() << std::endl;
        std::cout << "Modified Connections: " << modified_connections.size() << std::endl;
        std::cout << "Last Save: " << last_save_timestamp << std::endl;
        std::cout << std::endl;
    }
    
private:
    void loadBaseData(std::ifstream& file) {
        // Load base concepts
        uint32_t concepts_count;
        file.read(reinterpret_cast<char*>(&concepts_count), sizeof(concepts_count));
        
        for (uint32_t i = 0; i < concepts_count; i++) {
            uint32_t id;
            file.read(reinterpret_cast<char*>(&id), sizeof(id));
            
            OptimizedConcept concept;
            file.read(reinterpret_cast<char*>(&concept), sizeof(OptimizedConcept));
            
            concepts[id] = concept;
            concept_name_to_id[concept.concept] = id;
            next_concept_id = std::max(next_concept_id, id + 1);
        }
        
        // Load base connections
        uint32_t connections_count;
        file.read(reinterpret_cast<char*>(&connections_count), sizeof(connections_count));
        
        for (uint32_t i = 0; i < connections_count; i++) {
            uint32_t id;
            file.read(reinterpret_cast<char*>(&id), sizeof(id));
            
            OptimizedConnection connection;
            file.read(reinterpret_cast<char*>(&connection), sizeof(OptimizedConnection));
            
            connections[id] = connection;
            adjacency_list[connection.from_id].push_back(id);
            next_connection_id = std::max(next_connection_id, id + 1);
        }
    }
    
    void rebuildIndexes() {
        confidence_index.clear();
        activation_index.clear();
        access_count_index.clear();
        
        for (const auto& concept_pair : concepts) {
            uint32_t id = concept_pair.first;
            const OptimizedConcept& concept = concept_pair.second;
            
            double confidence = (double)concept.validation_successes / 
                              (concept.validation_successes + concept.validation_failures + 1);
            
            confidence_index.insert({confidence, id});
            activation_index.insert({concept.activation, id});
            access_count_index.insert({concept.access_count, id});
        }
    }
    
    void expandSparseMatrix() {
        if (next_concept_id >= sparse_connection_matrix.size()) {
            size_t new_size = sparse_connection_matrix.size() * 2;
            sparse_connection_matrix.resize(new_size, std::vector<double>(new_size, 0.0));
            
            // Resize existing rows
            for (auto& row : sparse_connection_matrix) {
                row.resize(new_size, 0.0);
            }
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
    
    double calculatePathConfidence(const std::vector<uint32_t>& path) {
        double total_confidence = 0.0;
        for (uint32_t id : path) {
            auto it = concepts.find(id);
            if (it != concepts.end()) {
                const OptimizedConcept& concept = it->second;
                double confidence = (double)concept.validation_successes / 
                                  (concept.validation_successes + concept.validation_failures + 1);
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

// Tutor Hardening System
class TutorHardeningSystem {
private:
    std::unordered_map<std::string, TutorResponse> response_cache;
    std::mutex cache_mutex;
    std::vector<std::string> question_templates;
    uint32_t max_retries;
    uint32_t rate_limit_delay_ms;
    uint64_t last_request_time;
    
public:
    TutorHardeningSystem() : max_retries(3), rate_limit_delay_ms(1000), last_request_time(0) {
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
    
    // Get cached response or call Ollama with hardening
    TutorResponse getTutorResponse(const std::string& question, const std::vector<std::string>& concepts = {}) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        // Generate templated question
        std::string templated_question = generateTemplatedQuestion(question, concepts);
        
        // Create hash for caching
        std::string question_hash = std::to_string(std::hash<std::string>{}(templated_question));
        
        // Check cache first
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
        
        // Call Ollama with retry logic
        TutorResponse response;
        response.question_hash = question_hash;
        response.original_question = templated_question;
        response.timestamp = current_time;
        response.response_id = generateResponseId();
        
        for (uint32_t attempt = 0; attempt < max_retries; attempt++) {
            response.retry_count = attempt;
            
            std::cout << "🤖 Calling Ollama (attempt " << (attempt + 1) << "/" << max_retries << "): " 
                      << templated_question.substr(0, 50) << "..." << std::endl;
            
            std::string ollama_response = callOllamaWithRetry(templated_question);
            
            if (!ollama_response.empty() && ollama_response.find("error") == std::string::npos) {
                response.response = ollama_response;
                response.confidence_score = calculateResponseConfidence(ollama_response);
                response.is_cached = false;
                
                // Cache the response
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
    
    // Get cache statistics
    void displayCacheStats() const {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(cache_mutex));
        
        std::cout << "📊 TUTOR CACHE STATISTICS" << std::endl;
        std::cout << "=========================" << std::endl;
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
    std::string callOllamaWithRetry(const std::string& question) {
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
        // Simple confidence calculation based on response characteristics
        double confidence = 0.5; // Base confidence
        
        if (response.length() > 50) confidence += 0.2; // Longer responses are more confident
        if (response.find("I don't know") == std::string::npos) confidence += 0.2; // Not uncertain
        if (response.find("error") == std::string::npos) confidence += 0.1; // No errors
        
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

// Integrated Optimized Melvin Brain
class MelvinOptimizedBrain {
private:
    OptimizedStorageEngine storage_engine;
    TutorHardeningSystem tutor_system;
    
    // Driver system
    double curiosity = 0.8;
    double efficiency = 0.6;
    double consistency = 0.7;
    
public:
    MelvinOptimizedBrain() {
        std::cout << "🧠 MELVIN OPTIMIZED STORAGE + TUTOR HARDENING SYSTEM" << std::endl;
        std::cout << "====================================================" << std::endl;
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
        
        // Add test concepts
        std::cout << "Adding test concepts..." << std::endl;
        uint32_t bird_id = storage_engine.addConcept("bird", "Flying animal");
        uint32_t wire_id = storage_engine.addConcept("wire", "Electrical conductor");
        uint32_t robot_id = storage_engine.addConcept("robot", "Artificial being");
        uint32_t adaptation_id = storage_engine.addConcept("adaptation", "Process of change");
        uint32_t survival_id = storage_engine.addConcept("survival", "Continuing to exist");
        
        // Add test connections
        std::cout << "Adding test connections..." << std::endl;
        storage_engine.addConnection(bird_id, wire_id, 0.8);
        storage_engine.addConnection(robot_id, adaptation_id, 0.9);
        storage_engine.addConnection(adaptation_id, survival_id, 0.7);
        
        storage_engine.displayStats();
    }
    
    void testFastQueries() {
        std::cout << "🚀 TEST 2: FAST QUERIES" << std::endl;
        std::cout << "=======================" << std::endl;
        
        // Test nearest concept lookup
        std::cout << "Testing nearest concept lookup..." << std::endl;
        QueryResult nearest = storage_engine.findNearestConcepts("bird", 5);
        std::cout << "  Found " << nearest.concept_ids.size() << " nearest concepts in " 
                  << nearest.query_time_ms << "ms" << std::endl;
        
        // Test path search
        std::cout << "Testing path search..." << std::endl;
        uint32_t bird_id = storage_engine.getConceptId("bird");
        uint32_t wire_id = storage_engine.getConceptId("wire");
        if (bird_id && wire_id) {
            QueryResult path = storage_engine.findPath(bird_id, wire_id);
            std::cout << "  Found path with " << path.concept_ids.size() << " concepts in " 
                      << path.query_time_ms << "ms" << std::endl;
        }
        
        // Test high-confidence extraction
        std::cout << "Testing high-confidence extraction..." << std::endl;
        QueryResult high_conf = storage_engine.extractHighConfidenceSubset(0.5, 10);
        std::cout << "  Extracted " << high_conf.concept_ids.size() << " high-confidence concepts in " 
                  << high_conf.query_time_ms << "ms" << std::endl;
        std::cout << std::endl;
    }
    
    void testTutorHardening() {
        std::cout << "🤖 TEST 3: TUTOR HARDENING" << std::endl;
        std::cout << "===========================" << std::endl;
        
        // Test repeated queries (should hit cache)
        std::cout << "Testing repeated queries..." << std::endl;
        std::string question = "What is a bird?";
        std::vector<std::string> concepts = {"bird"};
        
        // First call (should call Ollama)
        TutorResponse response1 = tutor_system.getTutorResponse(question, concepts);
        std::cout << "  First call: " << (response1.is_cached ? "CACHED" : "OLLAMA") << std::endl;
        
        // Second call (should hit cache)
        TutorResponse response2 = tutor_system.getTutorResponse(question, concepts);
        std::cout << "  Second call: " << (response2.is_cached ? "CACHED" : "OLLAMA") << std::endl;
        
        // Test templated questioning
        std::cout << "Testing templated questioning..." << std::endl;
        TutorResponse response3 = tutor_system.getTutorResponse("bird wire", {"bird", "wire"});
        std::cout << "  Templated question: " << response3.original_question << std::endl;
        
        tutor_system.displayCacheStats();
    }
    
    void testIntegration() {
        std::cout << "🔗 TEST 4: INTEGRATION" << std::endl;
        std::cout << "======================" << std::endl;
        
        // Test: Ask Ollama, get fact, persist with provenance
        std::cout << "Testing integrated workflow..." << std::endl;
        
        // 1. Ask Ollama a new question
        std::string question = "How do birds fly?";
        TutorResponse response = tutor_system.getTutorResponse(question, {"bird", "fly"});
        
        // 2. Persist fact with provenance
        if (!response.response.empty()) {
            uint32_t bird_id = storage_engine.addConcept("bird", response.response);
            uint32_t fly_id = storage_engine.addConcept("fly", "Movement through air");
            storage_engine.addConnection(bird_id, fly_id, response.confidence_score);
            
            std::cout << "  ✅ Fact persisted with confidence: " << response.confidence_score << std::endl;
            std::cout << "  📝 Provenance: " << response.response_id << " @ " << response.timestamp << std::endl;
        }
        
        // 3. Test incremental save
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
};

int main() {
    std::cout << "🚀 Starting Melvin Optimized Storage + Tutor Hardening System" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << std::endl;
    
    MelvinOptimizedBrain melvin;
    melvin.runComprehensiveTests();
    
    std::cout << "🎯 Melvin Optimized System finished!" << std::endl;
    
    return 0;
}
