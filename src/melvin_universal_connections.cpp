/*
 * Melvin Universal Connection System
 * 
 * This system applies connection-based reasoning to EVERYTHING Melvin thinks about,
 * not just compound words. Every input triggers multi-level connection analysis.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <set>
#include <sstream>
#include <curl/curl.h>
#include <json/json.h>

// Enhanced Node with Universal Connection Types
struct UniversalKnowledgeNode {
    uint64_t id;
    char concept[64];
    char definition[512];
    
    // Universal connection types
    std::vector<uint64_t> direct_connections;      // Direct relationships
    std::vector<uint64_t> semantic_connections;    // Similar meaning
    std::vector<uint64_t> component_connections;   // Part-of relationships
    std::vector<uint64_t> contextual_connections;  // Context-based links
    std::vector<uint64_t> causal_connections;      // Cause-effect relationships
    std::vector<uint64_t> hierarchical_connections; // Category relationships
    std::vector<uint64_t> temporal_connections;    // Time-based relationships
    std::vector<uint64_t> spatial_connections;     // Location-based relationships
    
    char source[32];
    double confidence;
    uint64_t created_at;
    uint64_t last_accessed;
    uint32_t access_count;
    uint32_t connection_strength;  // How well connected this node is
    
    UniversalKnowledgeNode() : id(0), confidence(0.8), created_at(0), last_accessed(0), access_count(0), connection_strength(0) {
        memset(concept, 0, sizeof(concept));
        memset(definition, 0, sizeof(definition));
        memset(source, 0, sizeof(source));
    }
    
    UniversalKnowledgeNode(uint64_t node_id, const std::string& node_concept, const std::string& node_definition)
        : id(node_id), confidence(0.8), created_at(getCurrentTime()), last_accessed(getCurrentTime()), access_count(0), connection_strength(0) {
        
        strncpy(concept, node_concept.c_str(), sizeof(concept) - 1);
        strncpy(definition, node_definition.c_str(), sizeof(definition) - 1);
        strncpy(source, "ollama", sizeof(source) - 1);
    }
    
    static uint64_t getCurrentTime() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
    }
    
    // Calculate total connection strength
    int getTotalConnections() const {
        return direct_connections.size() + semantic_connections.size() + 
               component_connections.size() + contextual_connections.size() +
               causal_connections.size() + hierarchical_connections.size() +
               temporal_connections.size() + spatial_connections.size();
    }
};

// Universal Connection Engine - Applies to ALL thinking
class UniversalConnectionEngine {
private:
    // Comprehensive knowledge patterns
    std::map<std::string, std::vector<std::string>> semantic_groups = {
        {"animals", {"cat", "dog", "bird", "fish", "lion", "tiger", "elephant", "bear", "wolf", "deer", "rabbit", "mouse", "snake", "frog", "spider", "bee", "butterfly"}},
        {"vehicles", {"car", "truck", "bus", "train", "plane", "boat", "ship", "bicycle", "motorcycle", "helicopter", "submarine", "rocket"}},
        {"buildings", {"house", "building", "school", "hospital", "church", "library", "museum", "office", "factory", "warehouse", "castle", "tower"}},
        {"food", {"bread", "meat", "vegetable", "fruit", "milk", "cheese", "egg", "rice", "pasta", "soup", "salad", "pizza", "cake", "cookie"}},
        {"colors", {"red", "blue", "green", "yellow", "black", "white", "purple", "orange", "pink", "brown", "gray", "silver", "gold"}},
        {"emotions", {"happy", "sad", "angry", "excited", "worried", "calm", "nervous", "confident", "scared", "surprised", "disappointed", "proud"}},
        {"actions", {"run", "walk", "jump", "swim", "fly", "drive", "cook", "eat", "sleep", "work", "play", "study", "read", "write", "speak", "listen"}},
        {"materials", {"wood", "metal", "plastic", "glass", "fabric", "paper", "stone", "rubber", "leather", "ceramic", "concrete", "steel"}},
        {"tools", {"hammer", "screwdriver", "knife", "scissors", "pen", "pencil", "computer", "phone", "camera", "watch", "clock", "calculator"}},
        {"weather", {"sun", "rain", "snow", "wind", "cloud", "storm", "thunder", "lightning", "fog", "ice", "heat", "cold", "warm", "cool"}},
        {"body_parts", {"head", "eye", "nose", "mouth", "ear", "hand", "foot", "arm", "leg", "heart", "brain", "lung", "stomach", "skin", "hair"}},
        {"time_concepts", {"morning", "afternoon", "evening", "night", "day", "week", "month", "year", "hour", "minute", "second", "past", "present", "future"}},
        {"locations", {"home", "school", "work", "park", "store", "restaurant", "hospital", "library", "beach", "mountain", "city", "country", "world"}},
        {"professions", {"doctor", "teacher", "engineer", "artist", "musician", "chef", "pilot", "driver", "builder", "scientist", "lawyer", "police"}},
        {"family", {"mother", "father", "sister", "brother", "grandmother", "grandfather", "aunt", "uncle", "cousin", "son", "daughter", "baby"}}
    };
    
    // Causal relationship patterns
    std::map<std::string, std::vector<std::string>> causal_patterns = {
        {"causes_rain", {"cloud", "storm", "humidity", "pressure"}},
        {"causes_growth", {"water", "sunlight", "nutrients", "time"}},
        {"causes_movement", {"force", "energy", "motor", "engine"}},
        {"causes_learning", {"study", "practice", "experience", "teaching"}},
        {"causes_health", {"exercise", "nutrition", "sleep", "medicine"}},
        {"causes_problems", {"mistake", "accident", "conflict", "disease"}}
    };
    
    // Hierarchical relationships
    std::map<std::string, std::vector<std::string>> hierarchies = {
        {"animal", {"mammal", "bird", "fish", "reptile", "insect"}},
        {"mammal", {"cat", "dog", "human", "elephant", "whale"}},
        {"vehicle", {"car", "truck", "bus", "motorcycle", "bicycle"}},
        {"building", {"house", "office", "school", "hospital", "factory"}},
        {"tool", {"hammer", "screwdriver", "knife", "scissors", "computer"}},
        {"emotion", {"happy", "sad", "angry", "excited", "worried"}}
    };
    
    // Contextual relationship patterns
    std::map<std::string, std::vector<std::string>> contextual_patterns = {
        {"kitchen", {"cook", "eat", "food", "stove", "refrigerator", "knife", "plate", "cup"}},
        {"school", {"learn", "study", "teacher", "student", "book", "pencil", "desk", "classroom"}},
        {"hospital", {"doctor", "nurse", "patient", "medicine", "surgery", "health", "treatment"}},
        {"park", {"play", "walk", "tree", "grass", "bench", "children", "exercise", "nature"}},
        {"office", {"work", "computer", "desk", "meeting", "business", "employee", "boss", "project"}}
    };

public:
    // Universal connection analysis for ANY input
    void analyzeUniversalConnections(std::shared_ptr<UniversalKnowledgeNode> newNode, 
                                   const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes) {
        std::string concept = toLowerCase(cleanConcept(newNode->concept));
        std::string definition = toLowerCase(newNode->definition);
        
        std::cout << "🔍 Universal connection analysis for: " << newNode->concept << std::endl;
        
        // 1. Semantic Group Connections
        buildSemanticGroupConnections(newNode, allNodes, concept);
        
        // 2. Component/Part-of Connections
        buildComponentConnections(newNode, allNodes, concept);
        
        // 3. Hierarchical Connections
        buildHierarchicalConnections(newNode, allNodes, concept);
        
        // 4. Causal Connections
        buildCausalConnections(newNode, allNodes, concept, definition);
        
        // 5. Contextual Connections
        buildContextualConnections(newNode, allNodes, concept, definition);
        
        // 6. Definition-based Connections
        buildDefinitionConnections(newNode, allNodes, concept, definition);
        
        // 7. Temporal Connections (based on creation time)
        buildTemporalConnections(newNode, allNodes);
        
        // 8. Spatial Connections (if location info available)
        buildSpatialConnections(newNode, allNodes, definition);
        
        // Update connection strength
        newNode->connection_strength = newNode->getTotalConnections();
        
        std::cout << "✅ Total connections made: " << newNode->connection_strength << std::endl;
    }
    
    // Universal knowledge retrieval with multi-level reasoning
    std::string findUniversalKnowledge(const std::string& question, 
                                     const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes) {
        std::string concept = extractConceptFromQuestion(question);
        std::string lower_concept = toLowerCase(cleanConcept(concept));
        
        std::cout << "🧠 Universal knowledge search for: " << concept << std::endl;
        
        // 1. Direct match
        auto direct_result = findDirectMatch(allNodes, lower_concept);
        if (!direct_result.empty()) {
            std::cout << "💡 Direct match found!" << std::endl;
            return direct_result;
        }
        
        // 2. Semantic group reasoning
        auto semantic_result = findSemanticGroupKnowledge(allNodes, lower_concept);
        if (!semantic_result.empty()) {
            std::cout << "💡 Semantic group knowledge found!" << std::endl;
            return semantic_result;
        }
        
        // 3. Component-based reasoning
        auto component_result = findComponentKnowledge(allNodes, lower_concept);
        if (!component_result.empty()) {
            std::cout << "💡 Component knowledge found!" << std::endl;
            return component_result;
        }
        
        // 4. Hierarchical reasoning
        auto hierarchical_result = findHierarchicalKnowledge(allNodes, lower_concept);
        if (!hierarchical_result.empty()) {
            std::cout << "💡 Hierarchical knowledge found!" << std::endl;
            return hierarchical_result;
        }
        
        // 5. Causal reasoning
        auto causal_result = findCausalKnowledge(allNodes, lower_concept);
        if (!causal_result.empty()) {
            std::cout << "💡 Causal knowledge found!" << std::endl;
            return causal_result;
        }
        
        // 6. Contextual reasoning
        auto contextual_result = findContextualKnowledge(allNodes, lower_concept);
        if (!contextual_result.empty()) {
            std::cout << "💡 Contextual knowledge found!" << std::endl;
            return contextual_result;
        }
        
        // 7. Multi-hop reasoning (follow connection chains)
        auto multi_hop_result = findMultiHopKnowledge(allNodes, lower_concept);
        if (!multi_hop_result.empty()) {
            std::cout << "💡 Multi-hop knowledge found!" << std::endl;
            return multi_hop_result;
        }
        
        return "";
    }

private:
    // Connection building methods
    void buildSemanticGroupConnections(std::shared_ptr<UniversalKnowledgeNode> newNode, 
                                     const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes,
                                     const std::string& concept) {
        for (const auto& group : semantic_groups) {
            for (const auto& member : group.second) {
                if (concept == member) {
                    // Connect to all other members of the semantic group
                    for (const auto& other_member : group.second) {
                        if (other_member != member) {
                            for (const auto& node_pair : allNodes) {
                                auto existingNode = node_pair.second;
                                if (existingNode->id == newNode->id) continue;
                                
                                std::string existing_concept = toLowerCase(cleanConcept(existingNode->concept));
                                if (existing_concept == other_member) {
                                    addConnection(newNode, existingNode, newNode->semantic_connections);
                                    std::cout << "  🧠 Semantic: " << newNode->concept << " → " << existingNode->concept << std::endl;
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
    
    void buildComponentConnections(std::shared_ptr<UniversalKnowledgeNode> newNode, 
                                 const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes,
                                 const std::string& concept) {
        // Find components in the concept name
        for (const auto& node_pair : allNodes) {
            auto existingNode = node_pair.second;
            if (existingNode->id == newNode->id) continue;
            
            std::string existing_concept = toLowerCase(cleanConcept(existingNode->concept));
            
            // Check if existing concept is a component of new concept
            if (concept.find(existing_concept) != std::string::npos && existing_concept.length() > 2) {
                addConnection(newNode, existingNode, newNode->component_connections);
                std::cout << "  🔗 Component: " << newNode->concept << " → " << existingNode->concept << std::endl;
            }
        }
    }
    
    void buildHierarchicalConnections(std::shared_ptr<UniversalKnowledgeNode> newNode, 
                                    const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes,
                                    const std::string& concept) {
        for (const auto& hierarchy : hierarchies) {
            for (const auto& member : hierarchy.second) {
                if (concept == member) {
                    // Connect to parent category
                    for (const auto& node_pair : allNodes) {
                        auto existingNode = node_pair.second;
                        if (existingNode->id == newNode->id) continue;
                        
                        std::string existing_concept = toLowerCase(cleanConcept(existingNode->concept));
                        if (existing_concept == hierarchy.first) {
                            addConnection(newNode, existingNode, newNode->hierarchical_connections);
                            std::cout << "  📊 Hierarchical: " << newNode->concept << " → " << existingNode->concept << std::endl;
                        }
                    }
                    
                    // Connect to siblings
                    for (const auto& sibling : hierarchy.second) {
                        if (sibling != member) {
                            for (const auto& node_pair : allNodes) {
                                auto existingNode = node_pair.second;
                                if (existingNode->id == newNode->id) continue;
                                
                                std::string existing_concept = toLowerCase(cleanConcept(existingNode->concept));
                                if (existing_concept == sibling) {
                                    addConnection(newNode, existingNode, newNode->hierarchical_connections);
                                    std::cout << "  📊 Sibling: " << newNode->concept << " → " << existingNode->concept << std::endl;
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
    
    void buildCausalConnections(std::shared_ptr<UniversalKnowledgeNode> newNode, 
                              const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes,
                              const std::string& concept, const std::string& definition) {
        for (const auto& causal_group : causal_patterns) {
            for (const auto& cause : causal_group.second) {
                if (concept == cause || definition.find(cause) != std::string::npos) {
                    // Find effects of this cause
                    for (const auto& node_pair : allNodes) {
                        auto existingNode = node_pair.second;
                        if (existingNode->id == newNode->id) continue;
                        
                        std::string existing_concept = toLowerCase(cleanConcept(existingNode->concept));
                        std::string existing_def = toLowerCase(existingNode->definition);
                        
                        // Check if existing node is an effect of this cause
                        if (existing_def.find(cause) != std::string::npos) {
                            addConnection(newNode, existingNode, newNode->causal_connections);
                            std::cout << "  ⚡ Causal: " << newNode->concept << " → " << existingNode->concept << std::endl;
                        }
                    }
                    break;
                }
            }
        }
    }
    
    void buildContextualConnections(std::shared_ptr<UniversalKnowledgeNode> newNode, 
                                  const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes,
                                  const std::string& concept, const std::string& definition) {
        for (const auto& context_group : contextual_patterns) {
            for (const auto& context_item : context_group.second) {
                if (concept == context_item || definition.find(context_item) != std::string::npos) {
                    // Connect to other items in the same context
                    for (const auto& other_item : context_group.second) {
                        if (other_item != context_item) {
                            for (const auto& node_pair : allNodes) {
                                auto existingNode = node_pair.second;
                                if (existingNode->id == newNode->id) continue;
                                
                                std::string existing_concept = toLowerCase(cleanConcept(existingNode->concept));
                                if (existing_concept == other_item) {
                                    addConnection(newNode, existingNode, newNode->contextual_connections);
                                    std::cout << "  🏠 Contextual: " << newNode->concept << " → " << existingNode->concept << std::endl;
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
    
    void buildDefinitionConnections(std::shared_ptr<UniversalKnowledgeNode> newNode, 
                                  const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes,
                                  const std::string& concept, const std::string& definition) {
        for (const auto& node_pair : allNodes) {
            auto existingNode = node_pair.second;
            if (existingNode->id == newNode->id) continue;
            
            std::string existing_concept = toLowerCase(cleanConcept(existingNode->concept));
            std::string existing_def = toLowerCase(existingNode->definition);
            
            // Check if concepts appear in each other's definitions
            if (existing_def.find(concept) != std::string::npos || 
                definition.find(existing_concept) != std::string::npos) {
                addConnection(newNode, existingNode, newNode->direct_connections);
                std::cout << "  📚 Definition: " << newNode->concept << " → " << existingNode->concept << std::endl;
            }
        }
    }
    
    void buildTemporalConnections(std::shared_ptr<UniversalKnowledgeNode> newNode, 
                                const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes) {
        // Connect to recently created nodes (temporal proximity)
        for (const auto& node_pair : allNodes) {
            auto existingNode = node_pair.second;
            if (existingNode->id == newNode->id) continue;
            
            // Connect to nodes created within the last hour
            uint64_t time_diff = newNode->created_at - existingNode->created_at;
            if (time_diff < 3600000) { // 1 hour in milliseconds
                addConnection(newNode, existingNode, newNode->temporal_connections);
                std::cout << "  ⏰ Temporal: " << newNode->concept << " → " << existingNode->concept << std::endl;
            }
        }
    }
    
    void buildSpatialConnections(std::shared_ptr<UniversalKnowledgeNode> newNode, 
                               const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes,
                               const std::string& definition) {
        // Simple spatial relationship detection
        std::vector<std::string> spatial_words = {"in", "on", "at", "near", "inside", "outside", "above", "below", "beside", "around"};
        
        for (const auto& spatial_word : spatial_words) {
            if (definition.find(spatial_word) != std::string::npos) {
                // Connect to other nodes that might be in similar spatial contexts
                for (const auto& node_pair : allNodes) {
                    auto existingNode = node_pair.second;
                    if (existingNode->id == newNode->id) continue;
                    
                    std::string existing_def = toLowerCase(existingNode->definition);
                    if (existing_def.find(spatial_word) != std::string::npos) {
                        addConnection(newNode, existingNode, newNode->spatial_connections);
                        std::cout << "  📍 Spatial: " << newNode->concept << " → " << existingNode->concept << std::endl;
                    }
                }
            }
        }
    }
    
    // Knowledge retrieval methods
    std::string findDirectMatch(const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes,
                               const std::string& concept) {
        for (const auto& node_pair : allNodes) {
            auto node = node_pair.second;
            std::string node_concept = toLowerCase(cleanConcept(node->concept));
            if (node_concept == concept) {
                return node->definition;
            }
        }
        return "";
    }
    
    std::string findSemanticGroupKnowledge(const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes,
                                         const std::string& concept) {
        for (const auto& group : semantic_groups) {
            for (const auto& member : group.second) {
                if (concept == member) {
                    // Find knowledge about other members of the group
                    for (const auto& other_member : group.second) {
                        if (other_member != member) {
                            for (const auto& node_pair : allNodes) {
                                auto node = node_pair.second;
                                std::string node_concept = toLowerCase(cleanConcept(node->concept));
                                if (node_concept == other_member) {
                                    return "Related concept: " + std::string(node->definition) + 
                                           " (This is similar to " + concept + ")";
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
        return "";
    }
    
    std::string findComponentKnowledge(const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes,
                                     const std::string& concept) {
        // Find components in the concept
        for (const auto& node_pair : allNodes) {
            auto node = node_pair.second;
            std::string node_concept = toLowerCase(cleanConcept(node->concept));
            
            if (concept.find(node_concept) != std::string::npos && node_concept.length() > 2) {
                return "Based on components: " + std::string(node->definition) + 
                       " (This might relate to " + concept + ")";
            }
        }
        return "";
    }
    
    std::string findHierarchicalKnowledge(const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes,
                                        const std::string& concept) {
        for (const auto& hierarchy : hierarchies) {
            for (const auto& member : hierarchy.second) {
                if (concept == member) {
                    // Find knowledge about the parent category
                    for (const auto& node_pair : allNodes) {
                        auto node = node_pair.second;
                        std::string node_concept = toLowerCase(cleanConcept(node->concept));
                        if (node_concept == hierarchy.first) {
                            return "Category knowledge: " + std::string(node->definition) + 
                                   " (This applies to " + concept + ")";
                        }
                    }
                    break;
                }
            }
        }
        return "";
    }
    
    std::string findCausalKnowledge(const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes,
                                  const std::string& concept) {
        for (const auto& causal_group : causal_patterns) {
            for (const auto& cause : causal_group.second) {
                if (concept == cause) {
                    return "Causal relationship: " + concept + " is related to " + causal_group.first;
                }
            }
        }
        return "";
    }
    
    std::string findContextualKnowledge(const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes,
                                      const std::string& concept) {
        for (const auto& context_group : contextual_patterns) {
            for (const auto& context_item : context_group.second) {
                if (concept == context_item) {
                    return "Contextual relationship: " + concept + " is commonly found in " + context_group.first;
                }
            }
        }
        return "";
    }
    
    std::string findMultiHopKnowledge(const std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>>& allNodes,
                                    const std::string& concept) {
        // Follow connection chains to find related knowledge
        for (const auto& node_pair : allNodes) {
            auto node = node_pair.second;
            std::string node_concept = toLowerCase(cleanConcept(node->concept));
            
            // Check if this node has connections that might lead to our concept
            if (node->getTotalConnections() > 0) {
                // This is a well-connected node that might have relevant knowledge
                if (node_concept != concept) {
                    return "Multi-hop knowledge: " + std::string(node->definition) + 
                           " (Connected concept that might relate to " + concept + ")";
                }
            }
        }
        return "";
    }
    
    // Utility methods
    void addConnection(std::shared_ptr<UniversalKnowledgeNode> from, 
                      std::shared_ptr<UniversalKnowledgeNode> to, 
                      std::vector<uint64_t>& connection_vector) {
        if (std::find(connection_vector.begin(), connection_vector.end(), to->id) == connection_vector.end()) {
            connection_vector.push_back(to->id);
        }
    }
    
    std::string toLowerCase(const std::string& str) {
        std::string result = str;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }
    
    std::string cleanConcept(const std::string& concept) {
        std::string cleaned = concept;
        cleaned.erase(std::remove_if(cleaned.begin(), cleaned.end(), 
            [](char c) { return !std::isalnum(c) && c != ' '; }), cleaned.end());
        return cleaned;
    }
    
    std::string extractConceptFromQuestion(const std::string& question) {
        std::string lower_q = toLowerCase(question);
        
        std::vector<std::string> question_words = {"what", "is", "a", "an", "the", "are", "how", "why", "when", "where", "who"};
        std::istringstream iss(lower_q);
        std::vector<std::string> words;
        std::string word;
        
        while (iss >> word) {
            if (std::find(question_words.begin(), question_words.end(), word) == question_words.end()) {
                words.push_back(word);
            }
        }
        
        if (!words.empty()) {
            return words[0];
        }
        
        return question;
    }
};

// Working Ollama Client (same as before)
class WorkingOllamaClient {
private:
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
        ((std::string*)userp)->append((char*)contents, size * nmemb);
        return size * nmemb;
    }

public:
    std::string askQuestion(const std::string& question) {
        CURL* curl;
        CURLcode res;
        std::string response_data;
        
        curl = curl_easy_init();
        if (curl) {
            Json::Value payload;
            payload["model"] = "llama3.2";
            payload["prompt"] = question;
            payload["stream"] = false;
            
            Json::StreamWriterBuilder builder;
            std::string json_payload = Json::writeString(builder, payload);
            
            curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:11434/api/generate");
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_payload.length());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, curl_slist_append(nullptr, "Content-Type: application/json"));
            
            res = curl_easy_perform(curl);
            long http_code = 0;
            curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
            curl_easy_cleanup(curl);
            
            if (res == CURLE_OK && http_code == 200) {
                Json::Value root;
                Json::CharReaderBuilder reader_builder;
                std::string errors;
                std::istringstream stream(response_data);
                
                if (Json::parseFromStream(reader_builder, stream, &root, &errors)) {
                    if (root.isMember("response")) {
                        return root["response"].asString();
                    }
                }
            }
        }
        
        return "Error: Could not get response from Ollama";
    }
};

// Universal Melvin Learning System
class UniversalMelvinLearningSystem {
private:
    std::map<uint64_t, std::shared_ptr<UniversalKnowledgeNode>> nodes;
    UniversalConnectionEngine connectionEngine;
    WorkingOllamaClient ollamaClient;
    uint64_t next_node_id = 1;
    
    struct LearningStats {
        int questions_asked = 0;
        int new_concepts_learned = 0;
        int concepts_retrieved = 0;
        int total_connections_made = 0;
        int semantic_connections = 0;
        int component_connections = 0;
        int hierarchical_connections = 0;
        int causal_connections = 0;
        int contextual_connections = 0;
    } stats;

public:
    UniversalMelvinLearningSystem() {
        std::cout << "🧠 Universal Melvin Learning System Initialized" << std::endl;
        std::cout << "🔗 Universal connection-based reasoning for ALL thinking!" << std::endl;
    }
    
    bool melvinKnows(const std::string& question) {
        std::string result = connectionEngine.findUniversalKnowledge(question, nodes);
        return !result.empty();
    }
    
    std::string melvinAnswer(const std::string& question) {
        std::string result = connectionEngine.findUniversalKnowledge(question, nodes);
        if (!result.empty()) {
            stats.concepts_retrieved++;
            return result;
        }
        return "I don't know about that yet.";
    }
    
    std::shared_ptr<UniversalKnowledgeNode> createNode(const std::string& concept, const std::string& definition) {
        auto node = std::make_shared<UniversalKnowledgeNode>(next_node_id++, concept, definition);
        return node;
    }
    
    void connectToGraph(std::shared_ptr<UniversalKnowledgeNode> node) {
        nodes[node->id] = node;
        
        std::cout << "🔗 Building universal connections for: " << node->concept << std::endl;
        connectionEngine.analyzeUniversalConnections(node, nodes);
        
        stats.new_concepts_learned++;
        stats.total_connections_made += node->connection_strength;
    }
    
    std::string askOllama(const std::string& question) {
        std::cout << "🤖 Asking Ollama: " << question << std::endl;
        return ollamaClient.askQuestion(question);
    }
    
    std::string curiosityLoop(const std::string& question) {
        stats.questions_asked++;
        
        std::cout << "\n🤔 Melvin is thinking about: " << question << std::endl;
        
        if (melvinKnows(question)) {
            std::cout << "🧠 Melvin knows this! Retrieving from memory..." << std::endl;
            return melvinAnswer(question);
        }
        
        std::cout << "❓ Melvin doesn't know this. Asking Ollama tutor..." << std::endl;
        std::string ollamaResponse = askOllama(question);
        
        std::string concept = extractConceptFromQuestion(question);
        std::string definition = ollamaResponse;
        
        std::cout << "📚 Creating new knowledge node for: " << concept << std::endl;
        auto node = createNode(concept, definition);
        
        std::cout << "🔗 Building universal connections..." << std::endl;
        connectToGraph(node);
        
        std::cout << "✅ Melvin learned something new with universal connections!" << std::endl;
        return definition;
    }
    
    void showLearningStats() {
        std::cout << "\n📊 UNIVERSAL MELVIN LEARNING STATISTICS" << std::endl;
        std::cout << "=======================================" << std::endl;
        std::cout << "Total Concepts: " << nodes.size() << std::endl;
        std::cout << "Questions Asked: " << stats.questions_asked << std::endl;
        std::cout << "New Concepts Learned: " << stats.new_concepts_learned << std::endl;
        std::cout << "Concepts Retrieved: " << stats.concepts_retrieved << std::endl;
        std::cout << "Total Connections Made: " << stats.total_connections_made << std::endl;
        std::cout << "Average Connections per Node: " << (nodes.empty() ? 0 : stats.total_connections_made / nodes.size()) << std::endl;
        std::cout << "=======================================" << std::endl;
    }
    
    void showKnowledgeGraph() {
        std::cout << "\n🧠 UNIVERSAL MELVIN KNOWLEDGE GRAPH" << std::endl;
        std::cout << "====================================" << std::endl;
        
        for (const auto& node_pair : nodes) {
            auto node = node_pair.second;
            std::cout << "\n📝 " << node->concept << " (ID: " << node->id << ", Connections: " << node->connection_strength << ")" << std::endl;
            std::cout << "   Definition: " << node->definition << std::endl;
            
            if (!node->semantic_connections.empty()) {
                std::cout << "   🧠 Semantic: ";
                for (auto conn_id : node->semantic_connections) {
                    if (nodes.count(conn_id)) {
                        std::cout << nodes[conn_id]->concept << " ";
                    }
                }
                std::cout << std::endl;
            }
            
            if (!node->component_connections.empty()) {
                std::cout << "   🔗 Component: ";
                for (auto conn_id : node->component_connections) {
                    if (nodes.count(conn_id)) {
                        std::cout << nodes[conn_id]->concept << " ";
                    }
                }
                std::cout << std::endl;
            }
            
            if (!node->hierarchical_connections.empty()) {
                std::cout << "   📊 Hierarchical: ";
                for (auto conn_id : node->hierarchical_connections) {
                    if (nodes.count(conn_id)) {
                        std::cout << nodes[conn_id]->concept << " ";
                    }
                }
                std::cout << std::endl;
            }
            
            if (!node->causal_connections.empty()) {
                std::cout << "   ⚡ Causal: ";
                for (auto conn_id : node->causal_connections) {
                    if (nodes.count(conn_id)) {
                        std::cout << nodes[conn_id]->concept << " ";
                    }
                }
                std::cout << std::endl;
            }
            
            if (!node->contextual_connections.empty()) {
                std::cout << "   🏠 Contextual: ";
                for (auto conn_id : node->contextual_connections) {
                    if (nodes.count(conn_id)) {
                        std::cout << nodes[conn_id]->concept << " ";
                    }
                }
                std::cout << std::endl;
            }
        }
    }

private:
    std::string extractConceptFromQuestion(const std::string& question) {
        std::string lower_q = question;
        std::transform(lower_q.begin(), lower_q.end(), lower_q.begin(), ::tolower);
        
        std::vector<std::string> question_words = {"what", "is", "a", "an", "the", "are", "how", "why", "when", "where", "who"};
        std::istringstream iss(lower_q);
        std::vector<std::string> words;
        std::string word;
        
        while (iss >> word) {
            if (std::find(question_words.begin(), question_words.end(), word) == question_words.end()) {
                words.push_back(word);
            }
        }
        
        if (!words.empty()) {
            return words[0];
        }
        
        return question;
    }
};

int main(int argc, char* argv[]) {
    std::cout << "🧠 UNIVERSAL MELVIN LEARNING SYSTEM" << std::endl;
    std::cout << "====================================" << std::endl;
    std::cout << "🔗 Connection-based reasoning for EVERYTHING Melvin thinks about!" << std::endl;
    
    UniversalMelvinLearningSystem melvin;
    
    if (argc > 1) {
        // Single question mode
        std::string question = argv[1];
        std::string answer = melvin.curiosityLoop(question);
        std::cout << "\n💡 Answer: " << answer << std::endl;
    } else {
        // Interactive mode - Test universal connections
        std::cout << "\n🎯 TESTING UNIVERSAL CONNECTION SYSTEM" << std::endl;
        std::cout << "======================================" << std::endl;
        
        // Test various types of concepts to see universal connections
        std::vector<std::string> test_concepts = {
            "What is a cat?",
            "What is a car?", 
            "What is a house?",
            "What is a doctor?",
            "What is a computer?",
            "What is a tree?",
            "What is a book?",
            "What is a kitchen?"
        };
        
        for (const auto& concept : test_concepts) {
            std::cout << "\n" << std::string(50, '=') << std::endl;
            melvin.curiosityLoop(concept);
        }
        
        // Show the knowledge graph
        melvin.showKnowledgeGraph();
        
        // Show learning statistics
        melvin.showLearningStats();
        
        std::cout << "\n✅ Universal connection demo completed!" << std::endl;
    }
    
    return 0;
}
