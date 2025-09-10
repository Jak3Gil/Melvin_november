#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <chrono>
#include <atomic>
#include <mutex>
#include <random>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <thread>
#include <queue>
#include <memory>

// Debug version of Melvin to inspect nodes
class MelvinDebug {
private:
    struct Node {
        uint64_t id;
        std::string content;
        double activation_strength;
        uint64_t creation_time;
        uint64_t last_access_time;
        uint32_t access_count;
        double confidence_score;
        std::string source;
        std::string nonce;
        uint64_t timestamp;
        std::vector<uint64_t> connections;
        bool oracle_used = false;
        std::string content_type = "TEXT";
        std::string compression_type = "NONE";
        uint8_t importance = 5;
        uint32_t content_length = 0;
        uint32_t connection_count = 0;
        
        Node(uint64_t node_id, const std::string& node_content, const std::string& node_source = "internal")
            : id(node_id), content(node_content), activation_strength(1.0), 
              creation_time(getCurrentTime()), last_access_time(getCurrentTime()),
              access_count(1), confidence_score(0.5), source(node_source),
              nonce(generateNonce()), timestamp(getCurrentTime()), content_length(node_content.length()) {}
    };
    
    std::unordered_map<uint64_t, std::shared_ptr<Node>> nodes;
    mutable std::mutex nodes_mutex;
    std::atomic<uint64_t> next_node_id{1};
    
    static uint64_t getCurrentTime() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    
    static std::string generateNonce() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(100000, 999999);
        return std::to_string(dis(gen));
    }
    
public:
    uint64_t createNode(const std::string& content, const std::string& source = "internal") {
        uint64_t id = next_node_id++;
        auto node = std::make_shared<Node>(id, content, source);
        
        std::lock_guard<std::mutex> lock(nodes_mutex);
        nodes[id] = node;
        
        std::cout << "🔍 NODE CREATED:" << std::endl;
        std::cout << "   ID: " << node->id << std::endl;
        std::cout << "   Content: \"" << node->content << "\"" << std::endl;
        std::cout << "   Source: " << node->source << std::endl;
        std::cout << "   Nonce: " << node->nonce << std::endl;
        std::cout << "   Timestamp: " << node->timestamp << std::endl;
        std::cout << "   Content Length: " << node->content_length << std::endl;
        std::cout << "   Confidence: " << node->confidence_score << std::endl;
        std::cout << "   Activation: " << node->activation_strength << std::endl;
        std::cout << "   Importance: " << (int)node->importance << std::endl;
        std::cout << "   Oracle Used: " << (node->oracle_used ? "Yes" : "No") << std::endl;
        std::cout << "   Connections: " << node->connections.size() << std::endl;
        std::cout << "---" << std::endl;
        
        return id;
    }
    
    void printAllNodes() {
        std::lock_guard<std::mutex> lock(nodes_mutex);
        
        std::cout << "\n🔍 ALL NODES IN MEMORY:" << std::endl;
        std::cout << "=======================" << std::endl;
        std::cout << "Total Nodes: " << nodes.size() << std::endl;
        std::cout << std::endl;
        
        for (const auto& [id, node] : nodes) {
            std::cout << "Node ID: " << node->id << std::endl;
            std::cout << "  Content: \"" << node->content << "\"" << std::endl;
            std::cout << "  Source: " << node->source << std::endl;
            std::cout << "  Nonce: " << node->nonce << std::endl;
            std::cout << "  Timestamp: " << node->timestamp << std::endl;
            std::cout << "  Content Length: " << node->content_length << std::endl;
            std::cout << "  Confidence: " << node->confidence_score << std::endl;
            std::cout << "  Activation: " << node->activation_strength << std::endl;
            std::cout << "  Importance: " << (int)node->importance << std::endl;
            std::cout << "  Oracle Used: " << (node->oracle_used ? "Yes" : "No") << std::endl;
            std::cout << "  Connections: " << node->connections.size() << std::endl;
            std::cout << "  Access Count: " << node->access_count << std::endl;
            std::cout << "  Last Access: " << node->last_access_time << std::endl;
            std::cout << "---" << std::endl;
        }
    }
    
    size_t getNodeCount() const {
        std::lock_guard<std::mutex> lock(nodes_mutex);
        return nodes.size();
    }
    
    std::string ask(const std::string& question) {
        std::cout << "\n🧠 PROCESSING QUESTION: " << question << std::endl;
        
        // Create input node
        uint64_t input_node = createNode(question, "user_input");
        
        // Generate response
        std::string answer = "Melvin processing [WHAT]: " + question;
        
        // Create response node
        uint64_t response_node = createNode(answer, "melvin_response");
        
        return answer;
    }
};

int main() {
    std::cout << "🔍 Melvin Debug - Node Inspector" << std::endl;
    std::cout << "================================" << std::endl;
    
    MelvinDebug melvin;
    
    std::cout << "\n🚀 Testing node creation..." << std::endl;
    
    // Test questions
    std::string response1 = melvin.ask("What is 2 + 2?");
    std::string response2 = melvin.ask("What is the capital of France?");
    
    std::cout << "\n📊 FINAL STATUS:" << std::endl;
    std::cout << "Total Nodes Created: " << melvin.getNodeCount() << std::endl;
    
    // Print all nodes
    melvin.printAllNodes();
    
    return 0;
}
