#pragma once

#include "melvin_common.hpp"
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <memory>

namespace melvin {

// Forward declarations
class Node;
class Connection;
class NodeManager;
class ConnectionManager;
class GraphPersistence;

// Node data structure
struct NodeData {
    NodeID id;
    NodeType type;
    std::string name;
    std::string description;
    std::map<std::string, std::string> attributes;
    Confidence confidence;
    Priority priority;
    TimePoint created_at;
    TimePoint last_updated;
    bool is_active;
    
    NodeData() : id(0), type(NodeType::CONCEPT), confidence(0.5f), priority(128), 
                 is_active(true) {}
};

// Connection data structure
struct ConnectionData {
    ConnectionID id;
    NodeID from_node;
    NodeID to_node;
    ConnectionType type;
    Weight weight;
    Confidence confidence;
    std::string description;
    std::map<std::string, std::string> attributes;
    TimePoint created_at;
    TimePoint last_updated;
    bool is_active;
    
    ConnectionData() : id(0), from_node(0), to_node(0), type(ConnectionType::EXCITATORY),
                       weight(1.0f), confidence(0.5f), is_active(true) {}
};

// Brain graph query result
struct GraphQueryResult {
    std::vector<NodeData> nodes;
    std::vector<ConnectionData> connections;
    size_t total_nodes;
    size_t total_connections;
    Duration query_time;
    
    GraphQueryResult() : total_nodes(0), total_connections(0) {}
};

// Brain graph statistics
struct GraphStatistics {
    size_t total_nodes;
    size_t active_nodes;
    size_t total_connections;
    size_t active_connections;
    std::map<NodeType, size_t> nodes_by_type;
    std::map<ConnectionType, size_t> connections_by_type;
    double average_confidence;
    double average_weight;
    TimePoint last_calculation;
    
    GraphStatistics() : total_nodes(0), active_nodes(0), total_connections(0), 
                       active_connections(0), average_confidence(0.0), average_weight(0.0) {}
};

// Brain graph configuration
struct GraphConfig {
    size_t max_nodes;
    size_t max_connections;
    size_t max_connections_per_node;
    double min_confidence_threshold;
    double max_weight;
    Duration node_timeout;
    Duration connection_timeout;
    bool enable_auto_cleanup;
    bool enable_confidence_decay;
    double confidence_decay_rate;
    
    GraphConfig() : max_nodes(MAX_NODES), max_connections(MAX_CONNECTIONS),
                   max_connections_per_node(100), min_confidence_threshold(0.1),
                   max_weight(10.0), node_timeout(Minutes(30)), connection_timeout(Minutes(60)),
                   enable_auto_cleanup(true), enable_confidence_decay(true), confidence_decay_rate(0.95) {}
};

// Brain Graph - Main cognitive architecture
class BrainGraph {
public:
    static BrainGraph& instance();
    
    // Initialization and shutdown
    Result<void> init(const GraphConfig& config = GraphConfig());
    Result<void> shutdown();
    
    // Node management
    Result<NodeID> create_node(const NodeData& data);
    Result<void> update_node(NodeID id, const NodeData& data);
    Result<void> delete_node(NodeID id);
    Result<NodeData> get_node(NodeID id) const;
    Result<std::vector<NodeData>> get_nodes_by_type(NodeType type) const;
    Result<std::vector<NodeData>> search_nodes(const std::string& query) const;
    
    // Connection management
    Result<ConnectionID> create_connection(const ConnectionData& data);
    Result<void> update_connection(ConnectionID id, const ConnectionData& data);
    Result<void> delete_connection(ConnectionID id);
    Result<ConnectionData> get_connection(ConnectionID id) const;
    Result<std::vector<ConnectionData>> get_connections_from(NodeID node_id) const;
    Result<std::vector<ConnectionData>> get_connections_to(NodeID node_id) const;
    Result<std::vector<ConnectionData>> get_connections_between(NodeID from, NodeID to) const;
    
    // Graph operations
    Result<GraphQueryResult> query_graph(const std::string& query, size_t limit = 100) const;
    Result<std::vector<NodeID>> find_path(NodeID start, NodeID end, size_t max_depth = 10) const;
    Result<std::vector<NodeID>> get_neighbors(NodeID node_id, size_t depth = 1) const;
    Result<double> calculate_similarity(NodeID node1, NodeID node2) const;
    
    // Learning and adaptation
    Result<void> update_confidence(NodeID node_id, Confidence new_confidence);
    Result<void> update_weight(ConnectionID connection_id, Weight new_weight);
    Result<void> strengthen_connection(ConnectionID connection_id);
    Result<void> weaken_connection(ConnectionID connection_id);
    Result<void> decay_confidence();
    Result<void> cleanup_old_connections();
    
    // Graph analysis
    GraphStatistics get_statistics() const;
    Result<std::vector<NodeID>> get_most_connected_nodes(size_t count = 10) const;
    Result<std::vector<NodeID>> get_most_confident_nodes(size_t count = 10) const;
    Result<std::vector<ConnectionID>> get_strongest_connections(size_t count = 10) const;
    
    // Persistence
    Result<void> save_graph(const std::string& filename);
    Result<void> load_graph(const std::string& filename);
    Result<void> export_graph(const std::string& filename, const std::string& format);
    
    // Event callbacks
    using NodeEventCallback = std::function<void(NodeID, const std::string&)>;
    using ConnectionEventCallback = std::function<void(ConnectionID, const std::string&)>;
    using GraphEventCallback = std::function<void(const std::string&)>;
    
    void set_node_event_callback(NodeEventCallback callback);
    void set_connection_event_callback(ConnectionEventCallback callback);
    void set_graph_event_callback(GraphEventCallback callback);
    
    // Configuration
    GraphConfig get_config() const;
    Result<void> update_config(const GraphConfig& config);
    
    // Maintenance
    Result<void> validate_graph();
    Result<void> optimize_graph();
    Result<void> backup_graph();

private:
    BrainGraph() = default;
    ~BrainGraph() = default;
    BrainGraph(const BrainGraph&) = delete;
    BrainGraph& operator=(const BrainGraph&) = delete;
    
    // Internal methods
    Result<void> validate_node_data(const NodeData& data) const;
    Result<void> validate_connection_data(const ConnectionData& data) const;
    Result<bool> would_create_cycle(const ConnectionData& data) const;
    Result<void> notify_node_event(NodeID id, const std::string& event);
    Result<void> notify_connection_event(ConnectionID id, const std::string& event);
    Result<void> notify_graph_event(const std::string& event);
    
    // Member variables
    std::unique_ptr<NodeManager> node_manager_;
    std::unique_ptr<ConnectionManager> connection_manager_;
    std::unique_ptr<GraphPersistence> persistence_;
    
    GraphConfig config_;
    GraphStatistics statistics_;
    
    NodeEventCallback node_event_callback_;
    ConnectionEventCallback connection_event_callback_;
    GraphEventCallback graph_event_callback_;
    
    mutable std::shared_mutex graph_mutex_;
    bool initialized_;
    
    // Background tasks
    std::thread maintenance_thread_;
    std::atomic<bool> running_;
    
    void maintenance_loop();
    void update_statistics();
};

// Utility functions for graph operations
namespace graph_utils {
    
    // Path finding algorithms
    std::vector<NodeID> breadth_first_search(const std::map<NodeID, std::vector<NodeID>>& adjacency_list,
                                           NodeID start, NodeID end, size_t max_depth);
    
    std::vector<NodeID> depth_first_search(const std::map<NodeID, std::vector<NodeID>>& adjacency_list,
                                         NodeID start, NodeID end, size_t max_depth);
    
    // Graph analysis
    double calculate_clustering_coefficient(const std::map<NodeID, std::vector<NodeID>>& adjacency_list);
    double calculate_average_path_length(const std::map<NodeID, std::vector<NodeID>>& adjacency_list);
    std::vector<NodeID> find_articulation_points(const std::map<NodeID, std::vector<NodeID>>& adjacency_list);
    
    // Similarity metrics
    double cosine_similarity(const std::vector<double>& vec1, const std::vector<double>& vec2);
    double jaccard_similarity(const std::set<std::string>& set1, const std::set<std::string>& set2);
    
} // namespace graph_utils

} // namespace melvin
