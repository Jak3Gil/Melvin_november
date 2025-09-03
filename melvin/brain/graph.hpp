#pragma once

#include "melvin_common.hpp"
#include "node.hpp"
#include "connection.hpp"
#include "db.hpp"
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <thread>
#include <atomic>

namespace melvin {

/**
 * @brief Graph query result structure
 */
struct GraphQueryResult {
    std::vector<Node> nodes;                    ///< Matching nodes
    std::vector<Connection> connections;        ///< Matching connections
    size_t total_matches = 0;                   ///< Total number of matches
    double query_time_ms = 0.0;                 ///< Query execution time
    std::string query_string;                   ///< Original query string
    bool has_more = false;                      ///< Whether more results exist
};

/**
 * @brief Graph statistics and health information
 */
struct GraphStatistics {
    size_t total_nodes = 0;                     ///< Total number of nodes
    size_t total_connections = 0;                ///< Total number of connections
    size_t active_nodes = 0;                    ///< Number of active nodes
    size_t active_connections = 0;              ///< Number of active connections
    size_t orphaned_connections = 0;            ///< Number of orphaned connections
    size_t cycles_detected = 0;                 ///< Number of cycles detected
    double average_confidence = 0.0;            ///< Average connection confidence
    double average_weight = 0.0;                ///< Average connection weight
    std::string last_health_check;              ///< Last health check timestamp
    bool is_healthy = false;                    ///< Overall graph health status
};

/**
 * @brief Graph configuration options
 */
struct GraphConfig {
    bool enable_auto_cleanup = true;            ///< Enable automatic cleanup of weak connections
    double confidence_threshold = 0.1;          ///< Minimum confidence for active connections
    double weight_threshold = 0.05;             ///< Minimum weight for active connections
    int max_nodes = 10000;                      ///< Maximum number of nodes
    int max_connections = 50000;                ///< Maximum number of connections
    bool detect_cycles = true;                  ///< Enable cycle detection
    bool validate_on_startup = true;            ///< Validate graph integrity on startup
    int health_check_interval_ms = 30000;       ///< Health check interval in milliseconds
    double confidence_decay_rate = 0.001;       ///< Confidence decay rate per hour
    std::string database_path = "/var/melvin/data/brain.db"; ///< Database path
};

/**
 * @brief Main brain graph management class
 * 
 * Provides a thread-safe interface for managing nodes, connections,
 * and graph operations with automatic persistence and health monitoring.
 */
class BrainGraph {
public:
    // Singleton access
    static BrainGraph& instance();
    
    // Constructor and destructor
    BrainGraph();
    ~BrainGraph();

    // Graph lifecycle
    Result<void> init(const GraphConfig& config = GraphConfig());
    Result<void> shutdown();
    bool is_initialized() const { return initialized_; }

    // Node management
    Result<NodeID> create_node(Node::Type type, const std::string& name, const std::string& description = "");
    Result<void> update_node(NodeID id, const Node& node);
    Result<void> remove_node(NodeID id);
    Result<Node> get_node(NodeID id) const;
    Result<std::vector<Node>> get_nodes_by_type(Node::Type type) const;
    Result<std::vector<Node>> search_nodes(const std::string& query, size_t limit = 100) const;

    // Connection management
    Result<ConnectionID> create_connection(NodeID source_id, NodeID target_id, 
                                         Weight weight = 0.0, Confidence confidence = 0.5,
                                         Connection::Type type = Connection::Type::DIRECT);
    Result<void> update_connection(ConnectionID id, const Connection& connection);
    Result<void> remove_connection(ConnectionID id);
    Result<Connection> get_connection(ConnectionID id) const;
    Result<std::vector<Connection>> get_connections(NodeID node_id) const;
    Result<std::vector<Connection>> get_incoming_connections(NodeID node_id) const;
    Result<std::vector<Connection>> get_outgoing_connections(NodeID node_id) const;

    // Learning and confidence updates
    Result<void> update_confidence(NodeID node_id, Confidence new_confidence);
    Result<void> update_connection_confidence(ConnectionID connection_id, Confidence delta);
    Result<void> update_connection_weight(ConnectionID connection_id, Weight delta);
    Result<void> mark_connection_used(ConnectionID connection_id);
    Result<void> decay_connections(double decay_rate = 0.001);

    // Graph operations
    Result<GraphQueryResult> query_graph(const std::string& query, size_t limit = 100) const;
    Result<std::vector<Node>> find_path(NodeID start_id, NodeID end_id, size_t max_depth = 10) const;
    Result<std::vector<Node>> get_neighbors(NodeID node_id, size_t max_depth = 1) const;
    Result<std::vector<Node>> find_similar_nodes(NodeID node_id, double similarity_threshold = 0.7) const;
    Result<std::vector<Connection>> find_strong_connections(double min_confidence = 0.8) const;

    // Analysis and health
    Result<GraphStatistics> get_graph_statistics() const;
    Result<void> check_graph_health();
    Result<std::vector<std::string>> get_health_issues() const;
    Result<void> cleanup_orphaned_connections();
    Result<void> detect_cycles();
    Result<void> optimize_graph();

    // Persistence
    Result<void> save_graph();
    Result<void> load_graph();
    Result<void> export_graph(const std::string& file_path) const;
    Result<void> import_graph(const std::string& file_path);
    Result<nlohmann::json> export_graph_json() const;
    Result<void> import_graph_json(const nlohmann::json& json);

    // Configuration and events
    Result<void> update_config(const GraphConfig& config);
    GraphConfig get_config() const { return config_; }
    
    using GraphChangeCallback = std::function<void(const std::string& operation, const std::string& id)>;
    void set_change_callback(GraphChangeCallback callback) { change_callback_ = callback; }

    // Maintenance
    Result<void> start_maintenance();
    Result<void> stop_maintenance();
    bool is_maintenance_running() const { return maintenance_running_; }

private:
    // Internal operations
    Result<void> validate_node_exists(NodeID id) const;
    Result<void> validate_connection_valid(ConnectionID id) const;
    Result<void> cleanup_weak_connections();
    Result<void> update_statistics();
    Result<void> log_operation(const std::string& operation, const std::string& details);
    
    // Background tasks
    void maintenance_thread();
    void health_check_thread();
    void auto_save_thread();
    
    // Graph algorithms
    bool has_cycle_dfs(NodeID node_id, std::set<NodeID>& visited, std::set<NodeID>& rec_stack) const;
    double calculate_similarity(const Node& node1, const Node& node2) const;
    std::vector<Node> find_path_dfs(NodeID current, NodeID target, std::set<NodeID>& visited, 
                                   size_t depth, size_t max_depth) const;

    // Member variables
    GraphConfig config_;
    bool initialized_ = false;
    bool maintenance_running_ = false;
    
    // Data storage
    std::map<NodeID, Node> nodes_;
    std::map<ConnectionID, Connection> connections_;
    std::map<NodeID, std::set<ConnectionID>> node_connections_;
    
    // Database
    std::unique_ptr<BrainDatabase> database_;
    
    // Thread safety
    mutable std::shared_mutex graph_mutex_;
    mutable std::shared_mutex stats_mutex_;
    
    // Background threads
    std::thread maintenance_thread_;
    std::thread health_check_thread_;
    std::thread auto_save_thread_;
    std::atomic<bool> shutdown_requested_{false};
    
    // Statistics and health
    mutable GraphStatistics statistics_;
    mutable std::vector<std::string> health_issues_;
    mutable std::chrono::steady_clock::time_point last_stats_update_;
    mutable std::chrono::steady_clock::time_point last_health_check_;
    
    // Event handling
    GraphChangeCallback change_callback_;
    
    // ID generation
    std::atomic<NodeID> next_node_id_{1};
    std::atomic<ConnectionID> next_connection_id_{1};
};

} // namespace melvin
