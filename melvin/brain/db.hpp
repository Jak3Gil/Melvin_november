#pragma once

#include "melvin_common.hpp"
#include "node.hpp"
#include "connection.hpp"
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace melvin {

/**
 * @brief Database configuration for the brain graph
 */
struct DatabaseConfig {
    std::string db_path = "/var/melvin/data/brain.db";  ///< Database file path
    bool auto_sync = true;                              ///< Auto-sync on changes
    int sync_interval_ms = 1000;                        ///< Sync interval in milliseconds
    bool enable_wal = true;                             ///< Enable WAL mode for better concurrency
    int max_connections = 1000;                         ///< Maximum concurrent connections
    std::string backup_path = "/var/melvin/data/backup/"; ///< Backup directory
};

/**
 * @brief Database statistics and health information
 */
struct DatabaseStats {
    size_t total_nodes = 0;           ///< Total number of nodes
    size_t total_connections = 0;     ///< Total number of connections
    size_t active_nodes = 0;          ///< Number of active nodes
    size_t active_connections = 0;    ///< Number of active connections
    double db_size_mb = 0.0;         ///< Database size in MB
    std::string last_backup;          ///< Last backup timestamp
    bool is_healthy = false;          ///< Database health status
    std::string last_error;           ///< Last error message
};

/**
 * @brief SQLite database interface for brain graph persistence
 * 
 * Handles all database operations including auto-sync, backup,
 * and transaction management for thread-safe operations.
 */
class BrainDatabase {
public:
    // Constructor and destructor
    explicit BrainDatabase(const DatabaseConfig& config = DatabaseConfig());
    ~BrainDatabase();

    // Database lifecycle
    Result<void> init();
    Result<void> shutdown();
    bool is_initialized() const { return initialized_; }

    // Node operations
    Result<void> save_node(const Node& node);
    Result<void> update_node(const Node& node);
    Result<void> delete_node(NodeID node_id);
    Result<Node> load_node(NodeID node_id);
    Result<std::vector<Node>> load_all_nodes();
    Result<std::vector<Node>> load_nodes_by_type(Node::Type type);
    Result<std::vector<Node>> search_nodes(const std::string& query);

    // Connection operations
    Result<void> save_connection(const Connection& connection);
    Result<void> update_connection(const Connection& connection);
    Result<void> delete_connection(ConnectionID connection_id);
    Result<Connection> load_connection(ConnectionID connection_id);
    Result<std::vector<Connection>> load_all_connections();
    Result<std::vector<Connection>> load_connections_by_node(NodeID node_id);
    Result<std::vector<Connection>> load_connections_by_type(Connection::Type type);

    // Graph operations
    Result<void> save_graph(const std::vector<Node>& nodes, const std::vector<Connection>& connections);
    Result<std::pair<std::vector<Node>, std::vector<Connection>>> load_graph();
    Result<void> clear_graph();
    Result<void> validate_graph_integrity();

    // Maintenance operations
    Result<void> backup_database(const std::string& backup_path = "");
    Result<void> restore_database(const std::string& backup_path);
    Result<void> vacuum_database();
    Result<void> optimize_database();
    Result<DatabaseStats> get_database_stats();

    // Transaction management
    Result<void> begin_transaction();
    Result<void> commit_transaction();
    Result<void> rollback_transaction();
    bool is_in_transaction() const { return in_transaction_; }

    // Configuration
    Result<void> update_config(const DatabaseConfig& config);
    DatabaseConfig get_config() const { return config_; }

    // Event callbacks
    using ChangeCallback = std::function<void(const std::string& operation, const std::string& id)>;
    void set_change_callback(ChangeCallback callback) { change_callback_ = callback; }

private:
    // Database initialization
    Result<void> create_tables();
    Result<void> create_indexes();
    Result<void> setup_triggers();
    
    // Internal operations
    Result<void> auto_sync();
    Result<void> log_operation(const std::string& operation, const std::string& details);
    Result<void> check_database_health();
    
    // SQL helpers
    Result<void> execute_sql(const std::string& sql);
    Result<void> execute_sql_with_params(const std::string& sql, const std::vector<std::string>& params);
    
    // Backup and recovery
    Result<void> create_backup_file(const std::string& backup_path);
    Result<void> verify_backup_integrity(const std::string& backup_path);

    // Member variables
    DatabaseConfig config_;
    bool initialized_ = false;
    bool in_transaction_ = false;
    std::string db_connection_string_;
    ChangeCallback change_callback_;
    
    // SQLite connection (forward declaration to avoid including sqlite3.h in header)
    struct SQLiteConnection;
    std::unique_ptr<SQLiteConnection> sqlite_conn_;
    
    // Thread safety
    mutable std::shared_mutex db_mutex_;
    std::atomic<bool> auto_sync_running_{false};
    std::thread auto_sync_thread_;
    
    // Statistics
    mutable DatabaseStats stats_;
    mutable std::chrono::steady_clock::time_point last_stats_update_;
};

} // namespace melvin
