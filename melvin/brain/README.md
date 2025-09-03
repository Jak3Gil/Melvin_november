# Melvin Brain Module

The brain module implements Melvin's cognitive architecture using a node-based graph system with persistent storage and learning capabilities.

## Architecture

### Core Components

- **Node**: Represents inputs, concepts, or outputs with metadata and state
- **Connection**: Directed edges between nodes with weights and confidence scores
- **BrainGraph**: Main graph management class with thread-safe operations
- **BrainDatabase**: SQLite persistence layer with auto-sync capabilities

### Node Types

- **INPUT**: Sensor data, external inputs, user commands
- **CONCEPT**: Learned patterns, rules, decision points
- **OUTPUT**: Actions, motor commands, system responses

### Connection Types

- **DIRECT**: Directly learned connections
- **INFERRED**: Pattern-inferred connections
- **TEMPORARY**: Short-term testing connections
- **PERSISTENT**: Long-term connections that don't decay

## Features

- **Thread-safe operations** with shared mutexes
- **Automatic persistence** to SQLite database
- **Confidence learning** with Bayesian updates
- **Connection decay** for unused connections
- **Health monitoring** and integrity checks
- **JSON export/import** for UI and backup
- **Background maintenance** threads

## Usage

### Basic Graph Operations

```cpp
#include "melvin/brain/graph.hpp"

using namespace melvin;

// Get brain graph instance
auto& brain = BrainGraph::instance();

// Initialize with configuration
GraphConfig config;
config.enable_auto_cleanup = true;
config.confidence_threshold = 0.1;
brain.init(config);

// Create nodes
auto input_node_id = brain.create_node(Node::Type::INPUT, "camera_frame", "Camera input");
auto concept_node_id = brain.create_node(Node::Type::CONCEPT, "face_detected", "Face detection concept");
auto output_node_id = brain.create_node(Node::Type::OUTPUT, "wave_hand", "Wave hand action");

// Create connections
auto connection_id = brain.create_connection(
    input_node_id, concept_node_id, 0.8, 0.9, Connection::Type::DIRECT
);

// Update confidence based on learning
brain.update_connection_confidence(connection_id, 0.1);

// Query the graph
auto result = brain.query_graph("face detection", 10);
```

### Database Operations

```cpp
// Save entire graph
brain.save_graph();

// Load graph from database
brain.load_graph();

// Export to JSON
auto json = brain.export_graph_json();

// Import from JSON
brain.import_graph_json(json);
```

### Health Monitoring

```cpp
// Check graph health
brain.check_graph_health();

// Get statistics
auto stats = brain.get_graph_statistics();
std::cout << "Total nodes: " << stats.total_nodes << std::endl;
std::cout << "Total connections: " << stats.total_connections << std::endl;
std::cout << "Health status: " << (stats.is_healthy ? "Healthy" : "Issues detected") << std::endl;

// Get health issues
auto issues = brain.get_health_issues();
for (const auto& issue : issues) {
    std::cout << "Issue: " << issue << std::endl;
}
```

## Configuration

### GraphConfig Options

- `enable_auto_cleanup`: Automatically remove weak connections
- `confidence_threshold`: Minimum confidence for active connections
- `weight_threshold`: Minimum weight for active connections
- `max_nodes`: Maximum number of nodes allowed
- `max_connections`: Maximum number of connections allowed
- `detect_cycles`: Enable cycle detection in graph
- `health_check_interval_ms`: Background health check frequency
- `confidence_decay_rate`: Rate of confidence decay per hour

### DatabaseConfig Options

- `db_path`: SQLite database file path
- `auto_sync`: Enable automatic database synchronization
- `sync_interval_ms`: Sync interval in milliseconds
- `enable_wal`: Enable WAL mode for better concurrency
- `backup_path`: Backup directory path

## Thread Safety

All public methods are thread-safe using shared mutexes:

- **Read operations** use shared locks for concurrent access
- **Write operations** use exclusive locks for data integrity
- **Background threads** handle maintenance and health checks

## Persistence

### SQLite Schema

```sql
-- Nodes table
CREATE TABLE nodes (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    state TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    attributes TEXT  -- JSON attributes
);

-- Connections table
CREATE TABLE connections (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    weight REAL NOT NULL,
    confidence REAL NOT NULL,
    type TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_used INTEGER NOT NULL,
    usage_count INTEGER NOT NULL,
    attributes TEXT,  -- JSON attributes
    FOREIGN KEY (source_id) REFERENCES nodes(id),
    FOREIGN KEY (target_id) REFERENCES nodes(id)
);
```

### Auto-sync Features

- **Automatic persistence** on node/connection changes
- **Background sync thread** for periodic saves
- **Transaction support** for batch operations
- **Backup and recovery** capabilities

## Learning and Adaptation

### Confidence Updates

- **Bayesian updates** based on success/failure feedback
- **Usage tracking** with timestamp and count
- **Automatic decay** for unused connections
- **Persistent connections** that don't decay

### Weight Management

- **Dynamic adjustment** based on learning feedback
- **Clamping** to valid range (-1.0 to 1.0)
- **Connection strength** affects decision making

## Testing

Run the brain module tests:

```bash
# Build with tests enabled
cmake -DBUILD_TESTS=ON ..
make

# Run brain tests
./brain_tests

# Run specific test suites
./brain_tests --gtest_filter=NodeTest.*
./brain_tests --gtest_filter=ConnectionTest.*
./brain_tests --gtest_filter=GraphTest.*
./brain_tests --gtest_filter=DatabaseTest.*
```

## Dependencies

- **C++20** standard library
- **SQLite3** for persistence
- **nlohmann/json** for serialization
- **Threads** for concurrency
- **melvin_common** for shared types

## Future Extensions

- **ML model integration** (TensorRT, ONNX)
- **Advanced graph algorithms** (clustering, centrality)
- **Distributed graph** support
- **Real-time streaming** updates
- **Advanced learning** algorithms
