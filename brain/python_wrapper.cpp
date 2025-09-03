/**
 * 🐍 PYTHON WRAPPER FOR FAST BRAIN CORE
 * High-performance C++ backend accessible from Python
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "fast_brain_core.hpp"

namespace py = pybind11;
using namespace melvin::brain;

PYBIND11_MODULE(fast_brain_core, m) {
    m.doc() = "High-performance C++ brain operations for Melvin";

    // Enums
    py::enum_<NodeType>(m, "NodeType")
        .value("LANGUAGE", NodeType::LANGUAGE)
        .value("CODE", NodeType::CODE)
        .value("VISUAL", NodeType::VISUAL)
        .value("AUDIO", NodeType::AUDIO)
        .value("CONCEPT", NodeType::CONCEPT)
        .value("EMOTION", NodeType::EMOTION)
        .value("ATOMIC_FACT", NodeType::ATOMIC_FACT)
        .value("CONSOLIDATED", NodeType::CONSOLIDATED)
        .value("SPECIALIZED", NodeType::SPECIALIZED);

    py::enum_<ConnectionType>(m, "ConnectionType")
        .value("SIMILARITY", ConnectionType::SIMILARITY)
        .value("TEMPORAL", ConnectionType::TEMPORAL)
        .value("HEBBIAN", ConnectionType::HEBBIAN)
        .value("MULTIMODAL", ConnectionType::MULTIMODAL)
        .value("ATOMIC_RELATION", ConnectionType::ATOMIC_RELATION)
        .value("CONSOLIDATION", ConnectionType::CONSOLIDATION);

    // FastNode class
    py::class_<FastNode>(m, "FastNode")
        .def_readonly("id", &FastNode::id)
        .def_readonly("type", &FastNode::type)
        .def_readonly("content", &FastNode::content)
        .def("get_activation_strength", &FastNode::get_activation_strength)
        .def("get_activation_count", &FastNode::get_activation_count)
        .def("get_connection_count", &FastNode::get_connection_count)
        .def("get_success_rate", &FastNode::get_success_rate)
        .def("get_content_size", &FastNode::get_content_size)
        .def("should_fragment", &FastNode::should_fragment)
        .def("should_consolidate", &FastNode::should_consolidate)
        .def("should_specialize", &FastNode::should_specialize)
        .def("activate", &FastNode::activate, py::arg("strength") = 1.0f)
        .def("update_success_rate", &FastNode::update_success_rate);

    // FastConnection class
    py::class_<FastConnection>(m, "FastConnection")
        .def_readonly("id", &FastConnection::id)
        .def_readonly("source_id", &FastConnection::source_id)
        .def_readonly("target_id", &FastConnection::target_id)
        .def_readonly("type", &FastConnection::type)
        .def("get_weight", &FastConnection::get_weight)
        .def("get_coactivation_count", &FastConnection::get_coactivation_count)
        .def("strengthen", &FastConnection::strengthen, py::arg("amount") = 0.01f)
        .def("weaken", &FastConnection::weaken, py::arg("amount") = 0.005f);

    // Performance Stats
    py::class_<FastBrainCore::PerformanceStats>(m, "PerformanceStats")
        .def_readonly("total_nodes", &FastBrainCore::PerformanceStats::total_nodes)
        .def_readonly("total_connections", &FastBrainCore::PerformanceStats::total_connections)
        .def_readonly("total_activations", &FastBrainCore::PerformanceStats::total_activations)
        .def_readonly("total_searches", &FastBrainCore::PerformanceStats::total_searches)
        .def_readonly("avg_search_time_ms", &FastBrainCore::PerformanceStats::avg_search_time_ms)
        .def_readonly("avg_activation_time_ms", &FastBrainCore::PerformanceStats::avg_activation_time_ms)
        .def_readonly("memory_usage_bytes", &FastBrainCore::PerformanceStats::memory_usage_bytes)
        .def_readonly("cache_hit_rate", &FastBrainCore::PerformanceStats::cache_hit_rate);

    // FastBrainCore class
    py::class_<FastBrainCore>(m, "FastBrainCore")
        .def(py::init<>())
        
        // Core operations
        .def("create_node", &FastBrainCore::create_node,
             "Create a new node", py::arg("type"), py::arg("content"))
        .def("create_connection", &FastBrainCore::create_connection,
             "Create a new connection", py::arg("source"), py::arg("target"), 
             py::arg("type"), py::arg("initial_weight") = 1.0f)
        
        // Node operations
        .def("activate_node", &FastBrainCore::activate_node,
             "Activate a node", py::arg("id"), py::arg("strength") = 1.0f)
        .def("update_node_content", &FastBrainCore::update_node_content,
             "Update node content", py::arg("id"), py::arg("new_content"))
        .def("remove_node", &FastBrainCore::remove_node,
             "Remove a node", py::arg("id"))
        
        // Connection operations
        .def("strengthen_connection", &FastBrainCore::strengthen_connection,
             "Strengthen a connection", py::arg("id"), py::arg("amount") = 0.01f)
        .def("weaken_connection", &FastBrainCore::weaken_connection,
             "Weaken a connection", py::arg("id"), py::arg("amount") = 0.005f)
        .def("remove_connection", &FastBrainCore::remove_connection,
             "Remove a connection", py::arg("id"))
        
        // Search operations
        .def("search_nodes_simd", &FastBrainCore::search_nodes_simd,
             "High-speed node search using SIMD", py::arg("query"), py::arg("max_results") = 10)
        .def("get_connected_nodes", &FastBrainCore::get_connected_nodes,
             "Get nodes connected to a given node", py::arg("id"), py::arg("max_results") = 100)
        .def("find_similar_nodes", &FastBrainCore::find_similar_nodes,
             "Find semantically similar nodes", py::arg("reference_id"), py::arg("max_results") = 10)
        
        // Hebbian learning
        .def("hebbian_update_batch", &FastBrainCore::hebbian_update_batch,
             "Update connections using Hebbian learning", py::arg("active_nodes"))
        .def("strengthen_coactivated_connections", &FastBrainCore::strengthen_coactivated_connections,
             "Strengthen connections between coactivated nodes", py::arg("node1"), py::arg("node2"))
        
        // Dynamic sizing
        .def("find_fragmentation_candidates", &FastBrainCore::find_fragmentation_candidates,
             "Find nodes that should be fragmented")
        .def("find_consolidation_candidates", &FastBrainCore::find_consolidation_candidates,
             "Find node groups that should be consolidated")
        .def("find_specialization_candidates", &FastBrainCore::find_specialization_candidates,
             "Find nodes that should be specialized")
        
        // Optimization operations
        .def("fragment_node", &FastBrainCore::fragment_node,
             "Fragment a large node", py::arg("id"), py::arg("fragments"))
        .def("consolidate_nodes", &FastBrainCore::consolidate_nodes,
             "Consolidate multiple nodes", py::arg("node_ids"), py::arg("consolidated_content"))
        .def("specialize_node", &FastBrainCore::specialize_node,
             "Create specialized variant", py::arg("base_id"), py::arg("specialization_context"))
        
        // Batch operations
        .def("batch_activate_nodes", [](FastBrainCore& self, const std::vector<std::pair<uint64_t, float>>& activations) {
            self.batch_activate_nodes(activations);
        }, "Batch activate multiple nodes")
        
        // Performance and statistics
        .def("get_performance_stats", &FastBrainCore::get_performance_stats,
             "Get performance statistics")
        .def("reset_performance_counters", &FastBrainCore::reset_performance_counters,
             "Reset performance counters")
        .def("get_memory_usage", &FastBrainCore::get_memory_usage,
             "Get current memory usage in bytes")
        
        // Getters
        .def("get_node_count", &FastBrainCore::get_node_count,
             "Get total number of nodes")
        .def("get_connection_count", &FastBrainCore::get_connection_count,
             "Get total number of connections")
        .def("get_node_count_by_type", &FastBrainCore::get_node_count_by_type,
             "Get node count by type", py::arg("type"))
        
        // Data persistence
        .def("load_from_sqlite", &FastBrainCore::load_from_sqlite,
             "Load brain from SQLite database", py::arg("db_path"))
        .def("save_to_sqlite", &FastBrainCore::save_to_sqlite,
             "Save brain to SQLite database", py::arg("db_path"))
        .def("export_to_json", &FastBrainCore::export_to_json,
             "Export brain to JSON format")
        .def("import_from_json", &FastBrainCore::import_from_json,
             "Import brain from JSON format", py::arg("json_data"));

    // Utility functions
    m.def("dot_product_simd", &utils::dot_product_simd,
          "SIMD-optimized dot product", py::arg("a"), py::arg("b"), py::arg("size"));
    
    m.def("cosine_similarity_simd", &utils::cosine_similarity_simd,
          "SIMD-optimized cosine similarity", py::arg("a"), py::arg("b"), py::arg("size"));
    
    m.def("split_string_fast", &utils::split_string_fast,
          "Fast string splitting", py::arg("str"), py::arg("delimiter"));

    // High-resolution timer
    py::class_<utils::HighResTimer>(m, "HighResTimer")
        .def(py::init<>())
        .def("elapsed_ms", &utils::HighResTimer::elapsed_ms,
             "Get elapsed time in milliseconds")
        .def("reset", &utils::HighResTimer::reset,
             "Reset the timer");
}
