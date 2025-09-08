#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <mutex>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <set>
#include <optional>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include "melvin_optimized_v2.h"

// ============================================================================
// BRAIN MONITORING SYSTEM
// ============================================================================

struct BrainActivity {
    uint64_t timestamp;
    uint64_t node_id;
    std::string activity_type;
    std::string details;
    double value;
    
    BrainActivity() : timestamp(0), node_id(0), value(0.0) {}
    
    BrainActivity(uint64_t ts, uint64_t id, const std::string& type, 
                  const std::string& det, double val)
        : timestamp(ts), node_id(id), activity_type(type), details(det), value(val) {}
};

struct BrainStats {
    uint64_t total_nodes;
    uint64_t total_connections;
    uint64_t hebbian_events;
    uint64_t pruning_events;
    uint64_t storage_bytes;
    double storage_mb;
    uint64_t uptime_seconds;
    uint64_t start_time;
    
    // Cognitive processing stats
    uint64_t cognitive_processing_events;
    uint64_t activation_clusters_formed;
    uint64_t interpretation_clusters_created;
    uint64_t candidate_responses_generated;
    uint64_t context_bias_applications;
    
    BrainStats() : total_nodes(0), total_connections(0), hebbian_events(0),
                   pruning_events(0), storage_bytes(0), storage_mb(0.0),
                   uptime_seconds(0), start_time(0), cognitive_processing_events(0),
                   activation_clusters_formed(0), interpretation_clusters_created(0),
                   candidate_responses_generated(0), context_bias_applications(0) {}
};

class MelvinBrainMonitor {
private:
    std::unique_ptr<MelvinOptimizedV2> melvin;
    std::vector<BrainActivity> activity_log;
    std::map<uint64_t, uint64_t> node_creation_times;
    std::map<uint64_t, uint64_t> connection_formation_times;
    std::vector<std::pair<uint64_t, uint64_t>> hebbian_events;
    std::vector<uint64_t> pruning_events;
    std::vector<BrainStats> stats_history;
    
    std::mutex monitor_mutex;
    std::thread monitoring_thread;
    bool monitoring_active;
    uint64_t monitor_start_time;
    
    static constexpr size_t MAX_ACTIVITY_LOG = 10000;
    static constexpr size_t MAX_STATS_HISTORY = 1000;
    static constexpr uint64_t MONITOR_INTERVAL_MS = 1000; // 1 second
    
public:
    MelvinBrainMonitor(const std::string& storage_path = "melvin_binary_memory") {
        melvin = std::make_unique<MelvinOptimizedV2>(storage_path);
        monitoring_active = false;
        monitor_start_time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        
        std::cout << "🧠 Melvin Brain Monitor initialized" << std::endl;
    }
    
    ~MelvinBrainMonitor() {
        stop_monitoring();
    }
    
    void start_monitoring() {
        if (monitoring_active) {
            std::cout << "⚠️  Monitoring already active" << std::endl;
            return;
        }
        
        monitoring_active = true;
        monitoring_thread = std::thread(&MelvinBrainMonitor::monitoring_loop, this);
        
        std::cout << "🔍 Brain monitoring started" << std::endl;
    }
    
    void stop_monitoring() {
        if (!monitoring_active) return;
        
        monitoring_active = false;
        if (monitoring_thread.joinable()) {
            monitoring_thread.join();
        }
        
        std::cout << "⏹️  Brain monitoring stopped" << std::endl;
    }
    
    void log_activity(uint64_t node_id, const std::string& activity_type, 
                     const std::string& details, double value = 0.0) {
        std::lock_guard<std::mutex> lock(monitor_mutex);
        
        uint64_t current_time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        
        BrainActivity activity(current_time, node_id, activity_type, details, value);
        activity_log.push_back(activity);
        
        // Keep log size manageable
        if (activity_log.size() > MAX_ACTIVITY_LOG) {
            activity_log.erase(activity_log.begin());
        }
        
        // Log to console
        std::cout << "📊 [" << format_timestamp(current_time) << "] "
                  << activity_type << " - Node " << std::hex << node_id 
                  << ": " << details << std::endl;
    }
    
    void log_node_creation(uint64_t node_id) {
        uint64_t current_time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        
        node_creation_times[node_id] = current_time;
        log_activity(node_id, "NODE_CREATED", "New node created");
    }
    
    void log_connection_formation(uint64_t source_id, uint64_t target_id) {
        uint64_t current_time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        
        uint64_t connection_id = std::hash<std::string>{}(
            std::to_string(source_id) + std::to_string(target_id));
        connection_formation_times[connection_id] = current_time;
        
        std::string details = "Connection " + std::to_string(source_id) + " -> " + std::to_string(target_id);
        log_activity(source_id, "CONNECTION_FORMED", details);
        
        hebbian_events.push_back({source_id, target_id});
    }
    
    void log_hebbian_event(uint64_t source_id, uint64_t target_id) {
        hebbian_events.push_back({source_id, target_id});
        std::string details = "Hebbian learning: " + std::to_string(source_id) + " -> " + std::to_string(target_id);
        log_activity(source_id, "HEBBIAN_LEARNING", details);
    }
    
    void log_pruning_event(uint64_t node_id) {
        pruning_events.push_back(node_id);
        log_activity(node_id, "NODE_PRUNED", "Node pruned due to low importance");
    }
    
    // Cognitive processing event logging
    void log_cognitive_processing_event(const std::string& input_text, uint64_t activated_nodes_count) {
        uint64_t current_time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
        
        std::string details = "Processed input: " + input_text.substr(0, 30) + 
                             "... (" + std::to_string(activated_nodes_count) + " nodes activated)";
        log_activity(0, "COGNITIVE_PROCESSING", details, activated_nodes_count);
    }
    
    void log_activation_cluster_formation(uint64_t cluster_size) {
        log_activity(0, "ACTIVATION_CLUSTER", "Formed activation cluster with " + 
                     std::to_string(cluster_size) + " nodes", cluster_size);
    }
    
    void log_interpretation_cluster_creation(uint64_t cluster_id, float confidence) {
        log_activity(cluster_id, "INTERPRETATION_CLUSTER", 
                     "Created interpretation cluster with confidence " + std::to_string(confidence), 
                     confidence);
    }
    
    void log_candidate_response_generation(uint64_t response_id, float confidence) {
        log_activity(response_id, "CANDIDATE_RESPONSE", 
                     "Generated candidate response with confidence " + std::to_string(confidence), 
                     confidence);
    }
    
    void log_context_bias_application(uint64_t node_id, float bias_strength) {
        log_activity(node_id, "CONTEXT_BIAS", 
                     "Applied context bias with strength " + std::to_string(bias_strength), 
                     bias_strength);
    }
    
    BrainStats get_current_stats() {
        BrainStats stats;
        
        // Get stats from Melvin
        auto state = melvin->get_unified_state();
        
        stats.total_nodes = state.global_memory.total_nodes;
        stats.total_connections = state.global_memory.total_edges;
        stats.storage_mb = state.global_memory.storage_used_mb;
        stats.storage_bytes = static_cast<uint64_t>(stats.storage_mb * 1024 * 1024);
        stats.hebbian_events = state.global_memory.stats.hebbian_updates;
        stats.pruning_events = pruning_events.size();
        stats.uptime_seconds = state.system.uptime_seconds;
        stats.start_time = monitor_start_time;
        
        // Count cognitive processing events from activity log
        stats.cognitive_processing_events = 0;
        stats.activation_clusters_formed = 0;
        stats.interpretation_clusters_created = 0;
        stats.candidate_responses_generated = 0;
        stats.context_bias_applications = 0;
        
        for (const auto& activity : activity_log) {
            if (activity.activity_type == "COGNITIVE_PROCESSING") {
                stats.cognitive_processing_events++;
            } else if (activity.activity_type == "ACTIVATION_CLUSTER") {
                stats.activation_clusters_formed++;
            } else if (activity.activity_type == "INTERPRETATION_CLUSTER") {
                stats.interpretation_clusters_created++;
            } else if (activity.activity_type == "CANDIDATE_RESPONSE") {
                stats.candidate_responses_generated++;
            } else if (activity.activity_type == "CONTEXT_BIAS") {
                stats.context_bias_applications++;
            }
        }
        
        return stats;
    }
    
    void save_activity_report(const std::string& filename = "melvin_brain_report.json") {
        std::lock_guard<std::mutex> lock(monitor_mutex);
        
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "❌ Failed to open report file: " << filename << std::endl;
            return;
        }
        
        auto stats = get_current_stats();
        
        file << "{\n";
        file << "  \"report_timestamp\": " << static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()) << ",\n";
        file << "  \"monitoring_duration_seconds\": " << stats.uptime_seconds << ",\n";
        file << "  \"brain_stats\": {\n";
        file << "    \"total_nodes\": " << stats.total_nodes << ",\n";
        file << "    \"total_connections\": " << stats.total_connections << ",\n";
        file << "    \"hebbian_events\": " << stats.hebbian_events << ",\n";
        file << "    \"pruning_events\": " << stats.pruning_events << ",\n";
        file << "    \"storage_bytes\": " << stats.storage_bytes << ",\n";
        file << "    \"storage_mb\": " << std::fixed << std::setprecision(2) << stats.storage_mb << ",\n";
        file << "    \"cognitive_processing_events\": " << stats.cognitive_processing_events << ",\n";
        file << "    \"activation_clusters_formed\": " << stats.activation_clusters_formed << ",\n";
        file << "    \"interpretation_clusters_created\": " << stats.interpretation_clusters_created << ",\n";
        file << "    \"candidate_responses_generated\": " << stats.candidate_responses_generated << ",\n";
        file << "    \"context_bias_applications\": " << stats.context_bias_applications << "\n";
        file << "  },\n";
        file << "  \"recent_activity\": [\n";
        
        // Write recent activity (last 100 entries)
        size_t start_idx = (activity_log.size() > 100) ? activity_log.size() - 100 : 0;
        for (size_t i = start_idx; i < activity_log.size(); ++i) {
            const auto& activity = activity_log[i];
            file << "    {\n";
            file << "      \"timestamp\": " << activity.timestamp << ",\n";
            file << "      \"node_id\": \"" << std::hex << activity.node_id << "\",\n";
            file << "      \"activity_type\": \"" << activity.activity_type << "\",\n";
            file << "      \"details\": \"" << activity.details << "\",\n";
            file << "      \"value\": " << activity.value << "\n";
            file << "    }";
            if (i < activity_log.size() - 1) file << ",";
            file << "\n";
        }
        
        file << "  ]\n";
        file << "}\n";
        
        std::cout << "📄 Activity report saved to: " << filename << std::endl;
    }
    
    void print_real_time_stats() {
        auto stats = get_current_stats();
        
        std::cout << "\n🧠 MELVIN BRAIN STATS" << std::endl;
        std::cout << "=====================" << std::endl;
        std::cout << "📊 Nodes: " << stats.total_nodes << std::endl;
        std::cout << "🔗 Connections: " << stats.total_connections << std::endl;
        std::cout << "⚡ Hebbian Events: " << stats.hebbian_events << std::endl;
        std::cout << "🗑️ Pruning Events: " << stats.pruning_events << std::endl;
        std::cout << "💾 Storage: " << std::fixed << std::setprecision(2) << stats.storage_mb << " MB" << std::endl;
        std::cout << "⏱️ Uptime: " << format_duration(stats.uptime_seconds) << std::endl;
        std::cout << "📈 Activity Log Entries: " << activity_log.size() << std::endl;
        std::cout << "\n🧠 Cognitive Processing Stats:" << std::endl;
        std::cout << "🔍 Cognitive Events: " << stats.cognitive_processing_events << std::endl;
        std::cout << "🎯 Activation Clusters: " << stats.activation_clusters_formed << std::endl;
        std::cout << "💭 Interpretation Clusters: " << stats.interpretation_clusters_created << std::endl;
        std::cout << "💬 Candidate Responses: " << stats.candidate_responses_generated << std::endl;
        std::cout << "🎛️ Context Bias Applications: " << stats.context_bias_applications << std::endl;
        std::cout << "=====================" << std::endl;
    }
    
private:
    void monitoring_loop() {
        while (monitoring_active) {
            // Capture current stats
            auto stats = get_current_stats();
            stats_history.push_back(stats);
            
            // Keep history size manageable
            if (stats_history.size() > MAX_STATS_HISTORY) {
                stats_history.erase(stats_history.begin());
            }
            
            // Print stats every 10 seconds
            static uint64_t last_print = 0;
            uint64_t current_time = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
            
            if (current_time - last_print >= 10) {
                print_real_time_stats();
                last_print = current_time;
            }
            
            // Sleep for monitoring interval
            std::this_thread::sleep_for(std::chrono::milliseconds(MONITOR_INTERVAL_MS));
        }
    }
    
    std::string format_timestamp(uint64_t timestamp) {
        auto time_point = std::chrono::system_clock::from_time_t(timestamp);
        auto time_t = std::chrono::system_clock::to_time_t(time_point);
        auto tm = *std::localtime(&time_t);
        
        std::ostringstream oss;
        oss << std::put_time(&tm, "%H:%M:%S");
        return oss.str();
    }
    
    std::string format_duration(uint64_t seconds) {
        uint64_t hours = seconds / 3600;
        uint64_t minutes = (seconds % 3600) / 60;
        uint64_t secs = seconds % 60;
        
        std::ostringstream oss;
        if (hours > 0) {
            oss << hours << "h " << minutes << "m " << secs << "s";
        } else if (minutes > 0) {
            oss << minutes << "m " << secs << "s";
        } else {
            oss << secs << "s";
        }
        return oss.str();
    }
};

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main() {
    std::cout << "🧠 MELVIN BRAIN MONITOR (C++)" << std::endl;
    std::cout << "=============================" << std::endl;
    
    try {
        // Initialize brain monitor
        MelvinBrainMonitor monitor;
        
        // Start monitoring
        monitor.start_monitoring();
        
        // Simulate some brain activity
        std::cout << "🧪 Simulating brain activity..." << std::endl;
        
        // Process some test inputs
        auto melvin = std::make_unique<MelvinOptimizedV2>();
        
        uint64_t text_id = melvin->process_text_input("Hello, this is a test of the brain monitor!");
        monitor.log_node_creation(text_id);
        
        uint64_t code_id = melvin->process_code_input("def test_function():\n    return 'Hello, World!'");
        monitor.log_node_creation(code_id);
        
        // Simulate some connections
        monitor.log_connection_formation(text_id, code_id);
        monitor.log_hebbian_event(text_id, code_id);
        
        // Simulate cognitive processing
        std::cout << "🧠 Simulating cognitive processing..." << std::endl;
        std::string cognitive_input = "How does machine learning work?";
        auto cognitive_result = melvin->process_cognitive_input(cognitive_input);
        
        monitor.log_cognitive_processing_event(cognitive_input, cognitive_result.activated_nodes.size());
        monitor.log_activation_cluster_formation(cognitive_result.activated_nodes.size());
        
        for (size_t i = 0; i < cognitive_result.clusters.size(); ++i) {
            monitor.log_interpretation_cluster_creation(i, cognitive_result.clusters[i].confidence);
        }
        
        monitor.log_candidate_response_generation(0, cognitive_result.confidence);
        
        // Let monitoring run for a bit
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        // Print final stats
        monitor.print_real_time_stats();
        
        // Save report
        monitor.save_activity_report();
        
        // Stop monitoring
        monitor.stop_monitoring();
        
        std::cout << "\n🎉 Brain monitoring test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
