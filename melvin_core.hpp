#pragma once

#include "melvin_common.hpp"
#include <memory>
#include <thread>
#include <atomic>

namespace melvin {

// Forward declarations
class HAL;
class BrainGraph;
class LearningEngine;
class HTTPServer;

// System health information
struct SystemHealth {
    SystemStatus status;
    bool hardware_ok;
    bool brain_ok;
    bool learning_ok;
    bool webserver_ok;
    std::string error_message;
    TimePoint last_check;
    Duration uptime;
    
    SystemHealth() : status(SystemStatus::INITIALIZING), hardware_ok(false), brain_ok(false),
                    learning_ok(false), webserver_ok(false) {}
};

// System configuration
struct SystemConfig {
    std::string config_file;
    std::string log_file;
    LogLevel log_level;
    bool enable_hardware;
    bool enable_brain;
    bool enable_learning;
    bool enable_webserver;
    bool enable_auto_recovery;
    Duration health_check_interval;
    Duration shutdown_timeout;
    
    SystemConfig() : config_file("/etc/melvin/melvin.conf"), log_file("/var/melvin/logs/melvin.log"),
                    log_level(LogLevel::INFO), enable_hardware(true), enable_brain(true),
                    enable_learning(true), enable_webserver(true), enable_auto_recovery(true),
                    health_check_interval(Seconds(5)), shutdown_timeout(Seconds(30)) {}
};

// Melvin Core - Main system orchestrator
class MelvinCore {
public:
    static MelvinCore& instance();
    
    // Initialization and shutdown
    Result<void> init(const SystemConfig& config = SystemConfig());
    Result<void> shutdown();
    
    // System control
    Result<void> start();
    Result<void> stop();
    Result<void> pause();
    Result<void> resume();
    Result<void> restart();
    
    // Status and health
    SystemHealth get_system_health() const;
    SystemStatus get_system_status() const;
    Result<void> check_system_health();
    Result<void> perform_self_diagnostic();
    
    // Subsystem access
    HAL& get_hardware() const;
    BrainGraph& get_brain() const;
    LearningEngine& get_learning() const;
    HTTPServer& get_webserver() const;
    
    // Configuration
    SystemConfig get_config() const;
    Result<void> update_config(const SystemConfig& config);
    Result<void> load_config_from_file(const std::string& filename);
    Result<void> save_config_to_file(const std::string& filename);
    
    // System events
    using SystemEventCallback = std::function<void(const std::string&, SystemStatus)>;
    void set_system_event_callback(SystemEventCallback callback);
    
    // Recovery and maintenance
    Result<void> recover_subsystem(const std::string& subsystem_name);
    Result<void> restart_subsystem(const std::string& subsystem_name);
    Result<void> backup_system_state();
    Result<void> restore_system_state(const std::string& backup_file);
    
    // Statistics and monitoring
    Result<std::map<std::string, std::string>> get_system_statistics() const;
    Result<void> enable_monitoring(bool enable);
    Result<void> set_monitoring_interval(Duration interval);

private:
    MelvinCore() = default;
    ~MelvinCore() = default;
    MelvinCore(const MelvinCore&) = delete;
    MelvinCore& operator=(const MelvinCore&) = delete;
    
    // Internal methods
    Result<void> init_hardware();
    Result<void> init_brain();
    Result<void> init_learning();
    Result<void> init_webserver();
    
    Result<void> start_hardware();
    Result<void> start_brain();
    Result<void> start_learning();
    Result<void> start_webserver();
    
    Result<void> stop_hardware();
    Result<void> stop_brain();
    Result<void> stop_learning();
    Result<void> stop_webserver();
    
    Result<void> shutdown_hardware();
    Result<void> shutdown_brain();
    Result<void> shutdown_learning();
    Result<void> shutdown_webserver();
    
    Result<void> perform_health_check();
    Result<void> handle_subsystem_failure(const std::string& subsystem_name, const std::string& error);
    Result<void> attempt_recovery(const std::string& subsystem_name);
    
    // Member variables
    SystemConfig config_;
    SystemHealth health_;
    
    std::unique_ptr<HAL> hardware_;
    std::unique_ptr<BrainGraph> brain_;
    std::unique_ptr<LearningEngine> learning_;
    std::unique_ptr<HTTPServer> webserver_;
    
    SystemEventCallback system_event_callback_;
    
    mutable std::shared_mutex core_mutex_;
    bool initialized_;
    std::atomic<bool> running_;
    
    // Background tasks
    std::thread health_monitor_thread_;
    std::thread recovery_thread_;
    
    void health_monitor_loop();
    void recovery_loop();
    void notify_system_event(const std::string& event, SystemStatus status);
    
    // Recovery state
    std::map<std::string, int> failure_counts_;
    std::map<std::string, TimePoint> last_failure_times_;
    std::map<std::string, bool> recovery_in_progress_;
    
    // Constants
    static constexpr int MAX_FAILURE_COUNT = 3;
    static constexpr Duration FAILURE_RESET_INTERVAL = Minutes(5);
    static constexpr Duration RECOVERY_TIMEOUT = Seconds(30);
};

// System utility functions
namespace system_utils {
    
    // Configuration parsing
    SystemConfig parse_config_file(const std::string& filename);
    Result<void> validate_config(const SystemConfig& config);
    
    // Health assessment
    double calculate_system_health_score(const SystemHealth& health);
    std::string get_health_status_description(const SystemHealth& health);
    
    // Recovery strategies
    std::vector<std::string> get_recovery_strategies(const std::string& subsystem_name);
    bool should_attempt_recovery(const std::string& subsystem_name, int failure_count);
    
    // System information
    std::string get_system_architecture();
    std::string get_system_platform();
    std::string get_system_version();
    
} // namespace system_utils

} // namespace melvin
