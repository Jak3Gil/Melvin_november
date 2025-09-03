#pragma once

#include "melvin_common.hpp"
#include "can_bus.hpp"
#include <memory>
#include <deque>
#include <mutex>

namespace melvin {

// RMD-X motor configuration
struct RMDXConfig {
    uint8_t motor_id;
    double gear_ratio;
    double max_velocity;
    double max_torque;
    double position_offset;
    bool invert_direction;
    std::string name;
    
    RMDXConfig() : motor_id(0), gear_ratio(1.0), max_velocity(1000.0), 
                   max_torque(10.0), position_offset(0.0), invert_direction(false) {}
};

// RMD-X motor status
struct RMDXStatus {
    double position;           // Current position in radians
    double velocity;           // Current velocity in rad/s
    double torque;             // Current torque in Nm
    double temperature;        // Motor temperature in Celsius
    double voltage;            // Motor voltage in V
    double current;            // Motor current in A
    bool is_enabled;           // Motor enabled status
    bool is_error;             // Error status
    uint16_t error_code;       // Error code if any
    TimePoint timestamp;       // When this status was read
    
    RMDXStatus() : position(0.0), velocity(0.0), torque(0.0), temperature(0.0),
                   voltage(0.0), current(0.0), is_enabled(false), is_error(false), 
                   error_code(0) {}
};

// RMD-X motor driver implementation
class RMDXDriver : public IMotor {
public:
    explicit RMDXDriver(const RMDXConfig& config, std::shared_ptr<CANBus> can_bus);
    ~RMDXDriver() override;
    
    // IMotor interface implementation
    Result<void> init() override;
    Result<void> shutdown() override;
    Result<void> set_position(double position, double velocity = 0.0) override;
    Result<void> set_velocity(double velocity) override;
    Result<void> set_torque(double torque) override;
    Result<void> enable() override;
    Result<void> disable() override;
    Result<void> emergency_stop() override;
    
    Result<double> get_position() const override;
    Result<double> get_velocity() const override;
    Result<double> get_torque() const override;
    Result<bool> is_enabled() const override;
    Result<bool> is_error() const override;
    
    MotorID get_id() const override { return static_cast<MotorID>(config_.motor_id); }
    MotorType get_type() const override { return MotorType::RMD_X8; }
    std::string get_name() const override { return config_.name; }
    
    // RMD-X specific methods
    Result<void> set_control_mode(uint8_t mode);
    Result<void> read_status();
    Result<RMDXStatus> get_detailed_status() const;
    Result<void> clear_errors();
    Result<void> set_limits(double max_velocity, double max_torque);
    
    // Configuration
    Result<void> update_config(const RMDXConfig& config);
    RMDXConfig get_config() const;
    
    // Statistics
    uint64_t get_command_count() const { return command_count_; }
    uint64_t get_response_count() const { return response_count_; }
    uint64_t get_error_count() const { return error_count_; }

private:
    // Internal methods
    Result<void> send_command(const CANFrame& frame);
    Result<CANFrame> wait_for_response(uint8_t expected_cmd, Duration timeout = Milliseconds(100));
    Result<void> process_response(const CANFrame& frame);
    
    // Command queue management
    void queue_command(const CANFrame& frame);
    Result<void> process_command_queue();
    
    // Status update methods
    void update_position_from_response(const CANFrame& frame);
    void update_status_from_response(const CANFrame& frame);
    
    // Utility methods
    double raw_to_radians(int32_t raw_value) const;
    int32_t radians_to_raw(double radians) const;
    double raw_to_velocity(int16_t raw_value) const;
    int16_t velocity_to_raw(double velocity) const;
    double raw_to_torque(int16_t raw_value) const;
    int16_t torque_to_raw(double torque) const;
    
    // Member variables
    RMDXConfig config_;
    std::shared_ptr<CANBus> can_bus_;
    RMDXStatus current_status_;
    
    std::deque<CANFrame> command_queue_;
    mutable std::mutex status_mutex_;
    mutable std::mutex queue_mutex_;
    
    std::atomic<uint64_t> command_count_;
    std::atomic<uint64_t> response_count_;
    std::atomic<uint64_t> error_count_;
    
    bool initialized_;
    std::thread command_thread_;
    std::atomic<bool> running_;
    
    // Constants
    static constexpr double RAW_TO_RAD_FACTOR = 0.01;  // 0.01 degrees per LSB
    static constexpr double RAW_TO_VEL_FACTOR = 0.01;  // 0.01 dps per LSB
    static constexpr double RAW_TO_TORQUE_FACTOR = 0.01; // 0.01 Nm per LSB
    static constexpr Duration COMMAND_TIMEOUT = Milliseconds(50);
    static constexpr size_t MAX_QUEUE_SIZE = 100;
};

// Factory function for creating RMD-X drivers
namespace rmd_x_factory {
    Ptr<IMotor> create_rmd_x8_driver(const RMDXConfig& config, std::shared_ptr<CANBus> can_bus);
    Ptr<IMotor> create_rmd_x6_driver(const RMDXConfig& config, std::shared_ptr<CANBus> can_bus);
    
    // Default configurations
    RMDXConfig get_default_rmd_x8_config();
    RMDXConfig get_default_rmd_x6_config();
}

} // namespace melvin
