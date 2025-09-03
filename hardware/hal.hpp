#pragma once

#include "melvin_common.hpp"
#include <vector>
#include <map>
#include <functional>

namespace melvin {

// Forward declarations
class MotorDriver;
class SensorDriver;
class CANBus;

// Hardware status
struct HardwareStatus {
    bool motors_ok;
    bool sensors_ok;
    bool can_bus_ok;
    std::string error_message;
    TimePoint last_update;
};

// Motor interface
class IMotor {
public:
    virtual ~IMotor() = default;
    
    virtual Result<void> init() = 0;
    virtual Result<void> shutdown() = 0;
    virtual Result<void> set_position(double position, double velocity = 0.0) = 0;
    virtual Result<void> set_velocity(double velocity) = 0;
    virtual Result<void> set_torque(double torque) = 0;
    virtual Result<void> enable() = 0;
    virtual Result<void> disable() = 0;
    virtual Result<void> emergency_stop() = 0;
    
    virtual Result<double> get_position() const = 0;
    virtual Result<double> get_velocity() const = 0;
    virtual Result<double> get_torque() const = 0;
    virtual Result<bool> is_enabled() const = 0;
    virtual Result<bool> is_error() const = 0;
    
    virtual MotorID get_id() const = 0;
    virtual MotorType get_type() const = 0;
    virtual std::string get_name() const = 0;
};

// Sensor interface
class ISensor {
public:
    virtual ~ISensor() = default;
    
    virtual Result<void> init() = 0;
    virtual Result<void> shutdown() = 0;
    virtual Result<void> update() = 0;
    virtual Result<void> calibrate() = 0;
    
    virtual SensorID get_id() const = 0;
    virtual SensorType get_type() const = 0;
    virtual std::string get_name() const = 0;
    virtual bool is_connected() const = 0;
    virtual TimePoint get_last_reading() const = 0;
};

// Touch sensor specific interface
class ITouchSensor : public ISensor {
public:
    virtual Result<bool> is_touched() const = 0;
    virtual Result<double> get_pressure() const = 0;
    virtual Result<std::vector<double>> get_pressure_map() const = 0;
};

// Temperature sensor specific interface
class ITemperatureSensor : public ISensor {
public:
    virtual Result<double> get_temperature() const = 0;
    virtual Result<double> get_humidity() const = 0;
};

// Camera sensor specific interface
class ICameraSensor : public ISensor {
public:
    virtual Result<void> start_streaming() = 0;
    virtual Result<void> stop_streaming() = 0;
    virtual Result<std::vector<uint8_t>> capture_frame() = 0;
    virtual Result<void> set_resolution(int width, int height) = 0;
    virtual Result<std::pair<int, int>> get_resolution() const = 0;
};

// Audio sensor specific interface
class IAudioSensor : public ISensor {
public:
    virtual Result<void> start_recording() = 0;
    virtual Result<void> stop_recording() = 0;
    virtual Result<std::vector<int16_t>> get_audio_data() = 0;
    virtual Result<void> play_audio(const std::vector<int16_t>& data) = 0;
    virtual Result<void> set_volume(double volume) = 0;
};

// Hardware Abstraction Layer
class HAL {
public:
    static HAL& instance();
    
    // Initialization and shutdown
    Result<void> init();
    Result<void> shutdown();
    
    // Motor management
    Result<MotorID> add_motor(MotorType type, const std::string& name, 
                             const std::map<std::string, std::string>& config);
    Result<void> remove_motor(MotorID id);
    Result<Ptr<IMotor>> get_motor(MotorID id);
    std::vector<Ptr<IMotor>> get_all_motors();
    
    // Sensor management
    Result<SensorID> add_sensor(SensorType type, const std::string& name,
                               const std::map<std::string, std::string>& config);
    Result<void> remove_sensor(SensorID id);
    Result<Ptr<ISensor>> get_sensor(SensorID id);
    std::vector<Ptr<ISensor>> get_all_sensors();
    
    // CAN bus management
    Result<void> init_can_bus(const std::string& interface, int bitrate);
    Result<void> shutdown_can_bus();
    Result<bool> is_can_bus_ok() const;
    
    // System status
    HardwareStatus get_status() const;
    Result<void> update();
    
    // Event callbacks
    using MotorEventCallback = std::function<void(MotorID, const std::string&)>;
    using SensorEventCallback = std::function<void(SensorID, const std::string&)>;
    
    void set_motor_event_callback(MotorEventCallback callback);
    void set_sensor_event_callback(SensorEventCallback callback);
    
    // Configuration
    Result<void> load_config(const std::string& config_file);
    Result<void> save_config(const std::string& config_file);

private:
    HAL() = default;
    ~HAL() = default;
    HAL(const HAL&) = delete;
    HAL& operator=(const HAL&) = delete;
    
    // Internal methods
    Result<Ptr<MotorDriver>> create_motor_driver(MotorType type, 
                                                const std::map<std::string, std::string>& config);
    Result<Ptr<SensorDriver>> create_sensor_driver(SensorType type,
                                                  const std::map<std::string, std::string>& config);
    
    // Member variables
    std::unique_ptr<CANBus> can_bus_;
    std::map<MotorID, Ptr<IMotor>> motors_;
    std::map<SensorID, Ptr<ISensor>> sensors_;
    
    MotorEventCallback motor_event_callback_;
    SensorEventCallback sensor_event_callback_;
    
    HardwareStatus status_;
    mutable std::shared_mutex hal_mutex_;
    
    bool initialized_;
    std::string config_file_;
};

// Factory functions for creating hardware components
namespace hardware_factory {
    Ptr<IMotor> create_rmd_x8_motor(const std::map<std::string, std::string>& config);
    Ptr<IMotor> create_rmd_x6_motor(const std::map<std::string, std::string>& config);
    
    Ptr<ITouchSensor> create_touch_sensor(const std::map<std::string, std::string>& config);
    Ptr<ITemperatureSensor> create_temperature_sensor(const std::map<std::string, std::string>& config);
    Ptr<ICameraSensor> create_camera_sensor(const std::map<std::string, std::string>& config);
    Ptr<IAudioSensor> create_audio_sensor(const std::map<std::string, std::string>& config);
}

} // namespace melvin
