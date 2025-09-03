#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <optional>
#include <variant>

namespace melvin {

// Version information
constexpr const char* VERSION = "1.0.0";
constexpr const char* BUILD_DATE = __DATE__;
constexpr const char* BUILD_TIME = __TIME__;

// System constants
constexpr size_t MAX_NODES = 10000;
constexpr size_t MAX_CONNECTIONS = 50000;
constexpr size_t MAX_SENSORS = 100;
constexpr size_t MAX_MOTORS = 50;
constexpr size_t MAX_THREADS = 16;
constexpr size_t BUFFER_SIZE = 8192;
constexpr int DEFAULT_PORT = 8080;

// Time types
using TimePoint = std::chrono::steady_clock::time_point;
using Duration = std::chrono::steady_clock::duration;
using Milliseconds = std::chrono::milliseconds;
using Microseconds = std::chrono::microseconds;

// ID types
using NodeID = uint64_t;
using ConnectionID = uint64_t;
using SensorID = uint32_t;
using MotorID = uint32_t;
using ProcessID = uint32_t;

// Confidence and weight types
using Confidence = float;  // 0.0 to 1.0
using Weight = float;      // Any real number
using Priority = uint8_t;  // 0-255

// Status enums
enum class SystemStatus {
    INITIALIZING,
    RUNNING,
    PAUSED,
    ERROR,
    SHUTDOWN
};

enum class NodeType {
    INPUT,      // Sensor data, external input
    CONCEPT,    // Internal processing node
    OUTPUT,     // Motor control, external output
    MEMORY,     // Persistent storage
    LEARNING    // ML/rule-based learning
};

enum class ConnectionType {
    EXCITATORY,     // Positive weight
    INHIBITORY,     // Negative weight
    BIDIRECTIONAL,  // Both directions
    CONDITIONAL     // Only active under conditions
};

enum class SensorType {
    TOUCH,
    TEMPERATURE,
    CAMERA,
    MICROPHONE,
    SPEAKER,
    PROXIMITY,
    IMU,
    BATTERY
};

enum class MotorType {
    RMD_X8,
    RMD_X6,
    SERVO,
    STEPPER
};

// Error codes
enum class ErrorCode {
    SUCCESS = 0,
    INVALID_PARAMETER,
    NOT_FOUND,
    ALREADY_EXISTS,
    PERMISSION_DENIED,
    RESOURCE_UNAVAILABLE,
    TIMEOUT,
    INTERNAL_ERROR,
    NETWORK_ERROR,
    HARDWARE_ERROR
};

// Result type for error handling
template<typename T>
struct Result {
    ErrorCode code;
    std::optional<T> value;
    std::string message;
    
    Result(ErrorCode c, T v) : code(c), value(v), message("") {}
    Result(ErrorCode c, std::string msg) : code(c), value(std::nullopt), message(std::move(msg)) {}
    
    bool is_success() const { return code == ErrorCode::SUCCESS; }
    bool is_error() const { return code != ErrorCode::SUCCESS; }
    
    T& operator*() { return *value; }
    const T& operator*() const { return *value; }
    
    T* operator->() { return value.operator->(); }
    const T* operator->() const { return value.operator->(); }
};

// Success/Error helper functions
template<typename T>
Result<T> Success(T value) {
    return Result<T>(ErrorCode::SUCCESS, value);
}

template<typename T>
Result<T> Error(ErrorCode code, std::string message) {
    return Result<T>(code, std::move(message));
}

// Thread-safe types
template<typename T>
using Atomic = std::atomic<T>;

template<typename T>
using SharedMutex = std::shared_mutex;

template<typename T>
using SharedLock = std::shared_lock<std::shared_mutex>;

template<typename T>
using UniqueLock = std::unique_lock<std::shared_mutex>;

// Smart pointer types
template<typename T>
using Ptr = std::unique_ptr<T>;

template<typename T>
using SharedPtr = std::shared_ptr<T>;

template<typename T>
using WeakPtr = std::weak_ptr<T>;

// Function types
template<typename... Args>
using Callback = std::function<void(Args...)>;

template<typename T>
using Producer = std::function<T()>;

template<typename T>
using Consumer = std::function<void(const T&)>;

// Utility functions
inline TimePoint now() {
    return std::chrono::steady_clock::now();
}

inline Duration since(TimePoint start) {
    return now() - start;
}

inline Milliseconds to_ms(Duration d) {
    return std::chrono::duration_cast<Milliseconds>(d);
}

inline Microseconds to_us(Duration d) {
    return std::chrono::duration_cast<Microseconds>(d);
}

// String utilities
std::string to_string(SystemStatus status);
std::string to_string(NodeType type);
std::string to_string(ConnectionType type);
std::string to_string(SensorType type);
std::string to_string(MotorType type);
std::string to_string(ErrorCode code);

// Validation utilities
bool is_valid_node_id(NodeID id);
bool is_valid_confidence(Confidence c);
bool is_valid_weight(Weight w);
bool is_valid_priority(Priority p);

} // namespace melvin
