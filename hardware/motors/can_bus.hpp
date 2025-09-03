#pragma once

#include "melvin_common.hpp"
#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <atomic>

namespace melvin {

// CAN frame structure
struct CANFrame {
    uint32_t can_id;
    uint8_t can_dlc;
    uint8_t data[8];
    bool is_extended;
    bool is_rtr;
    bool is_error;
    
    CANFrame() : can_id(0), can_dlc(0), is_extended(false), is_rtr(false), is_error(false) {
        std::memset(data, 0, sizeof(data));
    }
    
    CANFrame(uint32_t id, const std::vector<uint8_t>& payload, bool extended = false)
        : can_id(id), can_dlc(static_cast<uint8_t>(payload.size())), 
          is_extended(extended), is_rtr(false), is_error(false) {
        std::memset(data, 0, sizeof(data));
        std::copy(payload.begin(), payload.begin() + std::min(payload.size(), size_t(8)), data);
    }
};

// CAN bus interface
class CANBus {
public:
    virtual ~CANBus() = default;
    
    virtual Result<void> init(const std::string& interface, int bitrate) = 0;
    virtual Result<void> shutdown() = 0;
    virtual Result<void> send_frame(const CANFrame& frame) = 0;
    virtual Result<CANFrame> receive_frame() = 0;
    virtual Result<bool> is_connected() const = 0;
    virtual Result<void> set_filters(const std::vector<uint32_t>& ids) = 0;
    
    // Statistics
    virtual uint64_t get_tx_count() const = 0;
    virtual uint64_t get_rx_count() const = 0;
    virtual uint64_t get_error_count() const = 0;
};

// SocketCAN implementation for Linux
class SocketCAN : public CANBus {
public:
    SocketCAN();
    ~SocketCAN();
    
    Result<void> init(const std::string& interface, int bitrate) override;
    Result<void> shutdown() override;
    Result<void> send_frame(const CANFrame& frame) override;
    Result<CANFrame> receive_frame() override;
    Result<bool> is_connected() const override;
    Result<void> set_filters(const std::vector<uint32_t>& ids) override;
    
    uint64_t get_tx_count() const override { return tx_count_; }
    uint64_t get_rx_count() const override { return rx_count_; }
    uint64_t get_error_count() const override { return error_count_; }
    
    // Start/stop background thread
    Result<void> start_receive_thread();
    Result<void> stop_receive_thread();
    
    // Set receive callback
    using ReceiveCallback = std::function<void(const CANFrame&)>;
    void set_receive_callback(ReceiveCallback callback);

private:
    int socket_fd_;
    std::string interface_;
    int bitrate_;
    bool initialized_;
    
    std::thread receive_thread_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> tx_count_;
    std::atomic<uint64_t> rx_count_;
    std::atomic<uint64_t> error_count_;
    
    ReceiveCallback receive_callback_;
    mutable std::mutex callback_mutex_;
    
    void receive_loop();
    Result<void> setup_socket();
    Result<void> bind_socket();
    Result<void> set_bitrate();
    Result<void> set_filters_internal(const std::vector<uint32_t>& ids);
};

// CAN message types for RMD-X motors
namespace can_messages {
    
    // RMD-X command IDs
    constexpr uint8_t CMD_READ_MULTI_TURN_ANGLE = 0x92;
    constexpr uint8_t CMD_READ_SINGLE_TURN_ANGLE = 0x94;
    constexpr uint8_t CMD_READ_MOTOR_STATUS_1 = 0x9A;
    constexpr uint8_t CMD_READ_MOTOR_STATUS_2 = 0x9C;
    constexpr uint8_t CMD_READ_MOTOR_STATUS_3 = 0x9D;
    constexpr uint8_t CMD_SET_MOTOR_OFF = 0x80;
    constexpr uint8_t CMD_SET_MOTOR_STOP = 0x81;
    constexpr uint8_t CMD_SET_MOTOR_RUN = 0x88;
    constexpr uint8_t CMD_SET_MOTOR_POSITION = 0xA4;
    constexpr uint8_t CMD_SET_MOTOR_VELOCITY = 0xA2;
    constexpr uint8_t CMD_SET_MOTOR_TORQUE = 0xA1;
    
    // Helper functions for creating CAN frames
    CANFrame create_read_angle_frame(uint8_t motor_id);
    CANFrame create_read_status_frame(uint8_t motor_id, uint8_t status_type);
    CANFrame create_set_position_frame(uint8_t motor_id, double position, double velocity);
    CANFrame create_set_velocity_frame(uint8_t motor_id, double velocity);
    CANFrame create_set_torque_frame(uint8_t motor_id, double torque);
    CANFrame create_motor_control_frame(uint8_t motor_id, uint8_t command);
    
    // Helper functions for parsing CAN frames
    Result<double> parse_angle_response(const CANFrame& frame);
    Result<std::pair<double, double>> parse_status_response(const CANFrame& frame);
    Result<bool> parse_control_response(const CANFrame& frame);
    
} // namespace can_messages

} // namespace melvin
