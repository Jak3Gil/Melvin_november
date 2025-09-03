#include "logger.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cstring>

namespace melvin {

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

void Logger::init(const std::string& log_file, LogLevel level) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    if (initialized_) {
        return;
    }
    
    current_level_ = level;
    log_file_path_ = log_file;
    
    if (!log_file.empty()) {
        log_file_.open(log_file, std::ios::app);
        if (log_file_.is_open()) {
            output_stream_ = &log_file_;
            LOG_INFO("Logger initialized with file: " + log_file, "Logger");
        } else {
            output_stream_ = &std::cout;
            LOG_WARN("Failed to open log file, using console output", "Logger");
        }
    } else {
        output_stream_ = &std::cout;
        LOG_INFO("Logger initialized with console output", "Logger");
    }
    
    initialized_ = true;
}

void Logger::log(LogLevel level, const std::string& message, const std::string& module) {
    if (level < current_level_ || !initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    std::stringstream ss;
    ss << get_timestamp() << " [" << level_to_string(level) << "]";
    
    if (!module.empty()) {
        ss << " [" << module << "]";
    }
    
    ss << " " << message << std::endl;
    
    *output_stream_ << ss.str();
    output_stream_->flush();
}

void Logger::trace(const std::string& message, const std::string& module) {
    log(LogLevel::TRACE, message, module);
}

void Logger::debug(const std::string& message, const std::string& module) {
    log(LogLevel::DEBUG, message, module);
}

void Logger::info(const std::string& message, const std::string& module) {
    log(LogLevel::INFO, message, module);
}

void Logger::warn(const std::string& message, const std::string& module) {
    log(LogLevel::WARN, message, module);
}

void Logger::error(const std::string& message, const std::string& module) {
    log(LogLevel::ERROR, message, module);
}

void Logger::fatal(const std::string& message, const std::string& module) {
    log(LogLevel::FATAL, message, module);
}

void Logger::set_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    current_level_ = level;
    LOG_INFO("Log level set to: " + level_to_string(level), "Logger");
}

void Logger::flush() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (output_stream_) {
        output_stream_->flush();
    }
}

void Logger::shutdown() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (initialized_) {
        LOG_INFO("Logger shutting down", "Logger");
        if (log_file_.is_open()) {
            log_file_.close();
        }
        initialized_ = false;
    }
}

std::string Logger::level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

std::string Logger::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

// ScopedTimer implementation
ScopedTimer::ScopedTimer(const std::string& operation, const std::string& module)
    : operation_(operation), module_(module), start_time_(now()) {
    LOG_DEBUG("Starting: " + operation, module);
}

ScopedTimer::~ScopedTimer() {
    auto duration = since(start_time_);
    auto ms = to_ms(duration);
    LOG_DEBUG("Completed: " + operation_ + " in " + std::to_string(ms.count()) + "ms", module_);
}

} // namespace melvin
