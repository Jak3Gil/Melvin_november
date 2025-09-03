#pragma once

#include "melvin_common.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

namespace melvin {

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5
};

class Logger {
public:
    static Logger& instance();
    
    // Initialize logger
    void init(const std::string& log_file = "", LogLevel level = LogLevel::INFO);
    
    // Logging methods
    void trace(const std::string& message, const std::string& module = "");
    void debug(const std::string& message, const std::string& module = "");
    void info(const std::string& message, const std::string& module = "");
    void warn(const std::string& message, const std::string& module = "");
    void error(const std::string& message, const std::string& module = "");
    void fatal(const std::string& message, const std::string& module = "");
    
    // Set log level
    void set_level(LogLevel level);
    
    // Flush logs
    void flush();
    
    // Shutdown logger
    void shutdown();

private:
    Logger() = default;
    ~Logger() = default;
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    void log(LogLevel level, const std::string& message, const std::string& module);
    std::string level_to_string(LogLevel level);
    std::string get_timestamp();
    
    std::ofstream log_file_;
    std::ostream* output_stream_;
    LogLevel current_level_;
    std::mutex log_mutex_;
    bool initialized_;
    std::string log_file_path_;
};

// Convenience macros for logging
#define LOG_TRACE(msg, module) melvin::Logger::instance().trace(msg, module)
#define LOG_DEBUG(msg, module) melvin::Logger::instance().debug(msg, module)
#define LOG_INFO(msg, module) melvin::Logger::instance().info(msg, module)
#define LOG_WARN(msg, module) melvin::Logger::instance().warn(msg, module)
#define LOG_ERROR(msg, module) melvin::Logger::instance().error(msg, module)
#define LOG_FATAL(msg, module) melvin::Logger::instance().fatal(msg, module)

// Module-specific logging macros
#define LOG_TRACE_MOD(msg) LOG_TRACE(msg, __FUNCTION__)
#define LOG_DEBUG_MOD(msg) LOG_DEBUG(msg, __FUNCTION__)
#define LOG_INFO_MOD(msg) LOG_INFO(msg, __FUNCTION__)
#define LOG_WARN_MOD(msg) LOG_WARN(msg, __FUNCTION__)
#define LOG_ERROR_MOD(msg) LOG_ERROR(msg, __FUNCTION__)
#define LOG_FATAL_MOD(msg) LOG_FATAL(msg, __FUNCTION__)

// Performance logging
class ScopedTimer {
public:
    ScopedTimer(const std::string& operation, const std::string& module = "");
    ~ScopedTimer();
    
private:
    std::string operation_;
    std::string module_;
    TimePoint start_time_;
};

#define TIME_OPERATION(op, module) melvin::ScopedTimer timer(op, module)

} // namespace melvin
