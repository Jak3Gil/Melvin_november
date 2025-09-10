#include "logging.h"
#include <iostream>
#include <cstdarg>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace melvin {

std::mutex Logger::log_mutex;
std::ofstream Logger::log_file;
bool Logger::console_output = false;

void Logger::init_logging(const std::string& logfile, bool toConsole) {
    std::lock_guard<std::mutex> lock(log_mutex);
    
    if (log_file.is_open()) {
        log_file.close();
    }
    
    log_file.open(logfile, std::ios::app);
    console_output = toConsole;
    
    if (log_file.is_open()) {
        log_info("Logging initialized - file: %s, console: %s", 
                logfile.c_str(), toConsole ? "enabled" : "disabled");
    } else {
        std::cerr << "Failed to open log file: " << logfile << std::endl;
    }
}

void Logger::log_info(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message("INFO", fmt, args);
    va_end(args);
}

void Logger::log_warn(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message("WARN", fmt, args);
    va_end(args);
}

void Logger::log_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    log_message("ERROR", fmt, args);
    va_end(args);
}

void Logger::log_message(const std::string& level, const char* fmt, va_list args) {
    std::lock_guard<std::mutex> lock(log_mutex);
    
    // Format message
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    
    // Create log entry
    std::string timestamp = get_timestamp();
    std::string log_entry = "[" + timestamp + "] [" + level + "] " + buffer + "\n";
    
    // Write to file
    if (log_file.is_open()) {
        log_file << log_entry;
        log_file.flush();
    }
    
    // Write to console if enabled
    if (console_output) {
        if (level == "ERROR") {
            std::cerr << log_entry;
        } else {
            std::cout << log_entry;
        }
    }
}

std::string Logger::get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
    
    return oss.str();
}

} // namespace melvin