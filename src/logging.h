#pragma once

#include <string>
#include <mutex>
#include <fstream>

namespace melvin {

class Logger {
private:
    static std::mutex log_mutex;
    static std::ofstream log_file;
    static bool console_output;
    
public:
    static void init_logging(const std::string& logfile, bool toConsole);
    static void log_info(const char* fmt, ...);
    static void log_warn(const char* fmt, ...);
    static void log_error(const char* fmt, ...);
    
private:
    static void log_message(const std::string& level, const char* fmt, va_list args);
    static std::string get_timestamp();
};

} // namespace melvin