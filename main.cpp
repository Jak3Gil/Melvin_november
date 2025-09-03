#include "melvin_core.hpp"
#include "logger.hpp"
#include <iostream>
#include <csignal>
#include <cstdlib>

using namespace melvin;

// Global signal handler
std::atomic<bool> g_running(true);

void signal_handler(int signal) {
    LOG_INFO("Received signal " + std::to_string(signal) + ", shutting down...", "Main");
    g_running = false;
}

void setup_signal_handlers() {
    signal(SIGINT, signal_handler);   // Ctrl+C
    signal(SIGTERM, signal_handler);  // Termination request
    signal(SIGQUIT, signal_handler);  // Quit request
    
    // Ignore SIGPIPE (broken pipe)
    signal(SIGPIPE, SIG_IGN);
}

void print_banner() {
    std::cout << R"(
    __  ___      _ _             
   |  \/  |     | (_)            
   | .  . | ___ | |_ _ __   __ _ 
   | |\/| |/ _ \| | | '_ \ / _` |
   | |  | | (_) | | | | | | (_| |
   |_|  |_|\___/|_|_|_| |_|\__, |
                             __/ |
                            |___/ 
    )" << std::endl;
    
    std::cout << "Melvin Humanoid Robot System v" << VERSION << std::endl;
    std::cout << "Built on " << BUILD_DATE << " at " << BUILD_TIME << std::endl;
    std::cout << "================================================" << std::endl;
}

void print_system_info() {
    std::cout << "\nSystem Information:" << std::endl;
    std::cout << "  C++ Standard: C++20" << std::endl;
    std::cout << "  Architecture: " << sizeof(void*) * 8 << "-bit" << std::endl;
    std::cout << "  Max Nodes: " << MAX_NODES << std::endl;
    std::cout << "  Max Connections: " << MAX_CONNECTIONS << std::endl;
    std::cout << "  Max Sensors: " << MAX_SENSORS << std::endl;
    std::cout << "  Max Motors: " << MAX_MOTORS << std::endl;
    std::cout << "  Default Port: " << DEFAULT_PORT << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        // Print banner and system info
        print_banner();
        print_system_info();
        
        // Setup signal handlers
        setup_signal_handlers();
        
        LOG_INFO("Starting Melvin system...", "Main");
        
        // Initialize logger
        Logger::instance().init("/var/melvin/logs/melvin.log", LogLevel::INFO);
        LOG_INFO("Logger initialized", "Main");
        
        // Initialize Melvin core
        auto& core = MelvinCore::instance();
        auto init_result = core.init();
        
        if (init_result.is_error()) {
            LOG_FATAL("Failed to initialize Melvin core: " + init_result.message, "Main");
            return EXIT_FAILURE;
        }
        
        LOG_INFO("Melvin core initialized successfully", "Main");
        
        // Start all systems
        auto start_result = core.start();
        if (start_result.is_error()) {
            LOG_FATAL("Failed to start Melvin core: " + start_result.message, "Main");
            return EXIT_FAILURE;
        }
        
        LOG_INFO("Melvin core started successfully", "Main");
        LOG_INFO("System is now running. Press Ctrl+C to stop.", "Main");
        
        // Main loop
        while (g_running) {
            // Check system health
            auto health = core.get_system_health();
            if (health.is_error()) {
                LOG_ERROR("System health check failed: " + health.message, "Main");
            } else {
                auto status = *health;
                if (status.status != SystemStatus::RUNNING) {
                    LOG_WARN("System status: " + to_string(status.status), "Main");
                }
            }
            
            // Sleep for a short interval
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        LOG_INFO("Shutdown sequence initiated", "Main");
        
        // Stop all systems
        auto stop_result = core.stop();
        if (stop_result.is_error()) {
            LOG_ERROR("Error during shutdown: " + stop_result.message, "Main");
        }
        
        // Shutdown logger
        Logger::instance().shutdown();
        
        LOG_INFO("Melvin system shutdown complete", "Main");
        std::cout << "\nMelvin system has been shut down safely." << std::endl;
        
        return EXIT_SUCCESS;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown fatal error occurred" << std::endl;
        return EXIT_FAILURE;
    }
}
