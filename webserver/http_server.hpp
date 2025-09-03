#pragma once

#include "melvin_common.hpp"
#include <string>
#include <map>
#include <functional>
#include <thread>
#include <atomic>
#include <memory>

namespace melvin {

// Forward declarations
class APIHandler;
class WebSocketHandler;
class StaticFileServer;

// HTTP request structure
struct HTTPRequest {
    std::string method;
    std::string path;
    std::string query_string;
    std::map<std::string, std::string> headers;
    std::map<std::string, std::string> query_params;
    std::string body;
    std::string client_ip;
    uint16_t client_port;
    
    HTTPRequest() : client_port(0) {}
};

// HTTP response structure
struct HTTPResponse {
    int status_code;
    std::map<std::string, std::string> headers;
    std::string body;
    std::string content_type;
    
    HTTPResponse() : status_code(200), content_type("application/json") {}
};

// WebSocket message structure
struct WebSocketMessage {
    std::string type;
    std::string data;
    std::string client_id;
    TimePoint timestamp;
    
    WebSocketMessage() {}
};

// Server configuration
struct ServerConfig {
    int port;
    std::string host;
    std::string static_files_path;
    std::string api_prefix;
    size_t max_connections;
    Duration request_timeout;
    bool enable_cors;
    bool enable_compression;
    std::vector<std::string> allowed_origins;
    
    ServerConfig() : port(DEFAULT_PORT), host("0.0.0.0"), static_files_path("ui/frontend"),
                    api_prefix("/api"), max_connections(1000), request_timeout(Seconds(30)),
                    enable_cors(true), enable_compression(true) {}
};

// Server statistics
struct ServerStatistics {
    uint64_t total_requests;
    uint64_t total_responses;
    uint64_t active_connections;
    uint64_t total_websocket_connections;
    Duration uptime;
    double requests_per_second;
    std::map<int, uint64_t> status_code_counts;
    std::map<std::string, uint64_t> endpoint_counts;
    TimePoint last_calculation;
    
    ServerStatistics() : total_requests(0), total_responses(0), active_connections(0),
                        total_websocket_connections(0), requests_per_second(0.0) {}
};

// HTTP Server - Main web server for Melvin
class HTTPServer {
public:
    static HTTPServer& instance();
    
    // Initialization and shutdown
    Result<void> init(const ServerConfig& config = ServerConfig());
    Result<void> shutdown();
    
    // Server control
    Result<void> start();
    Result<void> stop();
    Result<void> restart();
    Result<bool> is_running() const;
    
    // Configuration
    ServerConfig get_config() const;
    Result<void> update_config(const ServerConfig& config);
    
    // API endpoint registration
    using EndpointHandler = std::function<HTTPResponse(const HTTPRequest&)>;
    Result<void> register_endpoint(const std::string& method, const std::string& path, 
                                 EndpointHandler handler);
    Result<void> unregister_endpoint(const std::string& method, const std::string& path);
    
    // WebSocket management
    Result<void> broadcast_message(const WebSocketMessage& message);
    Result<void> send_message_to_client(const std::string& client_id, const WebSocketMessage& message);
    Result<std::vector<std::string>> get_connected_clients() const;
    
    // Static file serving
    Result<void> set_static_files_path(const std::string& path);
    Result<void> add_mime_type(const std::string& extension, const std::string& mime_type);
    
    // Middleware support
    using Middleware = std::function<HTTPResponse(const HTTPRequest&, EndpointHandler)>;
    Result<void> add_middleware(const std::string& name, Middleware middleware);
    Result<void> remove_middleware(const std::string& name);
    
    // Statistics and monitoring
    ServerStatistics get_statistics() const;
    Result<void> reset_statistics();
    
    // Logging
    Result<void> set_log_level(LogLevel level);
    Result<void> enable_access_logging(bool enable);
    Result<void> set_log_file(const std::string& filename);

private:
    HTTPServer() = default;
    ~HTTPServer() = default;
    HTTPServer(const HTTPServer&) = delete;
    HTTPServer& operator=(const HTTPServer&) = delete;
    
    // Internal methods
    Result<void> setup_socket();
    Result<void> bind_socket();
    Result<void> listen_for_connections();
    Result<void> handle_connection(int client_socket);
    Result<HTTPRequest> parse_request(const std::string& raw_request);
    Result<HTTPResponse> process_request(const HTTPRequest& request);
    Result<std::string> serialize_response(const HTTPResponse& response);
    Result<void> apply_middleware(HTTPRequest& request, HTTPResponse& response);
    
    // Member variables
    ServerConfig config_;
    ServerStatistics statistics_;
    
    int server_socket_;
    std::atomic<bool> running_;
    std::thread accept_thread_;
    std::vector<std::thread> worker_threads_;
    
    std::unique_ptr<APIHandler> api_handler_;
    std::unique_ptr<WebSocketHandler> websocket_handler_;
    std::unique_ptr<StaticFileServer> static_file_server_;
    
    std::map<std::string, std::map<std::string, EndpointHandler>> endpoints_;
    std::map<std::string, Middleware> middleware_;
    std::map<std::string, std::string> mime_types_;
    
    mutable std::shared_mutex server_mutex_;
    bool initialized_;
    
    // Background tasks
    void accept_loop();
    void worker_loop();
    void update_statistics();
};

// Built-in API endpoints
namespace builtin_endpoints {
    
    // System status
    HTTPResponse get_system_status(const HTTPRequest& request);
    HTTPResponse get_system_health(const HTTPRequest& request);
    HTTPResponse get_system_info(const HTTPRequest& request);
    
    // Hardware status
    HTTPResponse get_hardware_status(const HTTPRequest& request);
    HTTPResponse get_motor_status(const HTTPRequest& request);
    HTTPResponse get_sensor_status(const HTTPRequest& request);
    
    // Brain graph
    HTTPResponse get_brain_graph(const HTTPRequest& request);
    HTTPResponse get_node_info(const HTTPRequest& request);
    HTTPResponse get_connection_info(const HTTPRequest& request);
    
    // Learning engine
    HTTPResponse get_learning_status(const HTTPRequest& request);
    HTTPResponse get_rules(const HTTPRequest& request);
    HTTPResponse create_rule(const HTTPRequest& request);
    
    // Control endpoints
    HTTPResponse send_command(const HTTPRequest& request);
    HTTPResponse set_motor_position(const HTTPRequest& request);
    HTTPResponse set_motor_velocity(const HTTPRequest& request);
    
    // Configuration
    HTTPResponse get_config(const HTTPRequest& request);
    HTTPResponse update_config(const HTTPRequest& request);
    HTTPResponse reset_config(const HTTPRequest& request);
    
} // namespace builtin_endpoints

// Utility functions for HTTP operations
namespace http_utils {
    
    // URL parsing
    std::map<std::string, std::string> parse_query_string(const std::string& query);
    std::string url_encode(const std::string& str);
    std::string url_decode(const std::string& str);
    
    // Headers
    std::string get_header_value(const std::map<std::string, std::string>& headers, 
                                const std::string& name);
    void set_header(std::map<std::string, std::string>& headers, 
                   const std::string& name, const std::string& value);
    
    // Response helpers
    HTTPResponse create_json_response(int status_code, const std::string& json_data);
    HTTPResponse create_error_response(int status_code, const std::string& error_message);
    HTTPResponse create_success_response(const std::string& message = "Success");
    
    // CORS
    void add_cors_headers(HTTPResponse& response, const std::vector<std::string>& allowed_origins);
    
    // Compression
    std::string compress_response(const std::string& data);
    std::string decompress_request(const std::string& data);
    
} // namespace http_utils

} // namespace melvin
