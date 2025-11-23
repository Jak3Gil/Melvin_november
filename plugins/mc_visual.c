#define _POSIX_C_SOURCE 200809L
#include "../melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <time.h>
#include <errno.h>

// External helpers (from melvin.c)
extern uint64_t alloc_node(Brain *);
extern void add_edge(Brain *, uint64_t, uint64_t, float, uint32_t);

// Web server state
static int server_fd = -1;
static int server_initialized = 0;
static const int VISUAL_PORT = 8080;

// Simple HTTP response helper
static void send_http_response(int client_fd, const char *body, const char *content_type) {
    char response[8192];
    snprintf(response, sizeof(response),
             "HTTP/1.1 200 OK\r\n"
             "Content-Type: %s\r\n"
             "Content-Length: %zu\r\n"
             "Access-Control-Allow-Origin: *\r\n"
             "Connection: close\r\n"
             "\r\n"
             "%s", content_type, strlen(body), body);
    send(client_fd, response, strlen(response), 0);
}

// Generate graph data as JSON
static void generate_graph_json(Brain *g, char *json_out, size_t json_size) {
    uint64_t n = g->header->num_nodes;
    uint64_t e_count = g->header->num_edges;
    
    // Limit nodes/edges for visualization performance
    uint64_t max_nodes = n < 10000 ? n : 10000;
    uint64_t max_edges = e_count < 50000 ? e_count : 50000;
    
    size_t pos = 0;
    
    // Start JSON
    pos += snprintf(json_out + pos, json_size - pos,
                    "{\n"
                    "  \"tick\": %llu,\n"
                    "  \"num_nodes\": %llu,\n"
                    "  \"num_edges\": %llu,\n"
                    "  \"nodes\": [\n",
                    (unsigned long long)g->header->tick,
                    (unsigned long long)n,
                    (unsigned long long)e_count);
    
    // Output nodes (sample)
    int first_node = 1;
    for (uint64_t i = 0; i < max_nodes && pos < json_size - 500; i++) {
        if (!first_node) pos += snprintf(json_out + pos, json_size - pos, ",\n");
        first_node = 0;
        
        Node *node = &g->nodes[i];
        pos += snprintf(json_out + pos, json_size - pos,
                        "    {\"id\": %llu, \"a\": %.3f, \"kind\": %u, \"value\": %.2f}",
                        (unsigned long long)i,
                        node->a,
                        node->kind,
                        node->value);
    }
    
    pos += snprintf(json_out + pos, json_size - pos, "\n  ],\n  \"edges\": [\n");
    
    // Output edges (sample)
    int first_edge = 1;
    for (uint64_t i = 0; i < max_edges && pos < json_size - 500; i++) {
        Edge *edge = &g->edges[i];
        if (edge->src < max_nodes && edge->dst < max_nodes) {
            if (!first_edge) pos += snprintf(json_out + pos, json_size - pos, ",\n");
            first_edge = 0;
            
            pos += snprintf(json_out + pos, json_size - pos,
                            "    {\"src\": %llu, \"dst\": %llu, \"w\": %.3f, \"flags\": %u}",
                            (unsigned long long)edge->src,
                            (unsigned long long)edge->dst,
                            edge->w,
                            edge->flags);
        }
    }
    
    pos += snprintf(json_out + pos, json_size - pos, "\n  ]\n}\n");
}

// HTML/JavaScript visualization page
static const char *html_page = 
"<!DOCTYPE html>\n"
"<html>\n"
"<head>\n"
"  <title>Melvin Hyperspace Visualization</title>\n"
"  <style>\n"
"    body { margin: 0; padding: 20px; background: #000; color: #0f0; font-family: monospace; }\n"
"    #graph { width: 100%; height: 90vh; border: 1px solid #0f0; background: #001; }\n"
"    #info { position: fixed; top: 10px; right: 10px; background: rgba(0,255,0,0.1); padding: 10px; border: 1px solid #0f0; }\n"
"    .node { fill: #0f0; stroke: #0ff; stroke-width: 1; }\n"
"    .edge { stroke: #00f; stroke-opacity: 0.3; stroke-width: 1; }\n"
"    .node.active { fill: #ff0; stroke: #fff; stroke-width: 2; }\n"
"  </style>\n"
"</head>\n"
"<body>\n"
"  <div id=\"info\">\n"
"    <div>Tick: <span id=\"tick\">0</span></div>\n"
"    <div>Nodes: <span id=\"nodes\">0</span></div>\n"
"    <div>Edges: <span id=\"edges\">0</span></div>\n"
"  </div>\n"
"  <svg id=\"graph\"></svg>\n"
"  <script>\n"
"    const svg = document.getElementById('graph');\n"
"    let graphData = null;\n"
"    \n"
"    function fetchGraph() {\n"
"      fetch('/graph.json')\n"
"        .then(r => r.json())\n"
"        .then(data => {\n"
"          graphData = data;\n"
"          document.getElementById('tick').textContent = data.tick;\n"
"          document.getElementById('nodes').textContent = data.num_nodes;\n"
"          document.getElementById('edges').textContent = data.num_edges;\n"
"          renderGraph(data);\n"
"        })\n"
"        .catch(e => console.error('Fetch error:', e));\n"
"    }\n"
"    \n"
"    function renderGraph(data) {\n"
"      svg.innerHTML = '';\n"
"      const width = svg.clientWidth;\n"
"      const height = svg.clientHeight;\n"
"      \n"
"      // Simple force-directed layout\n"
"      const nodes = data.nodes.map((n, i) => ({\n"
"        ...n,\n"
"        x: Math.random() * width,\n"
"        y: Math.random() * height,\n"
"        vx: 0, vy: 0\n"
"      }));\n"
"      \n"
"      // Render edges\n"
"      data.edges.forEach(e => {\n"
"        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');\n"
"        const src = nodes[e.src];\n"
"        const dst = nodes[e.dst];\n"
"        if (src && dst) {\n"
"          line.setAttribute('x1', src.x);\n"
"          line.setAttribute('y1', src.y);\n"
"          line.setAttribute('x2', dst.x);\n"
"          line.setAttribute('y2', dst.y);\n"
"          line.setAttribute('class', 'edge');\n"
"          line.setAttribute('stroke-width', Math.abs(e.w));\n"
"          svg.appendChild(line);\n"
"        }\n"
"      });\n"
"      \n"
"      // Render nodes\n"
"      nodes.forEach(n => {\n"
"        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');\n"
"        circle.setAttribute('cx', n.x);\n"
"        circle.setAttribute('cy', n.y);\n"
"        circle.setAttribute('r', Math.max(2, n.a * 10));\n"
"        circle.setAttribute('class', n.a > 0.5 ? 'node active' : 'node');\n"
"        svg.appendChild(circle);\n"
"      });\n"
"    }\n"
"    \n"
"    // Update every 500ms\n"
"    setInterval(fetchGraph, 500);\n"
"    fetchGraph();\n"
"  </script>\n"
"</body>\n"
"</html>\n";

// MC function: Start visualization web server
void mc_visual_server(Brain *g, uint64_t node_id) {
    static int server_running = 0;
    
    if (server_running) {
        // Server already running
        if (g->nodes[node_id].a < 0.5f) {
            // Stop server if node deactivated
            if (server_fd >= 0) {
                close(server_fd);
                server_fd = -1;
                server_running = 0;
                printf("[mc_visual] Server stopped\n");
            }
        }
        return;
    }
    
    // Only start if node is activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    // Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket");
        return;
    }
    
    // Set socket options
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    // Bind to port
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(VISUAL_PORT);
    
    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        server_fd = -1;
        return;
    }
    
    // Listen
    if (listen(server_fd, 5) < 0) {
        perror("listen");
        close(server_fd);
        server_fd = -1;
        return;
    }
    
    // Set non-blocking
    int flags = fcntl(server_fd, F_GETFL, 0);
    fcntl(server_fd, F_SETFL, flags | O_NONBLOCK);
    
    server_running = 1;
    printf("[mc_visual] Web server started on port %d\n", VISUAL_PORT);
    printf("[mc_visual] Open http://localhost:%d or http://<jetson-ip>:%d in browser\n", 
           VISUAL_PORT, VISUAL_PORT);
    
    // Create success node
    uint64_t success_node = alloc_node(g);
    if (success_node != UINT64_MAX) {
        Node *sn = &g->nodes[success_node];
        sn->kind = NODE_KIND_META;
        sn->a = 1.0f;
        sn->value = 0x56495355; // "VISU"
        add_edge(g, node_id, success_node, 1.0f, EDGE_FLAG_CONTROL);
    }
}

// MC function: Handle HTTP requests (non-blocking)
void mc_visual_serve(Brain *g, uint64_t node_id) {
    if (server_fd < 0) return;
    
    // Accept connections (non-blocking)
    struct sockaddr_in client_addr;
    socklen_t addr_len = sizeof(client_addr);
    int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &addr_len);
    
    if (client_fd < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            perror("accept");
        }
        return;
    }
    
    // Read request (simplified)
    char buffer[4096] = {0};
    ssize_t n = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
    
    if (n > 0) {
        buffer[n] = '\0';
        
        // Simple routing
        if (strstr(buffer, "GET / ") != NULL || strstr(buffer, "GET /index.html") != NULL) {
            send_http_response(client_fd, html_page, "text/html");
        } else if (strstr(buffer, "GET /graph.json") != NULL) {
            // Generate JSON graph data
            char json_data[1024 * 1024] = {0}; // 1MB buffer
            generate_graph_json(g, json_data, sizeof(json_data));
            send_http_response(client_fd, json_data, "application/json");
        } else {
            // 404
            send_http_response(client_fd, "Not Found", "text/plain");
        }
    }
    
    close(client_fd);
}

