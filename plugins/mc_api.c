#define _POSIX_C_SOURCE 200809L
#include "../melvin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/wait.h>

// External helpers (from melvin.c)
extern uint64_t alloc_node(Brain *);
extern void add_edge(Brain *, uint64_t, uint64_t, float, uint32_t);

// Use curl for HTTP requests (if available) or fallback to system calls
// For simplicity, we'll use system() with curl command
// In production, you'd want to link against libcurl

// Helper: Extract URL from connected nodes or use default
static int get_url_from_graph(Brain *g, uint64_t node_id, char *url_out, size_t url_size) {
    // Look for connected DATA nodes that might contain URL
    // For now, we'll use a simple approach: check if there's a connected node with URL-like data
    // In a full system, the graph would pass the URL via edges
    
    // Try to find a connected node with URL info
    for (uint64_t i = 0; i < g->header->num_edges; i++) {
        Edge *e = &g->edges[i];
        if (e->src == node_id && g->nodes[e->dst].kind == NODE_KIND_DATA) {
            // Could contain URL - for now we'll use a default or parameter
        }
    }
    
    // Return 0 to indicate we should use a default or parameter
    return 0;
}

// Helper: Store API response in graph as nodes
static void store_response_in_graph(Brain *g, uint64_t node_id, const char *response, size_t response_len) {
    if (!response || response_len == 0) return;
    
    // Create a response node
    uint64_t response_node = alloc_node(g);
    if (response_node == UINT64_MAX) return;
    
    Node *rn = &g->nodes[response_node];
    rn->kind = NODE_KIND_DATA;
    rn->a = 0.7f;
    
    // Store response hash in value (for identification)
    uint32_t hash = 0;
    size_t len = response_len < 32 ? response_len : 32;
    for (size_t i = 0; i < len; i++) {
        hash = hash * 31 + (unsigned char)response[i];
    }
    rn->value = (float)hash;
    
    // Link response to API call node
    add_edge(g, node_id, response_node, 1.0f, EDGE_FLAG_CONTROL);
    
    // Store response bytes as sequence of nodes (first 256 bytes as byte nodes)
    // For longer responses, we'd need a blob storage system
    size_t bytes_to_store = response_len < 256 ? response_len : 256;
    for (size_t i = 0; i < bytes_to_store; i++) {
        uint8_t byte = (uint8_t)response[i];
        uint64_t byte_node = alloc_node(g);
        if (byte_node != UINT64_MAX) {
            Node *bn = &g->nodes[byte_node];
            bn->kind = NODE_KIND_DATA;
            bn->a = 0.5f;
            bn->value = (float)byte;
            
            // Link to response node
            add_edge(g, response_node, byte_node, 1.0f, EDGE_FLAG_SEQ);
        }
    }
    
    printf("[mc_api] Stored %zu bytes of response in graph\n", bytes_to_store);
}

// MC function: Make HTTP GET request
void mc_api_get(Brain *g, uint64_t node_id) {
    static char last_url[512] = {0};
    static int last_request_tick = -1;
    
    // Only run if node is activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    // Prevent rapid-fire requests (rate limiting)
    if (last_request_tick >= 0 && 
        (int)g->header->tick - last_request_tick < 10) {
        g->nodes[node_id].a = 0.0f;
        return;
    }
    
    char url[512] = {0};
    
    // Try to get URL from graph
    if (get_url_from_graph(g, node_id, url, sizeof(url)) == 0) {
        // Use default API endpoint or check for URL in node data
        // For now, use a simple default
        const char *default_url = "https://api.github.com/zen";
        
        // Check if there's a URL stored in a connected node
        // This is a simplified version - in full system, graph would pass URL
        strncpy(url, default_url, sizeof(url) - 1);
    }
    
    printf("[mc_api] Making GET request to: %s\n", url);
    
    // Use curl via system call (simple approach)
    // In production, use libcurl for better control
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), 
             "curl -s -m 10 --max-time 10 '%s' 2>/dev/null", url);
    
    FILE *fp = popen(cmd, "r");
    if (!fp) {
        fprintf(stderr, "[mc_api] Failed to execute curl\n");
        g->nodes[node_id].a = 0.0f;
        return;
    }
    
    // Read response
    char response[4096] = {0};
    size_t total_read = 0;
    char buffer[256];
    
    while (fgets(buffer, sizeof(buffer), fp) != NULL && total_read < sizeof(response) - 1) {
        size_t len = strlen(buffer);
        if (total_read + len < sizeof(response)) {
            memcpy(response + total_read, buffer, len);
            total_read += len;
        } else {
            break;
        }
    }
    
    int rc = pclose(fp);
    
    if (rc != 0) {
        fprintf(stderr, "[mc_api] Request failed (exit code %d)\n", rc);
        g->nodes[node_id].a = 0.0f;
        return;
    }
    
    printf("[mc_api] Response received (%zu bytes): %.100s%s\n", 
           total_read, response, total_read > 100 ? "..." : "");
    
    // Store response in graph
    store_response_in_graph(g, node_id, response, total_read);
    
    // Create success node
    uint64_t success_node = alloc_node(g);
    if (success_node != UINT64_MAX) {
        Node *sn = &g->nodes[success_node];
        sn->kind = NODE_KIND_META;
        sn->a = 1.0f;
        sn->value = 0x4150494F; // "APIO"
        add_edge(g, node_id, success_node, 1.0f, EDGE_FLAG_CONTROL);
    }
    
    strncpy(last_url, url, sizeof(last_url) - 1);
    last_request_tick = g->header->tick;
    
    // Deactivate node
    g->nodes[node_id].a = 0.0f;
    g->nodes[node_id].bias = -5.0f;
}

// MC function: Make HTTP POST request
void mc_api_post(Brain *g, uint64_t node_id) {
    static char last_url[512] = {0};
    static int last_request_tick = -1;
    
    // Only run if node is activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    // Prevent rapid-fire requests
    if (last_request_tick >= 0 && 
        (int)g->header->tick - last_request_tick < 10) {
        g->nodes[node_id].a = 0.0f;
        return;
    }
    
    char url[512] = {0};
    char data[1024] = {0};
    
    // Try to get URL and data from graph
    if (get_url_from_graph(g, node_id, url, sizeof(url)) == 0) {
        strncpy(url, "https://httpbin.org/post", sizeof(url) - 1);
    }
    
    // For now, use default test data
    // In full system, graph would pass data via connected nodes
    strncpy(data, "{\"test\":\"data\"}", sizeof(data) - 1);
    
    printf("[mc_api] Making POST request to: %s\n", url);
    printf("[mc_api] Data: %s\n", data);
    
    // Use curl for POST
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), 
             "curl -s -m 10 -X POST -H 'Content-Type: application/json' -d '%s' '%s' 2>/dev/null",
             data, url);
    
    FILE *fp = popen(cmd, "r");
    if (!fp) {
        fprintf(stderr, "[mc_api] Failed to execute curl\n");
        g->nodes[node_id].a = 0.0f;
        return;
    }
    
    // Read response
    char response[4096] = {0};
    size_t total_read = 0;
    char buffer[256];
    
    while (fgets(buffer, sizeof(buffer), fp) != NULL && total_read < sizeof(response) - 1) {
        size_t len = strlen(buffer);
        if (total_read + len < sizeof(response)) {
            memcpy(response + total_read, buffer, len);
            total_read += len;
        } else {
            break;
        }
    }
    
    int rc = pclose(fp);
    
    if (rc != 0) {
        fprintf(stderr, "[mc_api] Request failed (exit code %d)\n", rc);
        g->nodes[node_id].a = 0.0f;
        return;
    }
    
    printf("[mc_api] Response received (%zu bytes)\n", total_read);
    
    // Store response in graph
    store_response_in_graph(g, node_id, response, total_read);
    
    // Create success node
    uint64_t success_node = alloc_node(g);
    if (success_node != UINT64_MAX) {
        Node *sn = &g->nodes[success_node];
        sn->kind = NODE_KIND_META;
        sn->a = 1.0f;
        sn->value = 0x4150494F; // "APIO"
        add_edge(g, node_id, success_node, 1.0f, EDGE_FLAG_CONTROL);
    }
    
    strncpy(last_url, url, sizeof(last_url) - 1);
    last_request_tick = g->header->tick;
    
    // Deactivate node
    g->nodes[node_id].a = 0.0f;
    g->nodes[node_id].bias = -5.0f;
}

// MC function: Query LLM API (e.g., OpenAI, Anthropic, or local Ollama)
void mc_api_llm_query(Brain *g, uint64_t node_id) {
    static int last_request_tick = -1;
    
    // Only run if node is activated
    if (g->nodes[node_id].a < 0.5f) {
        return;
    }
    
    // Rate limiting
    if (last_request_tick >= 0 && 
        (int)g->header->tick - last_request_tick < 50) {
        g->nodes[node_id].a = 0.0f;
        return;
    }
    
    // Default prompt (in full system, graph would provide this)
    const char *prompt = "What is intelligence?";
    const char *api_url = "http://localhost:11434/api/generate"; // Ollama default
    
    // Check for Ollama first (local, free)
    // Fallback to other APIs if needed
    
    printf("[mc_api] Querying LLM with prompt: %s\n", prompt);
    
    // Build JSON payload for Ollama
    char json_payload[2048];
    snprintf(json_payload, sizeof(json_payload),
             "{\"model\":\"llama2\",\"prompt\":\"%s\",\"stream\":false}",
             prompt);
    
    // Make POST request to Ollama
    char cmd[3072];
    snprintf(cmd, sizeof(cmd),
             "curl -s -m 30 -X POST -H 'Content-Type: application/json' -d '%s' '%s' 2>/dev/null",
             json_payload, api_url);
    
    FILE *fp = popen(cmd, "r");
    if (!fp) {
        fprintf(stderr, "[mc_api] Failed to execute curl for LLM\n");
        g->nodes[node_id].a = 0.0f;
        return;
    }
    
    // Read response
    char response[8192] = {0};
    size_t total_read = 0;
    char buffer[512];
    
    while (fgets(buffer, sizeof(buffer), fp) != NULL && total_read < sizeof(response) - 1) {
        size_t len = strlen(buffer);
        if (total_read + len < sizeof(response)) {
            memcpy(response + total_read, buffer, len);
            total_read += len;
        } else {
            break;
        }
    }
    
    int rc = pclose(fp);
    
    if (rc != 0) {
        fprintf(stderr, "[mc_api] LLM request failed (exit code %d)\n", rc);
        fprintf(stderr, "[mc_api] Make sure Ollama is running: ollama serve\n");
        g->nodes[node_id].a = 0.0f;
        return;
    }
    
    printf("[mc_api] LLM response received (%zu bytes)\n", total_read);
    
    // Parse Ollama response (simple JSON extraction)
    // Ollama returns: {"response":"...","done":true}
    // Extract the "response" field
    char *response_start = strstr(response, "\"response\":\"");
    if (response_start) {
        response_start += 12; // Skip "response":"
        char *response_end = strchr(response_start, '"');
        if (response_end) {
            size_t response_len = response_end - response_start;
            if (response_len < sizeof(response)) {
                char extracted[4096];
                memcpy(extracted, response_start, response_len);
                extracted[response_len] = '\0';
                
                printf("[mc_api] LLM says: %.200s%s\n", 
                       extracted, response_len > 200 ? "..." : "");
                
                // Store in graph
                store_response_in_graph(g, node_id, extracted, response_len);
            }
        }
    } else {
        // Store raw response if parsing fails
        store_response_in_graph(g, node_id, response, total_read);
    }
    
    // Create success node
    uint64_t success_node = alloc_node(g);
    if (success_node != UINT64_MAX) {
        Node *sn = &g->nodes[success_node];
        sn->kind = NODE_KIND_META;
        sn->a = 1.0f;
        sn->value = 0x4C4C4D52; // "LLMR"
        add_edge(g, node_id, success_node, 1.0f, EDGE_FLAG_CONTROL);
    }
    
    last_request_tick = g->header->tick;
    
    // Deactivate node
    g->nodes[node_id].a = 0.0f;
    g->nodes[node_id].bias = -5.0f;
}

