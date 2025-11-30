// Add this to melvin.c after line 667 (after loading other plugins)
// This enables display functionality

    // Load display plugin
    MCFn mc_display_graph = load_plugin_function("mc_display", "mc_display_graph");
    MCFn mc_display_init = load_plugin_function("mc_display", "mc_display_init");
    
    // Register display functions
    if (mc_display_graph) register_mc("display_graph", mc_display_graph);
    if (mc_display_init) register_mc("display_init", mc_display_init);
    
    // Create and activate display initialization node on startup
    if (mc_display_init) {
        uint64_t display_init_node = alloc_node(&g);
        if (display_init_node != UINT64_MAX && display_init_node < g.header->node_cap) {
            g.nodes[display_init_node].kind = NODE_KIND_CONTROL;
            // Find the MC ID for display_init
            for (uint32_t i = 0; i < g_mc_count; i++) {
                if (g_mc_table[i].name && strcmp(g_mc_table[i].name, "display_init") == 0) {
                    g.nodes[display_init_node].mc_id = i;
                    g.nodes[display_init_node].bias = 5.0f; // Activate on startup
                    g.nodes[display_init_node].a = 1.0f;
                    printf("[main] Created display initialization node %llu\n", (unsigned long long)display_init_node);
                    break;
                }
            }
        }
        
        // Create display rendering node (will be activated by graph)
        if (mc_display_graph) {
            uint64_t display_node = alloc_node(&g);
            if (display_node != UINT64_MAX && display_node < g.header->node_cap) {
                g.nodes[display_node].kind = NODE_KIND_CONTROL;
                for (uint32_t i = 0; i < g_mc_count; i++) {
                    if (g_mc_table[i].name && strcmp(g_mc_table[i].name, "display_graph") == 0) {
                        g.nodes[display_node].mc_id = i;
                        g.nodes[display_node].bias = 3.0f; // Moderate bias to activate when needed
                        g.nodes[display_node].a = 0.0f; // Start inactive, will be activated
                        printf("[main] Created display graph node %llu\n", (unsigned long long)display_node);
                        break;
                    }
                }
            }
        }
    }


