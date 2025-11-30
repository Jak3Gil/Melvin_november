#!/bin/bash
# Deploy GPU acceleration plugin to Jetson
# Automatically adds GPU support to melvin.c and compiles

set -e

echo "=========================================="
echo "Deploying GPU Acceleration Plugin"
echo "=========================================="
echo ""

JETSON_IP="169.254.123.100"
JETSON_USER="melvin"
JETSON_PASS="123456"
JETSON_HOST="${JETSON_USER}@${JETSON_IP}"

echo "Step 1: Deploying GPU plugin source..."
cd /Users/jakegilbert/melvin_november/Melvin_november
# Deploy both .c and .cu versions (nvcc needs .cu for CUDA code)
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no plugins/mc_gpu.cu "$JETSON_HOST:/home/melvin/melvin_system/plugins/" 2>/dev/null || \
sshpass -p "$JETSON_PASS" scp -o StrictHostKeyChecking=no plugins/mc_gpu.c "$JETSON_HOST:/home/melvin/melvin_system/plugins/mc_gpu.cu"
echo "✓ Plugin source deployed"
echo ""

echo "Step 2: Compiling GPU plugin on Jetson..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    cd /home/melvin/melvin_system/plugins
    
    echo "Compiling with CUDA (.cu file)..."
    # nvcc needs -Xcompiler for host compiler flags like -fPIC, and .cu extension for CUDA code
    /usr/local/cuda/bin/nvcc -shared -Xcompiler -fPIC -O3 -arch=sm_87 -o mc_gpu.so mc_gpu.cu -lcudart -I.. 2>&1 | grep -E 'error|warning' | head -10 || echo "✓ Compilation successful"
    
    if [ -f mc_gpu.so ]; then
        echo "✓ GPU plugin compiled"
        ls -lh mc_gpu.so
        # Make sure it's readable
        chmod 755 mc_gpu.so
    else
        echo "✗ Compilation failed"
        exit 1
    fi
EOF
echo ""

echo "Step 3: Adding GPU registration to melvin.c..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    cd /home/melvin/melvin_system
    
    # Backup melvin.c
    cp melvin.c melvin.c.backup.before_gpu
    
    # Check if GPU registration already exists
    if grep -q "mc_gpu" melvin.c; then
        echo "GPU registration already exists"
    else
        # Find where display plugin is loaded and add GPU after it
        # Look for the display plugin loading section
        if grep -q "mc_display_graph = load_plugin_function" melvin.c; then
            # Add GPU loading after display plugin
            sed -i '/mc_display_init = load_plugin_function/a\
    // Load GPU plugin\
    MCFn mc_gpu_init = load_plugin_function("mc_gpu", "mc_gpu_init");\
    MCFn mc_gpu_propagate = load_plugin_function("mc_gpu", "mc_gpu_propagate");\
    MCFn mc_gpu_compute_error = load_plugin_function("mc_gpu", "mc_gpu_compute_error");\
    MCFn mc_gpu_update_edges = load_plugin_function("mc_gpu", "mc_gpu_update_edges");
' melvin.c
            
            # Add GPU registration after display registration
            sed -i '/if (mc_display_init) register_mc("display_init", mc_display_init);/a\
    if (mc_gpu_init) register_mc("gpu_init", mc_gpu_init);\
    if (mc_gpu_propagate) register_mc("gpu_propagate", mc_gpu_propagate);\
    if (mc_gpu_compute_error) register_mc("gpu_compute_error", mc_gpu_compute_error);\
    if (mc_gpu_update_edges) register_mc("gpu_update_edges", mc_gpu_update_edges);
' melvin.c
            
            # Create GPU initialization node
            sed -i '/Created display graph node/a\
    \
    // Create GPU initialization node\
    if (mc_gpu_init) {\
        uint64_t gpu_init_node = alloc_node(&g);\
        if (gpu_init_node != UINT64_MAX && gpu_init_node < g.header->node_cap) {\
            g.nodes[gpu_init_node].kind = NODE_KIND_CONTROL;\
            for (uint32_t i = 0; i < g_mc_count; i++) {\
                if (g_mc_table[i].name && strcmp(g_mc_table[i].name, "gpu_init") == 0) {\
                    g.nodes[gpu_init_node].mc_id = i;\
                    g.nodes[gpu_init_node].bias = 5.0f;\
                    g.nodes[gpu_init_node].a = 1.0f;\
                    printf("[main] Created GPU initialization node %llu\\n", (unsigned long long)gpu_init_node);\
                    break;\
                }\
            }\
        }\
    }
' melvin.c
            
            echo "✓ GPU registration added to melvin.c"
        else
            echo "✗ Could not find insertion point in melvin.c"
            exit 1
        fi
    fi
EOF
echo ""

echo "Step 4: Recompiling Melvin with GPU support..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" << 'EOF'
    cd /home/melvin/melvin_system
    
    echo "Stopping service..."
    sudo systemctl stop melvin.service 2>/dev/null || true
    
    echo "Compiling..."
    gcc -O2 -o melvin melvin.c -ldl -lm -lpthread -I. 2>&1 | grep -E 'error|warning' | head -5 || echo "✓ Compiled successfully"
    
    if [ -f melvin ]; then
        echo "✓ Melvin recompiled with GPU support"
        ls -lh melvin
    else
        echo "✗ Compilation failed"
        exit 1
    fi
EOF
echo ""

echo "Step 5: Starting Melvin with GPU support..."
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "sudo systemctl start melvin.service"
sleep 3
sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no "$JETSON_HOST" "sudo systemctl status melvin.service --no-pager -l | head -12"
echo ""

echo "=========================================="
echo "GPU Plugin Deployment Complete!"
echo "=========================================="
echo ""
echo "GPU acceleration is now available!"
echo "The GPU will be initialized automatically on startup."
echo ""
echo "To verify GPU is working, check logs:"
echo "  sshpass -p '$JETSON_PASS' ssh $JETSON_HOST 'tail -f /home/melvin/melvin_system/melvin.log | grep gpu'"
echo ""

