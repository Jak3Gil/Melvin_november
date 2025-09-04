#include "optimized_node_system.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>

using namespace melvin;

void print_memory_usage(const OptimizedDynamicNodeSizer& sizer, const std::string& label) {
    size_t memory_bytes = sizer.get_memory_usage();
    double memory_mb = static_cast<double>(memory_bytes) / (1024 * 1024);
    
    std::cout << "📊 " << label << " Memory Usage: " 
              << std::fixed << std::setprecision(2) << memory_mb << " MB" << std::endl;
}

void print_statistics(const OptimizedDynamicNodeSizer& sizer, const std::string& label) {
    auto stats = sizer.get_statistics();
    
    std::cout << "📈 " << label << " Statistics:" << std::endl;
    std::cout << "   🔹 Tiny nodes: " << stats.tiny_nodes << std::endl;
    std::cout << "   🔹 Small nodes: " << stats.small_nodes << std::endl;
    std::cout << "   🔹 Medium nodes: " << stats.medium_nodes << std::endl;
    std::cout << "   🔹 Large nodes: " << stats.large_nodes << std::endl;
    std::cout << "   🔹 Extra large nodes: " << stats.extra_large_nodes << std::endl;
    std::cout << "   🔗 Total connections: " << stats.total_connections << std::endl;
    
    uint64_t total_nodes = stats.tiny_nodes + stats.small_nodes + 
                          stats.medium_nodes + stats.large_nodes + stats.extra_large_nodes;
    
    std::cout << "   📊 Total nodes: " << total_nodes << std::endl;
}

void benchmark_performance() {
    std::cout << "🚀 PERFORMANCE BENCHMARK" << std::endl;
    std::cout << "=" << std::string(50, '=') << std::endl;
    
    OptimizedDynamicNodeSizer sizer;
    
    // Test data
    std::string short_text = "AI machine learning neural networks";
    std::string medium_text = "Artificial intelligence is a branch of computer science that focuses on creating intelligent machines that can perform tasks that typically require human intelligence.";
    std::string long_text = "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. Computer vision is a field of artificial intelligence that trains computers to interpret and understand visual information from the world.";
    
    // Benchmark tiny nodes
    auto start = std::chrono::high_resolution_clock::now();
    auto tiny_nodes = sizer.create_dynamic_nodes(short_text, NodeSize::TINY);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "⏱️  Tiny nodes creation: " << duration.count() << " μs" << std::endl;
    print_statistics(sizer, "Tiny Nodes");
    print_memory_usage(sizer, "Tiny Nodes");
    
    // Benchmark small nodes
    start = std::chrono::high_resolution_clock::now();
    auto small_nodes = sizer.create_dynamic_nodes(medium_text, NodeSize::SMALL);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "⏱️  Small nodes creation: " << duration.count() << " μs" << std::endl;
    print_statistics(sizer, "Small Nodes");
    print_memory_usage(sizer, "Small Nodes");
    
    // Benchmark auto-sizing
    start = std::chrono::high_resolution_clock::now();
    auto auto_nodes = sizer.create_dynamic_nodes(long_text, NodeSize::MEDIUM);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "⏱️  Auto-sized nodes creation: " << duration.count() << " μs" << std::endl;
    print_statistics(sizer, "Auto-Sized Nodes");
    print_memory_usage(sizer, "Auto-Sized Nodes");
}

void test_memory_efficiency() {
    std::cout << "\n💾 MEMORY EFFICIENCY TEST" << std::endl;
    std::cout << "=" << std::string(50, '=') << std::endl;
    
    OptimizedDynamicNodeSizer sizer;
    
    // Create many nodes to test memory efficiency
    std::vector<std::string> test_texts = {
        "artificial intelligence",
        "machine learning", 
        "deep learning",
        "neural networks",
        "computer vision",
        "natural language processing",
        "reinforcement learning",
        "supervised learning",
        "unsupervised learning",
        "transfer learning"
    };
    
    size_t initial_memory = sizer.get_memory_usage();
    std::cout << "📊 Initial memory: " << initial_memory << " bytes" << std::endl;
    
    // Create nodes
    for (const auto& text : test_texts) {
        sizer.create_dynamic_nodes(text, NodeSize::TINY);
    }
    
    size_t final_memory = sizer.get_memory_usage();
    std::cout << "📊 Final memory: " << final_memory << " bytes" << std::endl;
    std::cout << "📊 Memory increase: " << (final_memory - initial_memory) << " bytes" << std::endl;
    
    auto stats = sizer.get_statistics();
    uint64_t total_nodes = stats.tiny_nodes + stats.small_nodes + 
                          stats.medium_nodes + stats.large_nodes + stats.extra_large_nodes;
    
    double bytes_per_node = static_cast<double>(final_memory - initial_memory) / total_nodes;
    std::cout << "📊 Average bytes per node: " << std::fixed << std::setprecision(1) << bytes_per_node << " bytes" << std::endl;
}

void test_node_sizes() {
    std::cout << "\n📏 NODE SIZE TEST" << std::endl;
    std::cout << "=" << std::string(50, '=') << std::endl;
    
    OptimizedDynamicNodeSizer sizer;
    
    std::string test_text = "Artificial intelligence is a complex field that combines computer science, mathematics, and cognitive science to create systems that can perform tasks requiring human intelligence.";
    
    // Test different size categories
    std::vector<std::pair<NodeSize, std::string>> size_tests = {
        {NodeSize::TINY, "Tiny"},
        {NodeSize::SMALL, "Small"},
        {NodeSize::MEDIUM, "Medium"},
        {NodeSize::LARGE, "Large"},
        {NodeSize::EXTRA_LARGE, "Extra Large"}
    };
    
    for (const auto& [size, label] : size_tests) {
        auto start = std::chrono::high_resolution_clock::now();
        auto nodes = sizer.create_dynamic_nodes(test_text, size);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "🔹 " << label << " nodes: " << nodes.size() << " nodes in " 
                  << duration.count() << " μs" << std::endl;
    }
}

void test_complexity_analysis() {
    std::cout << "\n🧠 COMPLEXITY ANALYSIS TEST" << std::endl;
    std::cout << "=" << std::string(50, '=') << std::endl;
    
    OptimizedDynamicNodeSizer sizer;
    
    std::vector<std::pair<std::string, std::string>> complexity_tests = {
        {"Simple", "AI ML"},
        {"Medium", "Artificial intelligence machine learning"},
        {"Complex", "Artificial intelligence encompasses machine learning, deep learning, neural networks, and various other computational approaches to mimic human cognitive functions."}
    };
    
    for (const auto& [complexity, text] : complexity_tests) {
        auto start = std::chrono::high_resolution_clock::now();
        auto nodes = sizer.create_dynamic_nodes(text, NodeSize::MEDIUM, 0.5f);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "🔹 " << complexity << " text: " << nodes.size() << " nodes in " 
                  << duration.count() << " μs" << std::endl;
    }
}

int main() {
    std::cout << "🧠 MELVIN OPTIMIZED NODE SYSTEM TEST" << std::endl;
    std::cout << "=" << std::string(60, '=') << std::endl;
    
    try {
        benchmark_performance();
        test_memory_efficiency();
        test_node_sizes();
        test_complexity_analysis();
        
        std::cout << "\n✅ All tests completed successfully!" << std::endl;
        std::cout << "\n🎉 Optimized C++ Node System Features:" << std::endl;
        std::cout << "   🔹 Byte-level memory management" << std::endl;
        std::cout << "   🔹 Content deduplication" << std::endl;
        std::cout << "   🔹 Efficient data structures" << std::endl;
        std::cout << "   🔹 SIMD optimizations" << std::endl;
        std::cout << "   🔹 Cache-friendly layouts" << std::endl;
        std::cout << "   🔹 Minimal memory overhead" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
