# Makefile for Melvin tests

CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2
LDFLAGS = -lm

# Test executables
TESTS = test_0_0_exec_smoke test_exec_basic test_0_7_multihop_math test_1_0_graph_add32 test_1_1_tool_selection test_master_8_capabilities

# Tools
TOOLS = melvin_pack_corpus melvin_seed_instincts melvin_seed_patterns melvin_seed_knowledge melvin_seed_arithmetic_exec melvin_feed_instincts melvin_build_learning_env

# Hardware runners (require ALSA and V4L2)
HARDWARE = melvin_hardware_runner melvin_run_continuous

.PHONY: all clean tools test_0_0_exec_smoke test_exec_basic test_0_7_multihop_math test_1_0_graph_add32 test_1_1_tool_selection test_master_8_capabilities

all: $(TESTS) tools $(HARDWARE)

tools: $(TOOLS)

test_exec_basic: test_exec_basic.c
	$(CC) $(CFLAGS) -o test_exec_basic test_exec_basic.c

test_0_7_multihop_math: test_0_7_multihop_math.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o test_0_7_multihop_math test_0_7_multihop_math.c

test_1_0_graph_add32: test_1_0_graph_add32.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o test_1_0_graph_add32 test_1_0_graph_add32.c

test_1_1_tool_selection: test_1_1_tool_selection.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o test_1_1_tool_selection test_1_1_tool_selection.c

test_0_0_exec_smoke: test_0_0_exec_smoke.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o test_0_0_exec_smoke test_0_0_exec_smoke.c

test_master_8_capabilities: test_master_8_capabilities.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o test_master_8_capabilities test_master_8_capabilities.c

# Corpus packing tool
melvin_pack_corpus: src/melvin_pack_corpus.c src/melvin.c src/melvin.h
	$(CC) $(CFLAGS) -o melvin_pack_corpus src/melvin_pack_corpus.c src/melvin.c $(LDFLAGS) -pthread

# Instinct seeding tool
melvin_seed_instincts: src/melvin_seed_instincts.c src/melvin.c src/melvin.h
	$(CC) $(CFLAGS) -o melvin_seed_instincts src/melvin_seed_instincts.c src/melvin.c $(LDFLAGS) -pthread

# Pattern seeding tool (data-driven)
melvin_seed_patterns: src/melvin_seed_patterns.c src/melvin_load_patterns.c src/melvin.c src/melvin.h
	$(CC) $(CFLAGS) -o melvin_seed_patterns src/melvin_seed_patterns.c src/melvin_load_patterns.c src/melvin.c $(LDFLAGS) -pthread

# Knowledge seeding tool (math, wiki, etc.)
melvin_seed_knowledge: src/melvin_seed_knowledge.c src/melvin.c src/melvin.h
	$(CC) $(CFLAGS) -o melvin_seed_knowledge src/melvin_seed_knowledge.c src/melvin.c $(LDFLAGS) -pthread

# Arithmetic EXEC node seeding tool
melvin_seed_arithmetic_exec: src/melvin_seed_arithmetic_exec.c src/melvin.c src/melvin.h
	$(CC) $(CFLAGS) -o melvin_seed_arithmetic_exec src/melvin_seed_arithmetic_exec.c src/melvin.c $(LDFLAGS) -pthread

# Machine code feeding tool
melvin_feed_instincts: src/melvin_feed_instincts.c src/melvin.c src/melvin.h
	$(CC) $(CFLAGS) -o melvin_feed_instincts src/melvin_feed_instincts.c src/melvin.c $(LDFLAGS) -pthread

# Hardware runner (requires ALSA and V4L2 on Linux)
melvin_hardware_runner: src/melvin_hardware_runner.c src/melvin_hardware_audio.c src/melvin_hardware_video.c src/melvin.c src/host_syscalls.c src/melvin_tools.c src/melvin_tool_layer.c src/melvin.h src/melvin_hardware.h
	$(CC) $(CFLAGS) -o melvin_hardware_runner \
		src/melvin_hardware_runner.c \
		src/melvin_hardware_audio.c \
		src/melvin_hardware_video.c \
		src/melvin.c \
		src/host_syscalls.c \
		src/melvin_tools.c \
		src/melvin_tool_layer.c \
		$(LDFLAGS) -pthread \
		$$(pkg-config --cflags --libs alsa 2>/dev/null || echo "-lasound") \
		$$(pkg-config --cflags --libs libv4l2 2>/dev/null || echo "") \
		|| $(CC) $(CFLAGS) -o melvin_hardware_runner \
			src/melvin_hardware_runner.c \
			src/melvin_hardware_audio.c \
			src/melvin_hardware_video.c \
			src/melvin.c \
			src/host_syscalls.c \
			src/melvin_tools.c \
			src/melvin_tool_layer.c \
			$(LDFLAGS) -pthread -lasound

# Continuous runner (no hardware dependencies)
melvin_run_continuous: src/melvin_run_continuous.c src/melvin.c src/host_syscalls.c src/melvin_tools.c src/melvin.h
	$(CC) $(CFLAGS) -o melvin_run_continuous \
		src/melvin_run_continuous.c \
		src/melvin.c \
		src/host_syscalls.c \
		src/melvin_tools.c \
		$(LDFLAGS) -pthread

# Test tool syscalls
test_tools_syscall: src/test_tools_syscall.c src/melvin_tools.c src/melvin.h
	$(CC) $(CFLAGS) -o test_tools_syscall \
		src/test_tools_syscall.c \
		src/melvin_tools.c \
		$(LDFLAGS)

# Comprehensive capability test
test_all_capabilities: src/test_all_capabilities.c src/melvin.c src/melvin_tools.c src/host_syscalls.c src/melvin.h
	$(CC) $(CFLAGS) -o test_all_capabilities \
		src/test_all_capabilities.c \
		src/melvin.c \
		src/melvin_tools.c \
		src/host_syscalls.c \
		$(LDFLAGS) -pthread

# Compile instinct functions to object file
instinct_functions.o: src/instinct_functions.c
	$(CC) $(CFLAGS) -c -o instinct_functions.o src/instinct_functions.c

# Build learning environment tool
melvin_build_learning_env: src/melvin_build_learning_env.c
	$(CC) $(CFLAGS) -o melvin_build_learning_env src/melvin_build_learning_env.c $(LDFLAGS)

clean:
	rm -f $(TESTS) $(TOOLS) *.o *.m

