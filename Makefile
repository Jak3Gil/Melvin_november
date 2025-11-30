# Makefile for Melvin tests

CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2
LDFLAGS = -lm

# Test executables
TESTS = test_0_0_exec_smoke test_exec_basic test_0_7_multihop_math test_1_0_graph_add32 test_1_1_tool_selection test_master_8_capabilities

# Tools
TOOLS = melvin_pack_corpus

.PHONY: all clean tools test_0_0_exec_smoke test_exec_basic test_0_7_multihop_math test_1_0_graph_add32 test_1_1_tool_selection test_master_8_capabilities

all: $(TESTS) tools

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
melvin_pack_corpus: melvin_pack_corpus.c melvin.c melvin.h
	$(CC) $(CFLAGS) -o melvin_pack_corpus melvin_pack_corpus.c melvin.c $(LDFLAGS)

clean:
	rm -f $(TESTS) $(TOOLS) *.o *.m

