CC = gcc
CFLAGS = -std=c11 -Wall -Wextra -O2
TARGET = melvin_demo
TEST_TARGET = melvin_tests
PROBE_TARGET = melvin_probe
LEARN_TARGET = melvin_learn_cli
DSL_TARGET = melvin_dsl
STATS_TARGET = graph_stats
QUERY_TARGET = query_graph
DESCRIBE_TARGET = melvin_describe
OUTPUT_TEST_TARGET = test_output_emission
PATTERN_TEST_TARGET = test_pattern_on_pattern
GRAPH_LEARN_TEST_TARGET = test_graph_driven_learning
MAINTENANCE_TEST_TARGET = test_graph_self_maintenance
SRCDIR = src
SOURCES = $(SRCDIR)/melvin.c $(SRCDIR)/ $(SRCDIR)/main.c
TEST_SOURCES = $(SRCDIR)/melvin.c $(SRCDIR)/ $(SRCDIR)/tests.c
PROBE_SOURCES = $(SRCDIR)/melvin.c $(SRCDIR)/graph_probe.c
LEARN_SOURCES = $(SRCDIR)/melvin.c $(SRCDIR)/ $(SRCDIR)/melvin_learn_cli.c
DSL_SOURCES = $(SRCDIR)/melvin.c $(SRCDIR)/ $(SRCDIR)/melvin_dsl.c
STATS_SOURCES = $(SRCDIR)/melvin.c teacher/graph_stats.c
QUERY_SOURCES = $(SRCDIR)/melvin.c teacher/query_graph.c
DESCRIBE_SOURCES = $(SRCDIR)/melvin.c $(SRCDIR)/melvin_describe.c
OUTPUT_TEST_SOURCES = $(SRCDIR)/melvin.c $(SRCDIR)/ test_output_emission.c
PATTERN_TEST_SOURCES = $(SRCDIR)/melvin.c $(SRCDIR)/ test_pattern_on_pattern.c
GRAPH_LEARN_TEST_SOURCES = $(SRCDIR)/melvin.c $(SRCDIR)/ test_graph_driven_learning.c
MAINTENANCE_TEST_SOURCES = $(SRCDIR)/melvin.c $(SRCDIR)/ test_graph_self_maintenance.c
OBJECTS = $(SOURCES:.c=.o)
TEST_OBJECTS = $(SRCDIR)/melvin.o $(SRCDIR)/tests.o
PROBE_OBJECTS = $(SRCDIR)/melvin.o $(SRCDIR)/graph_probe.o
LEARN_OBJECTS = $(SRCDIR)/melvin.o $(SRCDIR)/ $(SRCDIR)/melvin_learn_cli.o
DSL_OBJECTS = $(SRCDIR)/melvin.o $(SRCDIR)/ $(SRCDIR)/melvin_dsl.o
STATS_OBJECTS = $(SRCDIR)/melvin.o teacher/graph_stats.o
QUERY_OBJECTS = $(SRCDIR)/melvin.o teacher/query_graph.o
DESCRIBE_OBJECTS = $(SRCDIR)/melvin.o $(SRCDIR)/melvin_describe.o
OUTPUT_TEST_OBJECTS = $(SRCDIR)/melvin.o $(SRCDIR)/ test_output_emission.o
PATTERN_TEST_OBJECTS = $(SRCDIR)/melvin.o $(SRCDIR)/ test_pattern_on_pattern.o
GRAPH_LEARN_TEST_OBJECTS = $(SRCDIR)/melvin.o $(SRCDIR)/ test_graph_driven_learning.o
MAINTENANCE_TEST_OBJECTS = $(SRCDIR)/melvin.o $(SRCDIR)/ test_graph_self_maintenance.o

.PHONY: all clean test probe learn dsl stats query describe test_output test_pattern test_graph_learn test_maintenance

all: $(TARGET)

test: $(TEST_TARGET)

probe: $(PROBE_TARGET)

learn: $(LEARN_TARGET)

dsl: $(DSL_TARGET)

stats: $(STATS_TARGET)

query: $(QUERY_TARGET)

describe: $(DESCRIBE_TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS)

$(TEST_TARGET): $(TEST_OBJECTS)
	$(CC) $(CFLAGS) -o $(TEST_TARGET) $(TEST_OBJECTS)

$(PROBE_TARGET): $(PROBE_OBJECTS)
	$(CC) $(CFLAGS) -o $(PROBE_TARGET) $(PROBE_OBJECTS)

$(LEARN_TARGET): $(LEARN_OBJECTS)
	$(CC) $(CFLAGS) -o $(LEARN_TARGET) $(LEARN_OBJECTS)

$(DSL_TARGET): $(DSL_OBJECTS)
	$(CC) $(CFLAGS) -o $(DSL_TARGET) $(DSL_OBJECTS)

$(STATS_TARGET): $(STATS_OBJECTS)
	$(CC) $(CFLAGS) -o $(STATS_TARGET) $(STATS_OBJECTS)

$(QUERY_TARGET): $(QUERY_OBJECTS)
	$(CC) $(CFLAGS) -o $(QUERY_TARGET) $(QUERY_OBJECTS)

$(DESCRIBE_TARGET): $(DESCRIBE_OBJECTS)
	$(CC) $(CFLAGS) -o $(DESCRIBE_TARGET) $(DESCRIBE_OBJECTS)

$(OUTPUT_TEST_TARGET): $(OUTPUT_TEST_OBJECTS)
	$(CC) $(CFLAGS) -o $(OUTPUT_TEST_TARGET) $(OUTPUT_TEST_OBJECTS)

test_output: $(OUTPUT_TEST_TARGET)

$(PATTERN_TEST_TARGET): $(PATTERN_TEST_OBJECTS)
	$(CC) $(CFLAGS) -o $(PATTERN_TEST_TARGET) $(PATTERN_TEST_OBJECTS)

test_pattern: $(PATTERN_TEST_TARGET)

$(GRAPH_LEARN_TEST_TARGET): $(GRAPH_LEARN_TEST_OBJECTS)
	$(CC) $(CFLAGS) -o $(GRAPH_LEARN_TEST_TARGET) $(GRAPH_LEARN_TEST_OBJECTS)

test_graph_learn: $(GRAPH_LEARN_TEST_TARGET)

$(MAINTENANCE_TEST_TARGET): $(MAINTENANCE_TEST_OBJECTS)
	$(CC) $(CFLAGS) -o $(MAINTENANCE_TEST_TARGET) $(MAINTENANCE_TEST_OBJECTS)

test_maintenance: $(MAINTENANCE_TEST_TARGET)

$(SRCDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

teacher/%.o: teacher/%.c
	$(CC) $(CFLAGS) -c $< -o $@

test_output_emission.o: test_output_emission.c
	$(CC) $(CFLAGS) -I$(SRCDIR) -c $< -o $@

test_pattern_on_pattern.o: test_pattern_on_pattern.c
	$(CC) $(CFLAGS) -I$(SRCDIR) -c $< -o $@

test_graph_driven_learning.o: test_graph_driven_learning.c
	$(CC) $(CFLAGS) -I$(SRCDIR) -c $< -o $@

test_graph_self_maintenance.o: test_graph_self_maintenance.c
	$(CC) $(CFLAGS) -I$(SRCDIR) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TEST_OBJECTS) $(PROBE_OBJECTS) $(LEARN_OBJECTS) $(DSL_OBJECTS) $(STATS_OBJECTS) $(QUERY_OBJECTS) $(DESCRIBE_OBJECTS) $(OUTPUT_TEST_OBJECTS) $(PATTERN_TEST_OBJECTS) $(GRAPH_LEARN_TEST_OBJECTS) $(MAINTENANCE_TEST_OBJECTS) $(TARGET) $(TEST_TARGET) $(PROBE_TARGET) $(LEARN_TARGET) $(DSL_TARGET) $(STATS_TARGET) $(QUERY_TARGET) $(DESCRIBE_TARGET) $(OUTPUT_TEST_TARGET) $(PATTERN_TEST_TARGET) $(GRAPH_LEARN_TEST_TARGET) $(MAINTENANCE_TEST_TARGET)

