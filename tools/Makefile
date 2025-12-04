CC = gcc
CFLAGS = -O2 -Wall -Wextra -I.. -std=c11
LDFLAGS = -lm -lpthread

MELVIN_OBJ = ../melvin.o

all: create_exec_routing inspect_graph teach_hardware_operations create_port_patterns bootstrap_hardware_edges map_can_motors

create_exec_routing: create_exec_routing.c $(MELVIN_OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(MELVIN_OBJ) $(LDFLAGS)

inspect_graph: inspect_graph.c $(MELVIN_OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(MELVIN_OBJ) $(LDFLAGS)

teach_hardware_operations: teach_hardware_operations.c $(MELVIN_OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(MELVIN_OBJ) $(LDFLAGS)

create_port_patterns: create_port_patterns.c $(MELVIN_OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(MELVIN_OBJ) $(LDFLAGS)

bootstrap_hardware_edges: bootstrap_hardware_edges.c $(MELVIN_OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(MELVIN_OBJ) $(LDFLAGS)

map_can_motors: map_can_motors.c $(MELVIN_OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(MELVIN_OBJ) $(LDFLAGS)

clean:
	rm -f create_exec_routing inspect_graph teach_hardware_operations create_port_patterns bootstrap_hardware_edges map_can_motors *.o

.PHONY: all clean
