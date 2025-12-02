CC = gcc
CFLAGS = -O2 -Wall -Wextra -I.. -std=c11
LDFLAGS = -lm

MELVIN_OBJ = ../src/melvin.o

all: create_exec_routing inspect_graph

create_exec_routing: create_exec_routing.c $(MELVIN_OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(MELVIN_OBJ) $(LDFLAGS)

inspect_graph: inspect_graph.c $(MELVIN_OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(MELVIN_OBJ) $(LDFLAGS)

clean:
	rm -f create_exec_routing inspect_graph *.o

.PHONY: all clean
