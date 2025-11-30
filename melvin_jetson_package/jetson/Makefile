# Makefile for Melvin on Jetson

CC = gcc
CFLAGS = -std=c11 -Wall -I. -O2
LIBS = -lm -pthread -lasound

SRC_DIR = src
SRCS = $(SRC_DIR)/melvin.c \
       $(SRC_DIR)/host_syscalls.c \
       $(SRC_DIR)/melvin_tools.c

HARDWARE_SRCS = $(SRCS) \
                $(SRC_DIR)/melvin_hardware_audio.c \
                $(SRC_DIR)/melvin_hardware_video.c \
                $(SRC_DIR)/melvin_hardware_runner.c

CONTINUOUS_SRCS = $(SRCS) \
                  $(SRC_DIR)/melvin_run_continuous.c

all: melvin_hardware_runner melvin_run_continuous

melvin_hardware_runner: $(HARDWARE_SRCS)
	$(CC) $(CFLAGS) -o $@ $(HARDWARE_SRCS) $(LIBS)

melvin_run_continuous: $(CONTINUOUS_SRCS)
	$(CC) $(CFLAGS) -o $@ $(CONTINUOUS_SRCS) $(LIBS)

clean:
	rm -f melvin_hardware_runner melvin_run_continuous

install: all
	mkdir -p /usr/local/bin
	cp melvin_hardware_runner /usr/local/bin/
	cp melvin_run_continuous /usr/local/bin/

.PHONY: all clean install
