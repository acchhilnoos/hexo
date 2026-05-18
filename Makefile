CC := gcc
TARGET := main

SRCS := board.c main.c memory.c network.c tensor.c tree.c
OBJS := $(SRCS:.c=.o)
DEPS := $(SRCS:.c=.d)

CFLAGS := -std=c11 -Wall -Wextra -Wpedantic -g -O3

all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $@..."
	@$(CC) $(CFLAGS) -o $@ $^
	@echo "Build complete: $@"

%.o: %.c
	@echo "Compiling $<..."
	@$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(OBJS) $(DEPS) $(TARGET)
	@rm -rf *.dSYM
	@echo "Clean complete"

.PHONY: all clean run
