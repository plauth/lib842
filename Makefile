CC=gcc
CC_FLAGS=-Wall -fPIC


MODULES   := serial
OBJ_DIR := $(addprefix obj/,$(MODULES))
BIN_DIR := $(addprefix bin/,$(MODULES))

SRC_DIR_SERIAL := src/serial
OBJ_DIR_SERIAL := obj/serial

SRC_FILES_SERIAL := $(wildcard $(SRC_DIR_SERIAL)/*.c)
OBJ_FILES_SERIAL := $(patsubst src/serial/%.c,obj/serial/%.o,$(SRC_FILES_SERIAL))

.PHONY: all checkdirs clean
#.check-env:

all: checkdirs serial

$(OBJ_DIR_SERIAL)/%.o: $(SRC_DIR_SERIAL)/%.c
	$(CC) $(CC_FLAGS) -I$(SRC_DIR_SERIAL) -c $< -o $@

serial: checkdirs $(OBJ_FILES_SERIAL)
	$(CC) $(CC_FLAGS) -shared -Wl,-soname,lib842.so -Wl,--no-as-needed -lz -o bin/serial/lib842.so $(OBJ_FILES_SERIAL)

clean:
	rm -Rf obj
	rm -Rf bin
	rm -Rf test/test

checkdirs: $(OBJ_DIR) $(BIN_DIR)

test: serial
	$(CC) test/test.c -o test/test -I./include -L./bin/serial/ -l842
	LD_LIBRARY_PATH=$(shell pwd)/$(BIN_DIR) test/test

$(BIN_DIR) $(OBJ_DIR):
	@mkdir -p $@