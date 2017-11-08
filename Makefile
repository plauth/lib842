CC=g++
CC_FLAGS=-Wall -fPIC


MODULES   := serial
SRC_DIR_SERIAL := src/serial
OBJ_DIR := $(addprefix obj/,$(MODULES))
BIN_DIR := $(addprefix bin/,$(MODULES))

SRC_FILES_SERIAL := $(wildcard $(SRC_DIR_SERIAL)/*.c)
OBJ_FILES_SERIAL := $(patsubst src/serial/%.c,obj/serial/%.o,$(SRC_FILES_SERIAL))

.PHONY: all checkdirs clean
#.check-env:

all: checkdirs serial


$(OBJ_FILES_SERIAL): $(SRC_FILES_SERIAL)
	$(CC) $(CC_FLAGS) -c -o $@ $<

serial: $(OBJ_FILES_SERIAL)
	$(CC) $(CC_FLAGS) -shared -Wl,-soname,lib842.so.1 -o bin/serial/lib842.so.1 $(OBJ_FILES_SERIAL)

clean:
	rm -Rf obj
	rm -Rf bin

checkdirs: $(OBJ_DIR) $(BIN_DIR)

$(OBJ_DIR):
	@mkdir -p $@

$(BIN_DIR):
	@mkdir -p $@
