CXX := g++
CXXFLAGS := -std=c++11 -g
LIB_DIRS := 
LDFLAGS := 

ifeq ($(shell uname),Darwin)
	INC_DIRS :=
else
	INC_DIRS :=
endif

all:
	$(CXX) 842_compress.c 842_decompress.c test.c $(CXXFLAGS) $(INC_DIRS) $(LIB_DIRS) $(LDFLAGS) -o 842_test
	
clean:
	rm -f 842_test
	
