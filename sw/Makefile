CXX := g++
CXXFLAGS := -std=c++11 
LIB_DIRS := 
LDFLAGS := 

ifeq ($(shell uname),Darwin)
	INC_DIRS := -I/usr/local/Cellar/boost/1.64.0_1/include/
else
	INC_DIRS :=
endif

all:
	$(CXX) 842_compress.c test.c $(CXXFLAGS) $(INC_DIRS) $(LIB_DIRS) $(LDFLAGS) -o 842_compress
	
clean:
	rm -f 842_compress
	
