# useful shortcuts and functions
# $@ means everything to the left of the ":"
# $^ means everything to the right of the ":"
# $< means first item to the right of the ":"
# $(patsubst pattern, replacement, text)

all: model.o
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags`

model.o: model.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS)

clean:; rm -rf $(TARGET) *.o

.PHONY: clean all
