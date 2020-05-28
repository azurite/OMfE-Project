# useful shortcuts and functions
# $@ means everything to the left of the ":"
# $^ means everything to the right of the ":"
# $< means first item to the right of the ":"
# $(patsubst pattern, replacement, text)

BINARIES = main
KORALICXX=$(shell python3 -m korali.cxx --compiler)
KORALICFLAGS=`python3 -m korali.cxx --cflags` -O3 -march=native
KORALILIBS=`python3 -m korali.cxx --libs`

SDIR = src

# fixes linker error during the execution of make
# KORALILIBS := $(KORALILIBS) -lpython3

ifdef archlinux
KORALILIBS := $(KORALILIBS) -lpython3
endif

.SECONDARY:
.PHONY: all
all: $(BINARIES)

$(BINARIES) : % : %.o $(SDIR)/model/model.o
	$(KORALICXX) -o $@.out $^ $(KORALILIBS)

$(SDIR)/model/%.o:
	$(MAKE) -C $(SDIR)/model/ all

%.o: $(SDIR)/%.cpp
	$(KORALICXX) -c $(KORALICFLAGS) $<

.PHONY: clean
clean:
	$(MAKE) -C $(SDIR)/model/ clean
	$(RM) $(BINARIES) *.out *.o *.ti *.optrpt *.txt
	rm -rf _korali_result

plot:
	python -m korali.plotter
