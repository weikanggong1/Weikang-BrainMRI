include ${FSLCONFDIR}/default.mk

PROJNAME = fast4
XFILES   = fast
RUNTCLS  = Fast
LIBS     = -lfsl-newimage -lfsl-miscmaths -lfsl-NewNifti \
           -lfsl-cprob -lfsl-utils -lfsl-znz

all: ${XFILES}

fast: fast_two.o mriseg_two.o multi_mriseg_two.o
	$(CXX) ${CXXFLAGS} -o $@ $^ ${LDFLAGS}
