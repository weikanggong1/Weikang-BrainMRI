include ${FSLCONFDIR}/default.mk

PROJNAME = test_splinefield
XFILES   = test_splinefield
LIBS     = -lfsl-basisfield -lfsl-newimage -lfsl-miscmaths \
           -lfsl-NewNifti -lfsl-znz

# The test program can be run against
# an in-source checkout, or against
# an installed version of the
# libfsl-basisfield.so library.
#
# If the former, the basisfield library
# must have been compiled before the
# test can be compiled. We also assume
# that the source checkout is in a
# directory called "basisfield".
SYSTYPE := $(shell uname -s)
ifeq ($(SYSTYPE), Linux)
RPATH := -Wl,-rpath,'$$ORIGIN/..'
endif
ifeq ($(SYSTYPE), Darwin)
RPATH := -Wl,-rpath,'@executable_path/..'
endif

all: ${XFILES}

# We add -I../.., -L.., -Wl,-rpath so that
# in-source builds take precedence over
# $FSLDEVDIR/$FSLDIR
test_splinefield: test_splinefield.cpp
	${CXX} -I../.. ${CXXFLAGS} -o $@ $^ -L.. ${RPATH} ${LDFLAGS}
