# A Makefile for newimage unit tests.
include ${FSLCONFDIR}/default.mk

PROJNAME   = test-newimage
TESTXFILES = test-newimage

LIBS = -lfsl-newimage -lfsl-miscmaths -lfsl-NewNifti \
       -lfsl-cprob -lfsl-utils -lfsl-znz -lboost_unit_test_framework

# The test program can be run against
# an in-source checkout, or against
# an installed version of the
# libfsl-newimage.so library.
#
# If the former, the newimage library
# must have been compiled before the
# test can be compiled.

# The test program uses the Boost unit
# testing framework, which needs librt
# on linux.
SYSTYPE := $(shell uname -s)
ifeq ($(SYSTYPE), Linux)
LIBS  += -lrt
RPATH := -Wl,-rpath,'$$ORIGIN/..'
endif
ifeq ($(SYSTYPE), Darwin)
RPATH := -Wl,-rpath,'@executable_path/..'
endif

all: ${TESTXFILES}

OBJS := $(wildcard test_*.cc)
OBJS := $(OBJS:%.cc=%.o)

# We add -I.., -L.., -Wl,-rpath so that
# in-source builds take precedence over
# $FSLDEVDIR/$FSLDIR
%.o: %.cc
	$(CXX) -I.. ${CXXFLAGS} -c -o $@ $<

test-newimage: ${OBJS}
	$(CXX) -o $@ $^ -L.. ${RPATH} ${LDFLAGS}
