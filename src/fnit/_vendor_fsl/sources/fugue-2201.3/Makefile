include ${FSLCONFDIR}/default.mk

PROJNAME = fugue

LIBS=-lfsl-warpfns -lfsl-basisfield -lfsl-meshclass -lfsl-newimage -lfsl-miscmaths -lfsl-NewNifti -lfsl-cprob -lfsl-utils -lfsl-znz

RUNTCLS = Fsl_prepare_fieldmap
XFILES=prelude fugue applywarp convertwarp sigloss fnirtfileutils
SCRIPTS=fsl_prepare_fieldmap extractfidparams dof2warp
FSCRIPTS=fix_OCMR_fieldmaps
TESTXFILES=interpwarp

all: ${XFILES}

prelude: prelude.o unwarpfns.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

fugue: fugue.o unwarpfns.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

applywarp: applywarp.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

convertwarp: convertwarp.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

interpwarp: interpwarp.o unwarpfns.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

sigloss: sigloss.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

fnirtfileutils: fnirtfileutils.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}
