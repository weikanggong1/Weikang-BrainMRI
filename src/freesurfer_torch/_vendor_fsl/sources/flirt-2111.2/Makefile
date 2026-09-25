include ${FSLCONFDIR}/default.mk

PROJNAME = flirt

LIBS = -lfsl-warpfns -lfsl-basisfield -lfsl-meshclass -lfsl-newimage \
       -lfsl-miscmaths -lfsl-NewNifti -lfsl-cprob -lfsl-znz -lfsl-utils

RUNTCLS = Flirt InvertXFM ApplyXFM ConcatXFM Nudge
XFILES = flirt convert_xfm avscale rmsdiff std2imgcoord img2stdcoord \
	img2imgcoord applyxfm4D pointflirt makerot midtrans
SCRIPTS = extracttxt pairreg standard_space_roi flirt_average epi_reg aff2rigid

all: ${XFILES}

custominstall:
	@if [ ! -d ${DESTDIR}/etc ] ; then ${MKDIR} ${DESTDIR}/etc ; ${CHMOD} g+w ${DESTDIR}/etc ; fi
	@if [ ! -d ${DESTDIR}/etc/flirtsch ] ; then ${MKDIR} ${DESTDIR}/etc/flirtsch ; ${CHMOD} g+w ${DESTDIR}/etc/flirtsch ; fi
	${CP} -rf flirtsch/* ${DESTDIR}/etc/flirtsch/.

flirt: globaloptions.o flirt.o
	$(CXX) ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

%: %.cc
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}
