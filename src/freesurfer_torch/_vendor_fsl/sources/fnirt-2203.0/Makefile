include ${FSLCONFDIR}/default.mk

PROJNAME=fnirt

FNIRT_OBJS=fnirt_costfunctions.o intensity_mappers.o matching_points.o fnirtfns.o fnirt.o
INTMAP_OBJS=fnirt_costfunctions.o fnirtfns.o intensity_mappers.o matching_points.o mapintensities.o
INVWARP_OBJS=invwarp.o
MAKE_FNIRT_FIELD_OBJS=make_fnirt_field.o
LIBS=-lfsl-warpfns -lfsl-meshclass -lfsl-basisfield -lfsl-newimage -lfsl-miscmaths -lfsl-NewNifti -lfsl-cprob -lfsl-znz -lfsl-utils

XFILES=fnirt invwarp

all: ${XFILES} config

config:
	@if [ ! -d ${DESTDIR}/etc ] ; then ${MKDIR} ${DESTDIR}/etc ; ${CHMOD} g+w ${DESTDIR}/etc ; fi
	@if [ ! -d ${DESTDIR}/etc/flirtsch ] ; then ${MKDIR} ${DESTDIR}/etc/flirtsch ; ${CHMOD} g+w ${DESTDIR}/etc/flirtsch ; fi
	${CP} -rf fnirtcnf/* ${DESTDIR}/etc/flirtsch/.

fnirt: ${FNIRT_OBJS}
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

invwarp: ${INVWARP_OBJS}
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

mapintensities: ${INTMAP_OBJS}
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}

make_fnirt_field: ${MAKE_FNIRT_FIELD_OBJS}
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}
