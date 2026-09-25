#!/bin/sh

#   scheduleheader.sh
#
#   Mark Jenkinson, FMRIB Image Analysis Group
#
#   Copyright (C) 1999-2004 University of Oxford
#
#   SHCOPYRIGHT

if [ $# -lt 1 ] ; then
  echo "Usage: $0 rcfile"
  exit -1
fi

cat $1 | sed 's/^/  comms.push_back("/' | sed 's/$/");/'

