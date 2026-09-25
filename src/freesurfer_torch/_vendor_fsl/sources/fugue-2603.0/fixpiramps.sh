#!/bin/sh

# Mark Jenkinson, FMRIB Image Analysis Group
#
# Copyright (C) 2004 University of Oxford
#
# SHCOPYRIGHT

if [ $# -lt 2 ] ; then
  echo "Usage: $0 <input phase image> <output phase image>"
  exit 1;
fi

arg1=`$FSLDIR/bin/remove_ext $1`

dtype=`$FSLDIR/bin/fslval $arg1 datatype`;
if [ $dtype -eq 32 ] ; then
    phaseimt=${arg1}_realphase
    absim=${arg1}_realabs
    $FSLDIR/bin/fslcomplex -realpolar $arg1 $absim $phaseimt 
else
    phaseimt=${arg1}_phase;
    $FSLDIR/bin/fslmaths ${arg1} $phaseimt
fi

phaseout=${arg1}_phaseout

tt=0;
tmax=`fslval $arg1 dim4`;

while [ $tt -lt $tmax ] ; do

  phaseim=${phaseimt}_tmp
  $FSLDIR/bin/fslroi $phaseimt $phaseim $tt 1

  if [ $tt -eq 0 ] ; then
    xdim=`$FSLDIR/bin/fslval $phaseim pixdim1`;
    ydim=`$FSLDIR/bin/fslval $phaseim pixdim2`;
    zdim=`$FSLDIR/bin/fslval $phaseim pixdim3`;
    
    pixdim=`echo "scale=10; $xdim / 3.14159265 " | bc`;
    piydim=`echo "scale=10; $ydim / 3.14159265 " | bc`;
    pizdim=`echo "scale=10; $zdim / 3.14159265 " | bc`;
    
    echo "dimension / PI = $pixdim , $piydim , $pizdim"
    
    tmpfile=${phaseim}_$$
    
    echo "$pixdim 0 0 0" > $tmpfile
    echo "0 $piydim 0 0" >> $tmpfile
    echo "0 0 $pizdim 0" >> $tmpfile
    echo "0 0 0 1" >> $tmpfile
    
    echo "Creating xyzramp image"
    
    $FSLDIR/bin/convertwarp -m $tmpfile -o xyzramp -r $phaseim
    
    echo "Splitting into separate ramps"
    
    $FSLDIR/bin/fslroi xyzramp xramp 0 1
    $FSLDIR/bin/fslroi xyzramp yramp 1 1
    $FSLDIR/bin/fslroi xyzramp zramp 2 1
  fi

  echo "Adding pi ramps to original phase image"
  
  $FSLDIR/bin/fslmaths $phaseim -add xramp -add yramp -add zramp $2 -odt float
  
  echo "Wrapping phase back to +/- pi range"
  
  # Implements: ph_wrapped = ph - 2*pi*round(ph/(2*pi))
  #  where round(x) is implemented as int(x + 0.5) - (x<-0.5)
  $FSLDIR/bin/fslmaths $2 -div 6.2831853 -add 0.5 ${2}_tmp -odt float 
  $FSLDIR/bin/fslmaths ${2}_tmp ${2}_tmp -odt int
  $FSLDIR/bin/fslmaths ${2} -uthr -0.5 -abs -bin -mul -1 -add ${2}_tmp ${2}_tmp -odt float
  $FSLDIR/bin/fslmaths ${2}_tmp -mul -6.2831853 -add ${2} ${2} -odt float

  if [ $tt -ge 1 ] ; then
    $FSLDIR/bin/fslmerge -t $phaseout $phaseout ${2}
  else
    $FSLDIR/bin/fslmaths ${2} ${phaseout}
  fi

  tt=`echo $tt + 1 | bc`;
  # end while loop  
done

# Recombine phase with abs if complex arg1
if [ $dtype -eq 32 ] ; then
    $FSLDIR/bin/fslmaths $phaseout $phaseim
    $FSLDIR/bin/fslcomplex -complexpolar $absim $phaseim ${2}
else
    $FSLDIR/bin/fslmaths $phaseout ${2}
fi


echo "Cleaning up files"

/bin/rm -f xyzramp.* xramp.* yramp.* zramp.* $tmpfile ${2}_tmp.* ${phaseim}.* ${phaseout}.* ${phaseimt}.*
if [ $dtype -eq 32 ] ; then
    /bin/rm -f ${absim}.*
fi

