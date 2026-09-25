#

# FLIRT - FMRIB's Linear Image Registration Tool
#
# Mark Jenkinson and Stephen Smith, FMRIB Image Analysis Group
#
# Copyright (C) 1999-2006 University of Oxford
#
# TCLCOPYRIGHT


proc flirt:proc { regmode refname testname testname2 nstats statslist output dof doftwo bins searchrxmin searchrxmax searchrymin searchrymax searchrzmin searchrzmax disablesearch_yn cost interp sincwidth sincwindow refweight inweight inweight2 popups } {

global PXHOME FSLDIR USER HOME

# setup options

set flirtoptions "-bins $bins -cost $cost"
set dofoptions ""
set doftwooptions ""

if { $dof == "2D" } {
    set dofoptions "-2D -dof 12"
} else {
  if { $dof == "TRANS" } {
      set dofoptions "-dof 6 -schedule ${FSLDIR}/etc/flirtsch/sch3Dtrans_3dof"
  } else {
    set dofoptions "-dof $dof"
  }
}

if { $doftwo == "2D" } {
    set doftwooptions "-2D -dof 12"
} else {
  if { $doftwo == "TRANS" } {
      set doftwooptions "-dof 6 -schedule ${FSLDIR}/etc/flirtsch/sch3Dtrans_3dof"
  } else {
    set doftwooptions "-dof $doftwo"
  }
}


if { $disablesearch_yn } {
    set flirtoptions "$flirtoptions -searchrx 0 0 -searchry 0 0 -searchrz 0 0"
} else {
    set flirtoptions "$flirtoptions -searchrx $searchrxmin $searchrxmax -searchry $searchrymin $searchrymax -searchrz $searchrzmin $searchrzmax"
}

set flirtweights1 ""
set flirtweights2 ""
if { $refweight != "" } {
    set flirtweights1 "$flirtweights1 -refweight $refweight"
}
if { $inweight != "" } {
    set flirtweights1 "$flirtweights1 -inweight $inweight"
    set flirtweights2 "$flirtweights2 -refweight $inweight"
}
if { $inweight2 != "" } {
    set flirtweights2 "$flirtweights2 -inweight $inweight2"
}

set flirtinterp "-interp $interp"
if { $interp == "sinc" } {
    set flirtinterp "$flirtinterp -sincwidth $sincwidth -sincwindow $sincwindow"
}

set outroot [ remove_ext $output ]

# tell fsl:exec to pass to batch system if setup, and set job durations to 10 minutes (potentially used by batch scheduler)
if { $regmode == 1 } {
    fsl:exec "${FSLDIR}/bin/flirt -in $testname -ref $refname -out $output -omat ${outroot}.mat $flirtoptions $dofoptions $flirtweights1 $flirtinterp" -t 10
} else {
    fsl:exec "${FSLDIR}/bin/flirt -in $testname -ref $refname -omat ${outroot}1.mat $flirtoptions $dofoptions $flirtweights1" -t 10
    fsl:exec "${FSLDIR}/bin/flirt -in $testname2 -ref $testname -omat ${outroot}2.mat $flirtoptions $doftwooptions $flirtweights2" -t 10
    fsl:exec "${FSLDIR}/bin/convert_xfm -concat ${outroot}1.mat -omat ${outroot}.mat ${outroot}2.mat"
    fsl:exec "${FSLDIR}/bin/flirt -in $testname2 -ref $refname -out $output -applyxfm -init ${outroot}.mat $flirtinterp"
}

for { set i 1 } { $i <= $nstats } { incr i 1 } {
    set statsname [ lindex $statslist [ expr $i - 1 ] ]
    fsl:exec "${FSLDIR}/bin/flirt -in $statsname -ref $refname -out [ file rootname ${output} ]_shadowreg_[ file tail [ file rootname $statsname ] ] -applyxfm -init ${outroot}.mat $flirtinterp"
}

puts Finished

set returnval 0

    return $returnval
}

