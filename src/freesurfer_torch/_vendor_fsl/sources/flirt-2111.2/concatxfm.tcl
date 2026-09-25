#

# Concatxfm - the GUI for convert_xfm
#
# Mark Jenkinson, Stephen Smith and Matthew Webster, FMRIB Image Analysis Group
#
# Copyright (C) 2001 University of Oxford
#
# TCLCOPYRIGHT

#{{{ setup

source [ file dirname [ info script ] ]/fslstart.tcl

set VARS(history) {}

#}}}
#{{{ concatxfm

proc concatxfm { w } {

    #{{{ setup main window etc

    global entries FSLDIR PWD

    # ---- Set up Frames ----
    toplevel $w
    wm title $w "Concatxfm"
    wm iconname $w "Concatxfm"
    wm iconbitmap $w @${FSLDIR}/tcl/fmrib.xbm
    frame $w.f

#}}}

    TitleFrame $w.f.input -text "Input"  -relief groove 
    set lfinput [ $w.f.input getframe ]
    #{{{ input transform

set entries($w,xfm1) ""
set entries($w,xfm2) ""

FileEntry  $w.f.xfm -textvariable entries($w,xfm1) -label "Transformation Matrix for A to B           " -title "Select" -width 50 -filedialog directory  -filetypes *.mat
FileEntry  $w.f.xfm2 -textvariable entries($w,xfm2) -label "Transformation Matrix for B to C           " -title "Select" -width 50 -filedialog directory  -filetypes *.mat

#}}}

    pack $w.f.xfm $w.f.xfm2 -in $lfinput -side top -anchor w -pady 3 -padx 5

    TitleFrame $w.f.output -text "Output" -relief groove 
    set lfoutput [ $w.f.output getframe ]

    #{{{ output filename

set entries($w,outxfm) ""

FileEntry  $w.f.oxfm -textvariable entries($w,outxfm) -label  "Save Concatenated Transform (A to C) " -title "Select" -width 50 -filedialog directory  -filetypes *.mat

#}}}
    pack $w.f.oxfm -in $lfoutput -side top -anchor w -pady 3 -padx 5

    pack $w.f.input $w.f.output -in $w.f -side top -anchor w -pady 0 -padx 5

    #{{{ buttons

    # ---- Button Frame ----

    frame $w.btns
    frame $w.btns.b -relief raised -borderwidth 1
    
    button $w.apply     -command "Concatxfm:apply $w" \
	    -text "Go" -width 5

    button $w.cancel    -command "destroy $w" \
	    -text "Exit" -width 5

    button $w.help -command "FmribWebHelp file: ${FSLDIR}/doc/redirects/flirt.html" \
	    -text "Help" -width 5

    pack $w.btns.b -side bottom -fill x
    pack $w.apply $w.cancel $w.help -in $w.btns.b \
	    -side left -expand yes -padx 3 -pady 10 -fill y
    
    pack $w.f $w.btns -expand yes -fill both

#}}}
}

#}}}
#{{{ Concatxfm:apply

proc Concatxfm:apply { w } {
    global entries

    catch { concatxfm:proc $entries($w,xfm1) $entries($w,xfm2) $entries($w,outxfm) }

    update idletasks
    puts "Done"
}

#}}}
#{{{ concatxfm:proc

proc concatxfm:proc { transAB transBC outxfmfilename } {

    global FSLDIR

    fsl:exec "${FSLDIR}/bin/convert_xfm -omat $outxfmfilename -concat $transBC $transAB"

    if { [ file readable $outxfmfilename ] == 0 } {
	puts "No transformation saved!"
	return 4
    }

    return 0
}

#}}}
#{{{ call GUI

wm withdraw .
concatxfm .rename
tkwait window .rename

#}}}
