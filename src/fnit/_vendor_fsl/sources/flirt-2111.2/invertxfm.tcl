#

# InvertXFM - the GUI for convert_xfm
#
# Mark Jenkinson, Stephen Smith and Matthew Webster, FMRIB Image Analysis Group
#
# Copyright (C) 2001-2006 University of Oxford
#
# TCLCOPYRIGHT

#{{{ setup

source [ file dirname [ info script ] ]/fslstart.tcl

#}}}
#{{{ invertxfm

proc invertxfm { w } {

    #{{{ setup main window etc

    global entries FSLDIR PWD

    # ---- Set up Frames ----
    toplevel $w
    wm title $w "InvertXFM"
    wm iconname $w "InvertXFM"
    wm iconbitmap $w @${FSLDIR}/tcl/fmrib.xbm
    frame $w.f

#}}}
    TitleFrame $w.f.input -text "Input"  -relief groove 
    set lfinput [ $w.f.input getframe ]
    #{{{ input transform

set entries($w,1) ""


    FileEntry  $w.f.xfm -textvariable entries($w,1) -label "Transformation Matrix for A to B   " -title "Select" -width 40 -filedialog directory  -filetypes *.mat
#}}}
    pack $w.f.xfm -in $lfinput -side top -anchor w -pady 3 -padx 5

    TitleFrame $w.f.output -text "Output" -relief groove 
    set lfoutput [ $w.f.output getframe ]
    #{{{ output filename

set entries($w,4) ""

FileEntry  $w.f.oxfm -textvariable entries($w,4) -label  "Save Inverse Transform (B to A)    " -title "Select" -width 40 -filedialog directory  -filetypes *.mat

#}}}
    pack $w.f.oxfm -in $lfoutput -side top -anchor w -pady 3 -padx 5

    pack $w.f.input $w.f.output -in $w.f -side top -anchor w -pady 0 -padx 5

    #{{{ buttons

    # ---- Button Frame ----

    frame $w.btns
    frame $w.btns.b -relief raised -borderwidth 1
    
    button $w.apply     -command "InvertXFM:apply $w" \
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
#{{{ InvertXFM:apply

proc InvertXFM:apply { w } {
    global entries

    catch { invertxfm:proc $entries($w,1) $entries($w,4) }

    update idletasks
    puts "Done"
}

#}}}
#{{{ invertxfm:proc

proc invertxfm:proc { transAB invxfmfilename } {

    global FSLDIR

    fsl:exec "${FSLDIR}/bin/convert_xfm -omat $invxfmfilename -inverse $transAB"

    if { [ file readable $invxfmfilename ] == 0 } {
	puts "No transformation saved!"
	return 4
    }

    return 0
}

#}}}
#{{{ call GUI

wm withdraw .
invertxfm .rename
tkwait window .rename

#}}}
