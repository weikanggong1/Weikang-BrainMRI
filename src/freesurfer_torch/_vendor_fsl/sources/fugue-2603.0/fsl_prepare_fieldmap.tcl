#

# FMRIB_Prepare_Fieldmap - the GUI for fmrib_prepare_fieldmap
#
# Mark Jenkinson, FMRIB Image Analysis Group
#
# Copyright (C) 2008 University of Oxford
#
# TCLCOPYRIGHT


# setup
source [ file dirname [ info script ] ]/fslstart.tcl

proc prepfmap { w } {

    global entries FSLDIR PWD

    # ---- Set up Frames ----
    toplevel $w
    wm title $w "FSL Prepare Fieldmap"
    wm iconname $w "FSL Prepare Fieldmap"
    wm iconbitmap $w @${FSLDIR}/tcl/fmrib.xbm
    frame $w.f

    # ----- Inputs ------

    TitleFrame $w.f.input -text "Input Images"  -relief groove
    set lfinput [ $w.f.input getframe ]


    set entries($w,phim) ""
    set entries($w,magim) ""

    FileEntry  $w.f.phase -textvariable entries($w,phim) -label "  Phase Image" -title "Select" -width 40 -filedialog directory  -filetypes IMAGE
    $w.f.phase.labf configure -width 29
    FileEntry  $w.f.mag -textvariable entries($w,magim) -label  "  Magnitude Image (Brain Extracted) " -title "Select" -width 40 -filedialog directory  -filetypes IMAGE
    $w.f.mag.labf configure -width 29


    pack $w.f.phase $w.f.mag -in $lfinput -side top -anchor w -pady 3 -padx 5

    # ----- Scanner ------

    TitleFrame $w.f.scanner -text "Scanner" -relief groove
    set lfscanner [ $w.f.scanner getframe ]

    set entries($w,scanner) "SIEMENS"

    radiobutton $w.f.ocmr -text "  Siemens" \
        -variable entries($w,scanner) -value SIEMENS -anchor w -width 18 -command "prepfmap:updatephase $w"
    radiobutton $w.f.gehc -text "  GEHC" \
        -variable entries($w,scanner) -value GEHC_FIELDMAPHZ -anchor w -command "prepfmap:updatephase $w"
    pack $w.f.ocmr $w.f.gehc -in $lfscanner -side left -anchor w -pady 3 -padx 5 -expand yes -fill x

    # ---- Delta TE ----

    TitleFrame $w.f.dTE -text "Fieldmap Sequence Parameters" -relief groove
    set lfdte [ $w.f.dTE getframe ]

    set entries($w,deltate) 2.5

    LabelSpinBox $w.f.deltaTE -label "  Difference of Echo Times (in milliseconds)  " -textvariable entries($w,deltate) -range {0.0 100.0 0.01 }

    pack $w.f.deltaTE -in $lfdte -side top -anchor w -padx 5 -pady 5

    # ----- Output ------

    TitleFrame $w.f.output -text "Output" -relief groove
    set lfoutput [ $w.f.output getframe ]

    set entries($w,fmap) ""

    FileEntry  $w.f.fmap -textvariable entries($w,fmap) -label  "  Fieldmap image (rad/s)    " -title "Select" -width 40 -filedialog directory  -filetypes IMAGE

    pack $w.f.fmap -in $lfoutput -side top -anchor w -pady 3 -padx 5

    # ---- Global Pack -----

    pack $w.f.scanner $w.f.input $w.f.dTE $w.f.output -in $w.f -side top -anchor w -pady 0 -padx 5 -expand yes -fill x

    # ---- Button Frame ----

    frame $w.btns
    frame $w.btns.b -relief raised -borderwidth 1

    button $w.apply     -command "Prepfmap:apply $w" \
	    -text "Go" -width 5

    button $w.cancel    -command "destroy $w" \
	    -text "Exit" -width 5

    button $w.help -command "FmribWebHelp file: ${FSLDIR}/doc/redirects/flirt.html" \
	    -text "Help" -width 5

    pack $w.btns.b -side bottom -fill x
#    pack $w.apply $w.cancel $w.help -in $w.btns.b
    pack $w.apply $w.cancel -in $w.btns.b \
	    -side left -expand yes -padx 3 -pady 10 -fill y

    pack $w.f $w.btns -expand yes -fill both

}


proc prepfmap:updatephase { w } {
    global entries
    if { [ string match $entries($w,scanner) "GEHC_FIELDMAPHZ" ] } {
        $w.f.phase configure -label "  Fieldmap Image (in Hz)"
    } else {
        $w.f.phase configure -label "  Phase Image"
    }

}


proc Prepfmap:apply { w } {
    global entries

    catch { prepfmap:proc $entries($w,scanner) $entries($w,phim) $entries($w,magim) $entries($w,deltate) $entries($w,fmap) }

    update idletasks
    puts "Done"
}


proc prepfmap:proc { scanner phim magim deltate fmap } {

    global FSLDIR

    # Do pop-up

    set count 0
    set w1 ".dialog[incr count]"
    while { [ winfo exists $w1 ] } {
        set w1 ".dialog[incr count]"
    }
    toplevel $w1
    wm title $w1 "Preparing Fieldmap"
    wm iconname $w1 "PrepareFieldmapOutput"
    wm iconbitmap $w1 @${FSLDIR}/tcl/fmrib.xbm
    frame $w1.sprev
    label $w1.sprev.label -text "\n    Running script ... please wait    \n\n"
    pack $w1.sprev.label -in $w1.sprev
    pack $w1.sprev -in $w1
    # force this message to popup now
    update

    # run command
    set thecommand "${FSLDIR}/bin/fsl_prepare_fieldmap $scanner $phim $magim $fmap $deltate"
    puts $thecommand
    set ret [ catch { exec sh -c "$thecommand" } otxt ]

    # show output
    pack forget $w1.sprev.label
    pack forget $w1.sprev
    pack $w1.sprev.label -in $w1.sprev
    set tottxt "Output from script:

$otxt"
    $w1.sprev.label configure -text "$tottxt"
    button $w1.cancel -command "destroy $w1" -text "Dismiss"
    pack $w1.sprev $w1.cancel -in $w1
    update
    return 0
}

# Call GUI

wm withdraw .
prepfmap .rename
tkwait window .rename
