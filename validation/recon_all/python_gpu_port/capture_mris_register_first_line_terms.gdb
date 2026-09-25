set pagination off
set confirm off
set environment FREESURFER_logSSE 1
break *0x47edb7
run
finish
call (int) fflush(0)
quit
