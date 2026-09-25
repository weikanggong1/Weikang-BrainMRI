set pagination off
set confirm off
break *0x47e890
run
tbreak *0x47ea15
continue
printf "LINE_START_SSE %.17g\n", $xmm0.v2_double[0]
tbreak *0x47cb50
continue
printf "FIRST_TRIAL_DT %.17g\n", $xmm0.v2_double[0]
finish
printf "FIRST_TRIAL_SSE %.17g\n", $xmm0.v2_double[0]
call (int)fflush(0)
quit
