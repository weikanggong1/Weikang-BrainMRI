#!/usr/bin/env python3
"""Add scoped wall timers to an isolated FreeSurfer mrisurf_mri.cpp copy."""

import pathlib
import sys


source = pathlib.Path(sys.argv[1])
text = source.read_text()
start = text.index("int MRISpositionSurface(MRI_SURFACE *mris,")
end = text.index("int MRISpositionSurface_mef(", start)
body = text[start:end]


def add(old: str, new: str) -> None:
    global body
    if body.count(old) != 1:
        raise ValueError(f"Expected one match, found {body.count(old)}: {old[:70]}")
    body = body.replace(old, new)


add(
    "  printf(\"Entering MRISpositionSurface()\\n\");",
    """  using ProfileClock = std::chrono::steady_clock;
  auto profile_ms = [](ProfileClock::time_point a, ProfileClock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto profile_call_start = ProfileClock::now();
  auto profile_t = profile_call_start;
  double profile_mht = 0, profile_target = 0, profile_intensity = 0;
  double profile_gradient_other = 0, profile_repulsive = 0;
  double profile_step_prep = 0, profile_step = 0, profile_step_after = 0;
  double profile_metric = 0, profile_rms = 0, profile_sse = 0;
  int profile_iterations = 0, profile_trials = 0;
  printf("Entering MRISpositionSurface()\\n");""",
)
add(
    "  for (n = parms->start_t; n < parms->start_t + niterations; n++) {\n\n    parms->t = n;",
    "  for (n = parms->start_t; n < parms->start_t + niterations; n++) {\n\n    ++profile_iterations;\n    profile_t = ProfileClock::now();\n    parms->t = n;",
)
add(
    "    MRISclearGradient(mris);\n\n    // Compute the gradient direction",
    "    MRISclearGradient(mris);\n    profile_mht += profile_ms(profile_t, ProfileClock::now());\n    profile_t = ProfileClock::now();\n\n    // Compute the gradient direction",
)
add(
    "    mrisComputeTargetLocationTerm(mris, parms->l_location, parms);\n    mrisComputeIntensityTerm",
    "    mrisComputeTargetLocationTerm(mris, parms->l_location, parms);\n    profile_target += profile_ms(profile_t, ProfileClock::now());\n    profile_t = ProfileClock::now();\n    mrisComputeIntensityTerm",
)
add(
    "    mrisComputeIntensityTerm(mris, l_intensity, mri_brain, mri_smooth, parms->sigma, parms);\n    mrisComputeShrinkwrapTerm",
    "    mrisComputeIntensityTerm(mris, l_intensity, mri_brain, mri_smooth, parms->sigma, parms);\n    profile_intensity += profile_ms(profile_t, ProfileClock::now());\n    profile_t = ProfileClock::now();\n    mrisComputeShrinkwrapTerm",
)
add(
    "    mrisComputeRepulsiveTerm(mris, parms->l_repulse, mht_v_current, mht_f_current);\n    mrisComputeThicknessSmoothnessTerm",
    "    profile_gradient_other += profile_ms(profile_t, ProfileClock::now());\n    profile_t = ProfileClock::now();\n    mrisComputeRepulsiveTerm(mris, parms->l_repulse, mht_v_current, mht_f_current);\n    profile_repulsive += profile_ms(profile_t, ProfileClock::now());\n    profile_t = ProfileClock::now();\n    mrisComputeThicknessSmoothnessTerm",
)
add(
    "    size_t hash_count = 0, hash_limit = 1;",
    "    profile_gradient_other += profile_ms(profile_t, ProfileClock::now());\n    size_t hash_count = 0, hash_limit = 1;",
)
add(
    "    do { // do loops alway execute at least once\n      // save vertex positions",
    "    do { // do loops alway execute at least once\n      ++profile_trials;\n      profile_t = ProfileClock::now();\n      // save vertex positions",
)
add(
    "      delta_t = mrisAsynchronousTimeStep(mris, parms->momentum, dt, mht, max_mm);\n      parms->t = n + 1;",
    "      profile_step_prep += profile_ms(profile_t, ProfileClock::now());\n      profile_t = ProfileClock::now();\n      delta_t = mrisAsynchronousTimeStep(mris, parms->momentum, dt, mht, max_mm);\n      profile_step += profile_ms(profile_t, ProfileClock::now());\n      profile_t = ProfileClock::now();\n      parms->t = n + 1;",
)
add(
    "      MRIScomputeMetricProperties(mris);\n\n      // Compute RMS",
    "      profile_step_after += profile_ms(profile_t, ProfileClock::now());\n      profile_t = ProfileClock::now();\n      MRIScomputeMetricProperties(mris);\n      profile_metric += profile_ms(profile_t, ProfileClock::now());\n      profile_t = ProfileClock::now();\n\n      // Compute RMS",
)
add(
    "        rms = mrisRmsValError(mris, mri_brain);\n      }\n      \n      if (debugNonDeterminism)",
    "        rms = mrisRmsValError(mris, mri_brain);\n      }\n      profile_rms += profile_ms(profile_t, ProfileClock::now());\n      \n      if (debugNonDeterminism)",
)
add(
    "      sse = MRIScomputeSSE(mris, parms);\n\n      if (debugNonDeterminism)",
    "      profile_t = ProfileClock::now();\n      sse = MRIScomputeSSE(mris, parms);\n      profile_sse += profile_ms(profile_t, ProfileClock::now());\n\n      if (debugNonDeterminism)",
)
add(
    "  return (NO_ERROR);\n}",
    """  fprintf(stderr,
          "#@# PROFILE MRISpositionSurface verts=%d iterations=%d trials=%d "
          "total_ms=%.6f mht_ms=%.6f target_ms=%.6f intensity_ms=%.6f "
          "gradient_other_ms=%.6f repulsive_ms=%.6f step_prep_ms=%.6f "
          "step_ms=%.6f step_after_ms=%.6f metric_ms=%.6f rms_ms=%.6f sse_ms=%.6f\\n",
          mris->nvertices, profile_iterations, profile_trials,
          profile_ms(profile_call_start, ProfileClock::now()), profile_mht,
          profile_target, profile_intensity, profile_gradient_other,
          profile_repulsive, profile_step_prep, profile_step,
          profile_step_after, profile_metric, profile_rms, profile_sse);
  return (NO_ERROR);
}""",
)
text = text[:start] + body + text[end:]
text = text.replace('#include "mrisurf_mri.h"', '#include <chrono>\n#include "mrisurf_mri.h"', 1)
source.write_text(text)
