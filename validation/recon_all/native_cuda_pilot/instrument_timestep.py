#!/usr/bin/env python3
"""Time the partition and collision phases in an isolated FreeSurfer source copy."""

import pathlib
import sys


source = pathlib.Path(sys.argv[1])
text = source.read_text()
start = text.index("static void mrisAsynchronousTimeStep_optionalDxDyDzUpdate(")
end = text.index("double mrisAsynchronousTimeStep(", start)
body = text[start:end]


def add(old: str, new: str) -> None:
    global body
    if body.count(old) != 1:
        raise ValueError(f"Expected one match, found {body.count(old)}: {old[:70]}")
    body = body.replace(old, new)


add(
    "  // This is the parallel algorithm\n  bool   const debug",
    """  // This is the parallel algorithm
  using ProfileClock = std::chrono::steady_clock;
  auto profile_ms = [](ProfileClock::time_point a, ProfileClock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };
  auto profile_start = ProfileClock::now();
  auto profile_t = profile_start;
  double profile_bounds = 0, profile_faces = 0;
  double profile_vertices = 0, profile_merge = 0;
  double profile_pass_ms[2] = {0, 0};
  bool   const debug""",
)
add(
    "  // Assign the faces to the subvolumes",
    "  profile_bounds = profile_ms(profile_t, ProfileClock::now());\n  profile_t = ProfileClock::now();\n  // Assign the faces to the subvolumes",
)
add(
    "  // In parallel, assign each vertex to either a subvolume",
    "  profile_faces = profile_ms(profile_t, ProfileClock::now());\n  profile_t = ProfileClock::now();\n  // In parallel, assign each vertex to either a subvolume",
)
add(
    "  // Merge the per thread subvolumes into the tid==0 subvolume",
    "  profile_vertices = profile_ms(profile_t, ProfileClock::now());\n  profile_t = ProfileClock::now();\n  // Merge the per thread subvolumes into the tid==0 subvolume",
)
add(
    "  // Pass 0: In parallel, process each subvolume",
    "  profile_merge = profile_ms(profile_t, ProfileClock::now());\n  // Pass 0: In parallel, process each subvolume",
)
add(
    "    for (pass=0; pass<2; pass++) {\n      int const sviLo",
    "    for (pass=0; pass<2; pass++) {\n      auto profile_pass_start = ProfileClock::now();\n      int const sviLo",
)
add(
    "      ROMP_PF_end\n    }\n\n    MHT_maybeParallel_end();",
    "      ROMP_PF_end\n      profile_pass_ms[pass] += profile_ms(profile_pass_start, ProfileClock::now());\n    }\n\n    MHT_maybeParallel_end();",
)
add(
    "  free(vertexInfos);\n#endif",
    """  free(vertexInfos);
  fprintf(stderr,
          "#@# PROFILE mrisAsynchronousTimeStep verts=%d faces=%d "
          "total_ms=%.6f bounds_ms=%.6f faces_ms=%.6f vertices_ms=%.6f "
          "merge_ms=%.6f pass0_ms=%.6f pass1_ms=%.6f\\n",
          mris->nvertices, mris->nfaces,
          profile_ms(profile_start, ProfileClock::now()),
          profile_bounds, profile_faces, profile_vertices, profile_merge,
          profile_pass_ms[0], profile_pass_ms[1]);
#endif""",
)
text = text[:start] + body + text[end:]
text = text.replace('#include "mrisurf_timeStep.h"', '#include <chrono>\n#include "mrisurf_timeStep.h"', 1)
source.write_text(text)
