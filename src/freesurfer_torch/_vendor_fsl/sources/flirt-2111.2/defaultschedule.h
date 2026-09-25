/*  defaultschedule.h

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 1999-2008 University of Oxford  */

/*  CCOPYRIGHT  */

// Set default schedule file
//  Written by Mark Jenkinson  11/10/99
//  Modified by Mark Jenkinson    08/08

#if !defined(__defaultschedule_h)
#define __defaultschedule_h

#include <vector>
#include <string>

void setdefaultschedule(std::vector<std::string>& comms)
{
  comms.clear();
  comms.push_back("# 8mm scale");
  comms.push_back("setscale 8");
  comms.push_back("setoption smoothing 8");
  comms.push_back("clear S");
  comms.push_back("clear P");
  comms.push_back("search");

  comms.push_back("# 4mm scale");
  comms.push_back("setscale 4");
  comms.push_back("setoption smoothing 4");
  comms.push_back("clear U");
  comms.push_back("clear UA ");
  comms.push_back("clear UB");
  comms.push_back("clear US");
  comms.push_back("clear UP");

  comms.push_back("# remeasure costs at this scale");
  comms.push_back("measurecost 7 S 0 0 0 0 0 0 rel");
  comms.push_back("copy U US");
  comms.push_back("clear U");
  comms.push_back("measurecost 7 P 0 0 0 0 0 0 rel");
  comms.push_back("copy U UP");
  comms.push_back("dualsort US UP");

  comms.push_back("# optimise best 3 candidates (pre and post 8mm optimisations)");
  comms.push_back("clear U");
  comms.push_back("optimise 7 US:1-3  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("optimise 7 UP:1-3  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("# also try the identity transform as a starting point at this resolution");
  comms.push_back("clear UQ");
  comms.push_back("setrow UQ  1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1");
  comms.push_back("optimise 7 UQ  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("sort U");
  comms.push_back("copy U UA");

  comms.push_back("# select best 4 optimised solutions and try perturbations of these");
  comms.push_back("clear U");
  comms.push_back("copy UA:1-4 U");
  comms.push_back("optimise 7 UA:1-4  1.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("optimise 7 UA:1-4 -1.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("optimise 7 UA:1-4  0.0   1.0   0.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("optimise 7 UA:1-4  0.0  -1.0   0.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("optimise 7 UA:1-4  0.0   0.0   1.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("optimise 7 UA:1-4  0.0   0.0  -1.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("optimise 7 UA:1-4  0.0   0.0   0.0   0.0   0.0   0.0   0.1  abs 4");
  comms.push_back("optimise 7 UA:1-4  0.0   0.0   0.0   0.0   0.0   0.0  -0.1  abs 4");
  comms.push_back("optimise 7 UA:1-4  0.0   0.0   0.0   0.0   0.0   0.0   0.2  abs 4");
  comms.push_back("optimise 7 UA:1-4  0.0   0.0   0.0   0.0   0.0   0.0  -0.2  abs 4");
  comms.push_back("sort U");
  comms.push_back("copy U UB");

  comms.push_back("# 2mm scale");
  comms.push_back("setscale 2");
  comms.push_back("setoption smoothing 2");
  comms.push_back("clear U");
  comms.push_back("clear UC");
  comms.push_back("clear UD");
  comms.push_back("clear UE");
  comms.push_back("clear UF");

  comms.push_back("# remeasure costs at this scale");
  comms.push_back("measurecost 7 UB 0 0 0 0 0 0 rel");
  comms.push_back("sort U");
  comms.push_back("copy U UC");

  comms.push_back("clear U");
  comms.push_back("optimise 7  UC:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("copy U UD");
  comms.push_back("setoption boundguess 1");
  comms.push_back("if MAXDOF > 7");
  comms.push_back(" clear U");
  comms.push_back("if MAXDOF > 7");
  comms.push_back(" optimise 9  UD:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 1");
  comms.push_back("copy U UE");
  comms.push_back("if MAXDOF > 9");
  comms.push_back(" clear U");
  comms.push_back("if MAXDOF > 9");
  comms.push_back(" optimise 12 UE:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 2");
  comms.push_back("sort U");
  comms.push_back("copy U UF");

  comms.push_back("# 1mm scale");
  comms.push_back("setscale 1");
  comms.push_back("setoption smoothing 1");
  comms.push_back("setoption boundguess 1");
  comms.push_back("clear U");
  comms.push_back("# also try the qsform as a starting point at this resolution");
  comms.push_back("setrowqsform UF");
  comms.push_back("optimise 12 UF:1-2  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 1");
  comms.push_back("# in addition, try qsform as the final transformation, not just an initialisation");
  comms.push_back("clear UG");
  comms.push_back("setrowqsform UG");
  comms.push_back("measurecost 12 UG:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 1");
  comms.push_back("sort U");
}


void set2Ddefaultschedule(std::vector<std::string>& comms)
{
  comms.clear();
  comms.push_back("# 8mm scale");
  comms.push_back("setscale 8");
  comms.push_back("setoption smoothing 8");
  comms.push_back("setoption paramsubset 3  0 0 1 0 0 0 0 0 0 0 0 0  0 0 0 1 0 0 0 0 0 0 0 0  0 0 0 0 1 0 0 0 0 0 0 0");
  comms.push_back("clear U");
  comms.push_back("clear UA");
  comms.push_back("setrow UA 1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1");
  comms.push_back("optimise 12 UA:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4 ");
  comms.push_back("");
  comms.push_back("# 4mm scale");
  comms.push_back("setscale 4");
  comms.push_back("setoption smoothing 4");
  comms.push_back("setoption paramsubset 3  0 0 1 0 0 0 0 0 0 0 0 0  0 0 0 1 0 0 0 0 0 0 0 0  0 0 0 0 1 0 0 0 0 0 0 0");
  comms.push_back("clear UB");
  comms.push_back("clear UL");
  comms.push_back("clear UM");
  comms.push_back("# remeasure costs at this scale");
  comms.push_back("clear U");
  comms.push_back("measurecost 12 UA 0 0 0 0 0 0 rel");
  comms.push_back("sort U");
  comms.push_back("copy U UL");
  comms.push_back("# optimise best 3 candidates");
  comms.push_back("clear U");
  comms.push_back("optimise 12 UL:1-3  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("# also try the identity transform as a starting point at this resolution");
  comms.push_back("clear UQ");
  comms.push_back("setrow UQ  1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1");
  comms.push_back("optimise 7 UQ  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("sort U");
  comms.push_back("copy U UM");
  comms.push_back("# select best 4 optimised solutions and try perturbations of these");
  comms.push_back("clear U");
  comms.push_back("copy UM:1-4 U");
  comms.push_back("optimise 12 UM:1-4  0.0   0.0   1.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("optimise 12 UM:1-4  0.0   0.0  -1.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("sort U");
  comms.push_back("clear UB");
  comms.push_back("copy U UB");
  comms.push_back("");
  comms.push_back("# 2mm scale");
  comms.push_back("setscale 2");
  comms.push_back("setoption smoothing 2");
  comms.push_back("setoption paramsubset 3  0 0 1 0 0 0 0 0 0 0 0 0  0 0 0 1 0 0 0 0 0 0 0 0  0 0 0 0 1 0 0 0 0 0 0 0");
  comms.push_back("clear U");
  comms.push_back("clear UC");
  comms.push_back("clear UD");
  comms.push_back("clear UE");
  comms.push_back("clear UF");
  comms.push_back("# remeasure costs at this scale");
  comms.push_back("measurecost 12 UB 0 0 0 0 0 0 rel");
  comms.push_back("sort U");
  comms.push_back("copy U UC");
  comms.push_back("clear U");
  comms.push_back("optimise 12  UC:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 4");
  comms.push_back("copy U UD");
  comms.push_back("setoption boundguess 1");
  comms.push_back("if MAXDOF > 7");
  comms.push_back(" clear U");
  comms.push_back("if MAXDOF > 7");
  comms.push_back(" optimise 9  UD:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 1");
  comms.push_back("copy U UE");
  comms.push_back("if MAXDOF > 9");
  comms.push_back(" clear U");
  comms.push_back("if MAXDOF > 9");
  comms.push_back(" optimise 12 UE:1  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 2");
  comms.push_back("sort U");
  comms.push_back("copy U UF");
  comms.push_back("");
  comms.push_back("# 1mm scale");
  comms.push_back("setscale 1");
  comms.push_back("setoption smoothing 1");
  comms.push_back("setoption boundguess 1");
  comms.push_back("setoption paramsubset 3  0 0 1 0 0 0 0 0 0 0 0 0  0 0 0 1 0 0 0 0 0 0 0 0  0 0 0 0 1 0 0 0 0 0 0 0");
  comms.push_back("clear U");
  comms.push_back("# also try the qsform as a starting point at this resolution");
  comms.push_back("setrowqsform UF");
  comms.push_back("optimise 12 UF:1-2  0.0   0.0   0.0   0.0   0.0   0.0   0.0  rel 1");
  comms.push_back("sort U");
}

#endif
