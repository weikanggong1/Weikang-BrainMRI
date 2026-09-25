/*  globaloptions.h

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 1999-2000 University of Oxford  */

/*  CCOPYRIGHT  */

#ifndef __GLOBALOPTIONS_
#define __GLOBALOPTIONS_

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "armawrap/newmat.h"
#include "newimage/costfns.h"


namespace NEWIMAGE {
  enum anglereps { Euler, Quaternion };
  enum interps { TriLinear, NearestNeighbour, Sinc, Spline };
  enum windowtype { Rect, Hanning, Blackman };
}

typedef std::vector<NEWMAT::RowVector> MatVec;
typedef MatVec* MatVecPtr;


class globaloptions {
 public:
  static globaloptions& get();
  ~globaloptions() { delete gopt; }

  std::string version;

  std::vector<MatVec> usrmat;
  MatVec searchoptmat;
  MatVec preoptsearchmat;

  std::string inputfname;
  std::string outputfname;
  std::string reffname;
  std::string outputmatascii;
  std::string initmatfname;
  std::string refweightfname;
  std::string testweightfname;
  std::string wmsegfname;
  std::string wmcoordsfname;
  std::string wmnormsfname;
  std::string fmapfname;
  std::string fmapmaskfname;
  bool initmatsqform;
  bool printinit;
  NEWMAT::Matrix initmat;

  std::string schedulefname;

  NEWIMAGE::Costfn *impair;
  NEWMAT::ColumnVector refparams;
  NEWMAT::Matrix parammask;
  int no_params;
  int dof;
  bool usrsubset;
  int searchdof;
  int no_bins;
  NEWIMAGE::costfns maincostfn;
  NEWIMAGE::costfns searchcostfn;
  NEWIMAGE::costfns currentcostfn;
  std::string optimisationtype;
  NEWIMAGE::anglereps anglerep;
  float isoscale;
  float min_sampling;
  float lastsampling;
  float requestedscale;
  bool force_basescale;
  float basescale;
  bool force_scaling;
  float smoothsize;
  float fuzzyfrac;
  NEWMAT::ColumnVector tolerance;
  NEWMAT::ColumnVector boundguess;

  NEWMAT::ColumnVector searchrx;
  NEWMAT::ColumnVector searchry;
  NEWMAT::ColumnVector searchrz;
  float coarsedelta;
  float finedelta;

  short datatype;
  bool forcedatatype;
  int verbose;
  bool debug;
  bool interactive;
  bool do_optimise;
  bool nosave;
  bool iso;
  bool resample;
  bool useweights;
  bool useseg;
  bool usecoords;
  bool mode2D;
  bool clamping;
  bool forcebackgnd;
  float backgndval;
  bool interpblur;
  NEWIMAGE::interps interpmethod;
  float sincwidth;
  NEWIMAGE::windowtype sincwindow;
  float paddingsize;
  int pe_dir;
  float echo_spacing;
  std::string bbr_type;
  float bbr_slope;

  int single_param;

  void parse_command_line(int argc, char** argv, const std::string &);

 private:
  globaloptions();

  const globaloptions& operator=(globaloptions&);
  globaloptions(globaloptions&);

  static globaloptions* gopt;

  void print_usage(int argc, char *argv[]);
  void print_version();

};



//--------------------------------------------------------------------------//


inline globaloptions& globaloptions::get(){
  if(gopt == NULL)
    gopt = new globaloptions();

  return *gopt;
}

inline globaloptions::globaloptions()
{
  // set up defaults

  version = "";

  searchoptmat.clear();
  preoptsearchmat.clear();
  usrmat.resize(27);
  for (unsigned int i=0; i<usrmat.size(); i++) {
    usrmat[i].clear();
  }

  outputfname = "";
  reffname = "";

  inputfname = "";
  outputmatascii = "";
  initmatfname = "";
  refweightfname = "";
  testweightfname = "";
  wmsegfname = "";
  wmcoordsfname = "";
  wmnormsfname = "";
  fmapfname = "";
  fmapmaskfname = "";
  initmat = NEWMAT::IdentityMatrix(4);
  initmatsqform = false;
  printinit = false;

  schedulefname = "";

  impair = 0;
  refparams.ReSize(12);
  refparams << 0.0 << 0.0 << 0.0 << 0.0 << 0.0 << 0.0 << 1.0 << 1.0 << 1.0
	    << 0.0 << 0.0 << 0.0;
  parammask=NEWMAT::IdentityMatrix(12);
  no_params = 12;
  dof = 12;
  usrsubset = false;
  searchdof = 12;
  no_bins = 256;
  maincostfn = NEWIMAGE::CorrRatio;
  searchcostfn = NEWIMAGE::CorrRatio;
  currentcostfn = NEWIMAGE::CorrRatio;
  optimisationtype = "brent";
  anglerep = NEWIMAGE::Euler;
  isoscale = 1.0;
  min_sampling = 1.0;
  lastsampling = 8;
  requestedscale = 1.0;
  force_basescale = false;
  basescale = 1.0;
  force_scaling = false;
  smoothsize = 1.0;
  fuzzyfrac = 0.5;
  tolerance.ReSize(12);
  tolerance << 0.005 << 0.005 << 0.005 << 0.2 << 0.2 << 0.2 << 0.002
	    << 0.002 << 0.002 << 0.001 << 0.001 << 0.001;
  boundguess.ReSize(2);
  boundguess << 10.0 << 1.0;

  searchrx.ReSize(2);
  searchry.ReSize(2);
  searchrz.ReSize(2);
  searchrx << -M_PI/2.0 << M_PI/2.0;
  searchry << -M_PI/2.0 << M_PI/2.0;
  searchrz << -M_PI/2.0 << M_PI/2.0;
  coarsedelta = 60.0*M_PI/180.0;
  finedelta = 18.0*M_PI/180.0;

  datatype = -1;
  forcedatatype = false;
  verbose = 0;
  debug = false;
  interactive = false;
  do_optimise = true;
  nosave = true;
  iso = true;
  resample = true;
  useweights = false;
  useseg = false;
  usecoords = false;
  mode2D = false;
  clamping = true;
  forcebackgnd = false;
  backgndval = 0.0;
  interpblur = true;
  interpmethod = NEWIMAGE::TriLinear;
  sincwidth = 7.0; // voxels
  sincwindow = NEWIMAGE::Hanning;
  paddingsize = 0.0;
  pe_dir=0;   // 1=x, 2=y, 3=z, -1=-x, -2=-y, -3=-z, 0=none
  echo_spacing = 5e-4;  // random guess (0.5ms) - units of seconds
  bbr_type = "signed";
  bbr_slope = -0.5;

  single_param = -1;
}

#endif
