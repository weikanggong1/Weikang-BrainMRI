/*  procrustes.cc

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2001 University of Oxford  */

/*  CCOPYRIGHT  */

#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "utils/options.h"
#include "newimage/newimageall.h"

using namespace std;
using namespace Utilities;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;

////////////////////////////////////////////////////////////////////////////

// COMMAND LINE OPTIONS

string title="pointflirt \nCopyright(c) 2001, University of Oxford (Mark Jenkinson)";
string examples="pointflirt -i <invol coords>  -r <refvol coords> -o <output matrix>\npointflirt -i <invol coords>  -r <refvol coords> -o <output matrix> --vox --invol=<input vol> --refvol=<ref vol>";

Option<bool> verbose(string("-v,--verbose"), false,
		     string("switch on diagnostic messages"),
		     false, no_argument);
Option<bool> help(string("-h,--help"), false,
		  string("display this message"),
		  false, no_argument);
Option<bool> voxcoord(string("--vox"), false,
		  string("use voxel coordinates, not mm, for input"),
		  false, no_argument);
Option<string> omatname(string("-o,--out"), string(""),
			  string("filename of affine matrix output"),
			  false, requires_argument);
Option<string> incoord(string("-i"), string(""),
			  string("filename of input volume coordinates"),
			  true, requires_argument);
Option<string> refcoord(string("-r"), string(""),
			  string("filename of reference volume coordinates"),
			  true, requires_argument);
Option<string> invol(string("--invol"), string(""),
			  string("filename of input volume (needed for --vox option)"),
			  false, requires_argument);
Option<string> refvol(string("--refvol"), string(""),
			  string("filename of reference volume (needed for --vox option)"),
			  false, requires_argument);

////////////////////////////////////////////////////////////////////////////


int procrustes()
{

  Matrix x, y, s, w, q, b, p, zz, z3, xbar, ybar, xyt, xytv;
  int n;

  x=read_ascii_matrix(incoord.value());
  y=read_ascii_matrix(refcoord.value());

  if (x.Nrows() > x.Ncols()) { x = x.t(); y = y.t(); }

  n = x.Ncols();
  if (y.Ncols() != n ) {
    cerr << "ERROR: Different number of coordinates in the two files" << endl;
    exit(-1);
  }

  if (voxcoord.value()) {
    // convert coords from voxels to mm
    if (invol.unset() || refvol.unset()) {
      cerr << "ERROR: need to specify --invol and --refvol to use --vox" << endl;
      exit(2);
    }
    volume<float> inv,refv;
    read_volume_hdr_only(inv,invol.value());
    read_volume_hdr_only(refv,refvol.value());
    for (int m=1; m<=n; m++) {
	x(1,m) = x(1,m)*inv.xdim();
	x(2,m) = x(2,m)*inv.ydim();
	x(3,m) = x(3,m)*inv.zdim();
	y(1,m) = y(1,m)*refv.xdim();
	y(2,m) = y(2,m)*refv.ydim();
	y(3,m) = y(3,m)*refv.zdim();
    }
  }

  if (verbose.value()) {
    cout << "Input volume coordinates are (in mm)" << endl << x << endl;
    cout << "Reference volume coordinates are (in mm)" << endl << y << endl;
  }


  s=x*x.t();
  xbar = mean(x,2);
  ybar = mean(y,2);
  xyt = x*y.t();

  q = zeros(12,1);
  q.SubMatrix(1,3,1,1) = xyt.SubMatrix(1,3,1,1);
  q.SubMatrix(4,6,1,1) = xyt.SubMatrix(1,3,2,2);
  q.SubMatrix(7,9,1,1) = xyt.SubMatrix(1,3,3,3);
  q.SubMatrix(10,12,1,1) = ybar;

  w=zeros(9,3);
  w.SubMatrix(1,3,1,1) = xbar;
  w.SubMatrix(4,6,2,2) = xbar;
  w.SubMatrix(7,9,3,3) = xbar;

  b=zeros(12,12);
  b.SubMatrix(1,3,1,3) = s;
  b.SubMatrix(4,6,4,6) = s;
  b.SubMatrix(7,9,7,9) = s;
  b.SubMatrix(1,9,10,12) = n*w;
  b.SubMatrix(10,12,1,9) = w.t();
  b.SubMatrix(10,12,10,12) = IdentityMatrix(3);

  Matrix uu, vv;
  DiagonalMatrix dd;
  SVD(b,dd,uu,vv);
  int nullspace=0;
  for (int k=1; k<=12; k++) {
    if (dd(k,k) > 1e-8) {
      dd(k,k) = 1.0/dd(k,k);
    } else {
      dd(k,k) = 0.0;
      nullspace++;
    }
  }

  if (nullspace>0) {
    cout << "WARNING: nullspace of dimension " << nullspace << endl;
    cout << "  ... lacks sufficient independent coordinates" << endl;
    cout << "Finding non-unique solution" << endl;
  }

  p = vv*dd*uu.t() * q;
  Matrix affmat = zeros(4,4);
  affmat.SubMatrix(1,1,1,3) = p.SubMatrix(1,3,1,1).t();
  affmat.SubMatrix(2,2,1,3) = p.SubMatrix(4,6,1,1).t();
  affmat.SubMatrix(3,3,1,3) = p.SubMatrix(7,9,1,1).t();
  affmat(1,4) = p(10,1);
  affmat(2,4) = p(11,1);
  affmat(3,4) = p(12,1);
  affmat(4,4) = 1.0;

  if (omatname.set()) {
    write_ascii_matrix(affmat,omatname.value());
  } else {
    cout << affmat << endl;
  }

  return 0;
}




int main(int argc, char* argv[])
{
  Tracer tr("main");

  OptionParser options(title, examples);

  try {
    options.add(incoord);
    options.add(refcoord);
    options.add(omatname);
    options.add(voxcoord);
    options.add(invol);
    options.add(refvol);
    options.add(verbose);
    options.add(help);

    options.parse_command_line(argc, argv);

    if ( (help.value()) || (!options.check_compulsory_arguments(true)) )
      {
	options.usage();
	exit(EXIT_FAILURE);
      }
  }  catch(X_OptionError& e) {
    options.usage();
    cerr << endl << e.what() << endl;
    exit(EXIT_FAILURE);
  } catch(std::exception &e) {
    cerr << e.what() << endl;
  }

  return procrustes();
}
