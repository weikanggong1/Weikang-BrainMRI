/*  img2stdcoord.cc

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2000 University of Oxford  */

/*  CCOPYRIGHT  */

#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>

#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS
#endif

#include "NewNifti/NewNifti.h"
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"
#include "warpfns/warpfns.h"
#include "warpfns/fnirt_file_reader.h"

using namespace std;
using namespace NiftiIO;
using namespace MISCMATHS;
using namespace NEWMAT;
using namespace NEWIMAGE;

////////////////////////////////////////////////////////////////////////////
// the real defaults are provided in the function parse_command_line

class globaloptions {
public:
  string stdfname;
  string imgfname;
  string prexfmfname;
  string coordfname;
  string warpfname;
  bool usestd;
  bool mm;
  int verbose;
public:
  globaloptions();
  ~globaloptions() {};
};

globaloptions globalopts;


globaloptions::globaloptions()
{
  // set up defaults
  stdfname = "";
  imgfname = "";
  coordfname = "";
  prexfmfname = "";
  warpfname = "";
  verbose = 0;
  usestd = false;
  mm = false;
}


////////////////////////////////////////////////////////////////////////////

// Parsing functions for command line parameters

void print_usage(int argc, char *argv[])
{
  cout << "Usage: " << argv[0] << " [options] <filename containing coordinates>\n"
       << "e.g.   " << argv[0] << " -img <invol> -std <standard image> -xfm <img2standard mat file> <coordinate file>\n"
       << "       " << argv[0] << " -img <invol> <coordinate file>\n"
       << "       " << argv[0] << " -img <invol> - \n\n"
       << "  Options are:\n"
       << "        -img <filename of input image>\n"
       << "        -std <filename of standard image>\n"
       << "        -xfm <filename of affine transform   (e.g. example_func2standard.mat)>\n"
       << "        -warp <filename of warpfield (e.g. highres2standard_warp.nii.gz)>\n"
       << "        -premat <filename of pre-warp affine transform  (e.g. example_func2highres.mat)>   (default=identity)\n"
       << "        -vox                                 (input coordinates in voxels - default)\n"
       << "        -mm                                  (input coordinates in mm)\n"
       << "        -v                                   (verbose output)\n"
       << "        -verbose                             (more verbose output)\n"
       << "        -help\n\n"
       << " Notes:\n"
       << "  (1) if '-' is used as coordinate filename then coordinates are read from standard input\n"
       << "  (2) the -img option is compulsory\n";
}


void parse_command_line(int argc, char* argv[])
{
  if(argc<2){
    print_usage(argc,argv);
    exit(1);
  }


  int n=1;
  string arg;
  char first;

  while (n<argc) {
    arg=argv[n];
    if (arg.size()<1) { n++; continue; }
    first = arg[0];
    if (first!='-') {
      globalopts.coordfname = arg;
      n++;
      continue;
    }

    // put options without arguments here
    if ( arg == "-help" ) {
      print_usage(argc,argv);
      exit(0);
    } else if ( arg == "-vox" ) {
      globalopts.mm = false;
      n++;
      continue;
    } else if ( arg == "-mm" ) {
      globalopts.mm = true;
      n++;
      continue;
    } else if ( arg == "-flirt" ) {
      cerr << "WARNING::Using outdated options, please update to new usage" << endl;
      // do nothing anymore
      n++;
      continue;
    } else if ( arg == "-v" ) {
      globalopts.verbose = 1;
      n++;
      continue;
    } else if ( arg == "-verbose" ) {
      globalopts.verbose = 5;
      n++;
      continue;
    } else if ( arg == "-") {
      globalopts.coordfname = "-";
      n++;
      continue;
    }

    if (n+1>=argc)
      {
	cerr << "Lacking argument to option " << arg << endl;
	break;
      }

    // put options with 1 argument here
    if ( arg == "-std") {
      globalopts.stdfname = argv[n+1];
      globalopts.usestd = true;
      n+=2;
      continue;
    } else if ( arg == "-img") {
      globalopts.imgfname = argv[n+1];
      n+=2;
      continue;
    } else if ( ( arg == "-xfm") || ( arg == "-premat") ) {
      globalopts.prexfmfname = argv[n+1];
      n+=2;
      continue;
    } else if ( arg == "-warp") {
      globalopts.warpfname = argv[n+1];
      n+=2;
      continue;
    } else {
      cerr << "Unrecognised option " << arg << endl;
      exit(-1);
    }

  }  // while (n<argc)

  if ((globalopts.imgfname.size()<1)) {
    cerr << "ERROR:: input image filename not found\n\n";
  }
  if ((globalopts.usestd) && (globalopts.stdfname.size()<1)) {
    cerr << "ERROR:: standard image filename not found\n\n";
  }
}

////////////////////////////////////////////////////////////////////////////

void print_info(const volume<float>& vol, const string& name) {
  cout << name << ":: SIZE = " << vol.xsize() << " x " << vol.ysize()
       << " x " << vol.zsize() << endl;
  cout << name << ":: DIMS = " << vol.xdim() << " x " << vol.ydim()
       << " x " << vol.zdim() << " mm" << endl << endl;
}

////////////////////////////////////////////////////////////////////////////

ColumnVector NewimageCoord2NewimageCoord(const FnirtFileReader& fnirtfile, const Matrix& affmat,
					 const volume<float>& srcvol, const volume<float>& destvol,
					 const ColumnVector& srccoord)
{
  ColumnVector retvec;
  if (fnirtfile.IsValid()) {
    // in the following affmat=example_func2highres.mat, fnirtfile=highres2standard_warp.nii.gz
    static volume4D<float> fieldVolume( fnirtfile.FieldAsNewimageVolume4D(true) );
    retvec = NewimageCoord2NewimageCoord(affmat,fieldVolume,true,srcvol,destvol,srccoord);
  } else {
    retvec = NewimageCoord2NewimageCoord(affmat,srcvol,destvol,srccoord);
  }
  return retvec;
}

////////////////////////////////////////////////////////////////////////////

int main(int argc,char *argv[])
{
  parse_command_line(argc,argv);


  volume<float> imgvol, stdvol;
    // read volumes
  if (read_volume_hdr_only(imgvol,globalopts.imgfname)<0) {
    cerr << "Cannot read input image" << endl;
    return -1;
  }
  if (globalopts.usestd && (read_volume_hdr_only(stdvol,globalopts.stdfname)<0)) {
    cerr << "Cannot read standard image" << endl;
    return -1;
  }

  if (globalopts.verbose>3) {
    if (globalopts.usestd) { print_info(stdvol,"standard image"); }
    print_info(imgvol,"input image");
  }

  // read matrices
  Matrix affmat(4,4);
  bool use_sform=false;
  if (globalopts.prexfmfname.length()>0) {
    affmat = read_ascii_matrix(globalopts.prexfmfname);
    use_sform = false;
    if (affmat.Nrows()<4) {
      cerr << "Cannot read transform file" << endl;
      return -2;
    }
  } else {
    use_sform = true;
    affmat = IdentityMatrix(4);
  }


  if (globalopts.verbose>3) {
    cout << " affmat =" << endl << affmat << endl << endl;
  }

  // Read in warps from file (if specified)
  FnirtFileReader  fnirtfile;
  AbsOrRelWarps    wt = UnknownWarps;
  if (globalopts.warpfname != "") {
    try {
      fnirtfile.Read(globalopts.warpfname,wt,globalopts.verbose>3);
    }
    catch (...) {
      cerr << "An error occured while reading file: " << globalopts.warpfname << endl;
      exit(EXIT_FAILURE);
    }
  }

  /////////////// SET UP MATRICES ////////////////

  if ( (stdvol.sform_code()==NIFTI_XFORM_UNKNOWN) &&
       (stdvol.qform_code()==NIFTI_XFORM_UNKNOWN) ) {
    cerr << "WARNING:: mm coordinate space not set in standard volume" << endl;
  }
  if (globalopts.verbose>3) {
    cout << " stdvox2world =" << endl << stdvol.newimagevox2mm_mat() << endl << endl;
  }

  // initialise coordinate vectors

  ColumnVector imgcoord(4), stdcoord(4), oldimg(4);
  imgcoord = 0;
  stdcoord = 0;
  imgcoord(4)=1;
  stdcoord(4)=1;
  oldimg = 0;  // 4th component set to 0, so that initially oldimg -ne imgcoord


  bool use_stdin = false;
  if ( (globalopts.coordfname=="-") || (globalopts.coordfname.size()<1)) {
    use_stdin = true;
  }

  if (globalopts.verbose>0) {
    cout << "Coordinates in standard image (in mm):" << endl;
  }

  // set up coordinate reading (from file or stdin) //
  ifstream matfile(globalopts.coordfname.c_str());

  if (use_stdin) {
    if (globalopts.verbose>0) {
      cout << "Please type in input image coordinates";
      if (globalopts.mm) {
	cout << " (in mm) :" << endl;
      } else {
	cout << " (in voxels) :" << endl;
      }
    }
  } else {
    if (!matfile) {
      cerr << "Could not open matrix file " << globalopts.coordfname << endl;
      return -1;
    }
  }

  if (use_stdin) {
    	if (isatty(STDIN_FILENO)){
    		cerr << "Interactive mode has been removed. Please pipe input such as: cat coordinate_file.txt | img2stdcoord, OR supply a coordinate file" << endl;
    		exit(1);
    	}
    }
  // loop around reading coordinates and displaying output
  while ( (use_stdin && (cin >> imgcoord(1) >> imgcoord(2) >> imgcoord(3))) || ((!use_stdin) && (matfile >> imgcoord(1) >> imgcoord(2) >> imgcoord(3))) ) {

    if (globalopts.mm) {  // in mm
      stdcoord = stdvol.newimagevox2mm_mat() *
	NewimageCoord2NewimageCoord(fnirtfile,affmat,imgvol,stdvol,imgvol.newimagevox2mm_mat().i() * imgcoord);
    } else { // in voxels
      stdcoord = stdvol.newimagevox2mm_mat() *
	NewimageCoord2NewimageCoord(fnirtfile,affmat,imgvol,stdvol,imgvol.niftivox2newimagevox_mat() * imgcoord);
    }
    cout << stdcoord(1) << "  " << stdcoord(2) << "  " << stdcoord(3) << endl;
  }

  if (!use_stdin) { matfile.close(); }
  return 0;
}
