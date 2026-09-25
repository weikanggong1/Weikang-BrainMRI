/*  imapper.cc

    Mark Jenkinson and Matt D'Netto, FMRIB Image Analysis Group

    Copyright (C) 2003 University of Oxford  */

/*  CCOPYRIGHT  */

// Intensity mapper based on registration cost function

#define _GNU_SOURCE 1
#define POSIX_SOURCE 1

#include "newimage/newimageall.h"
#include "newimage/costfns.h"
#include "miscmaths/miscmaths.h"
#include "utils/options.h"

using namespace MISCMATHS;
using namespace NEWIMAGE;
using namespace Utilities;

// The two strings below specify the title and example usage that is
//  printed out as the help or usage message

string title="imapper \nCopyright(c) 2008, University of Oxford (Mark Jenkinson and Matt D'Netto)";
string examples="imapper [options] -i <input image> -r <reference image> -o <output image> -a <affine matrix> --inweight=<input image weighting>";

// Each (global) object below specificies as option and can be accessed
//  anywhere in this file (since they are global).  The order of the
//  arguments needed is: name(s) of option, default value, help message,
//       whether it is compulsory, whether it requires arguments
// Note that they must also be included in the main() function or they
//  will not be active.

Option<bool> verbose(string("-v,--verbose"), false, 
		     string("switch on diagnostic messages"), 
		     false, no_argument);
Option<bool> help(string("-h,--help"), false,
		  string("display this message"),
		  false, no_argument);
Option<string> inname(string("-i,--in"), string(""),
		  string("input image filename"),
		  true, requires_argument);
Option<string> refname(string("-r,--ref"), string(""),
		  string("reference image filename"),
		  true, requires_argument);
Option<string> inweightname(string("--inweight"), string(""),
		  string("input weighting image filename"),
		  true, requires_argument);
Option<string> outname(string("-o,--out"), string(""),
		  string("output image filename"),
		  false, requires_argument);
Option<string> affmatname(string("-a,--affmat"), string(""),
		  string("affine matrix filename"),
		  true, requires_argument);
int nonoptarg;

////////////////////////////////////////////////////////////////////////////

// Local functions

int do_work(int argc, char* argv[]) 
{
  volume<float> vin, vref, vnew, vinweight, vrefweight;
  read_volume(vin,inname.value());
  read_volume(vinweight,inweightname.value());
  read_volume(vref,refname.value());
  vrefweight = vref*0.0f + 1.0f;
  Matrix aff;
  aff = read_ascii_matrix(affmatname.value());
  // convert aff into a voxel-voxel matrix
  aff = NewimageVox2NewimageVoxMatrix(aff,vin,vref);
  Costfn *cost;
  cost = new Costfn(vref,vin,vrefweight,vinweight);
  cost->set_costfn(CorrRatio);
  cost->set_no_bins(256);
  if (verbose.value()) { cout << "Cost = " << cost->cost(aff) << endl; }
  if (outname.set()) { 
    vnew = cost->image_mapper(aff);
    save_volume(vnew,outname.value());
  }
  Matrix mapper = cost->mappingfn(aff);
  cout << "Mapping function is" << endl << mapper << endl;
  return 0;
}

////////////////////////////////////////////////////////////////////////////

int main(int argc,char *argv[])
{

  Tracer tr("main");
  OptionParser options(title, examples);

  try {
    // must include all wanted options here (the order determines how
    //  the help message is printed)
    options.add(inname);
    options.add(refname);
    options.add(inweightname);
    options.add(affmatname);
    options.add(outname);
    options.add(verbose);
    options.add(help);
    
    nonoptarg = options.parse_command_line(argc, argv);

    // line below stops the program if the help was requested or 
    //  a compulsory option was not set
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

  // Call the local functions

  return do_work(argc,argv);
}

