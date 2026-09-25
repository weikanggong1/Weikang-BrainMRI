/*  sigloss.cc

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2004 University of Oxford  */

/*  CCOPYRIGHT  */

// Calculates an estimated signal loss from a b0map
// Can also use this for separate left-right gradient estimation (internal function)

#include <iostream>
#include "utils/options.h"
#include "armawrap/newmat.h"
#include "warpfns/warpfns.h"
#include "miscmaths/miscmaths.h"

using namespace std;
using namespace Utilities;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;

// The two strings below specify the title and example usage that is
//  printed out as the help or usage message

string title="sigloss \nCopyright(c) 2004, University of Oxford (Mark Jenkinson)\nEstimates signal loss from a field map (in rad/s) - possibly saved by fugue\n";
string examples="sigloss [options] -i b0map -m mask -s sigloss_estimate";

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
Option<float> te(string("--te"), 1.0,
		  string("echo time (in seconds)"),
		  false, requires_argument);
Option<string> slicedir(string("-d,--slicedir"), string("z"),
		  string("slice direction (either x , y or z)"),
		  false, requires_argument);
Option<string> inname(string("-i,--in"), string(""),
		  string("input b0 map image filename (in rad/s)"),
		  true, requires_argument);
Option<string> maskname(string("-m,--mask"), string(""),
		  string("input mask filename"),
		  false, requires_argument);
Option<string> siglossname(string("-s,--sigloss"), string(""),
		  string("output signal loss image filename"),
		  true, requires_argument);

int nonoptarg;

////////////////////////////////////////////////////////////////////////////

// Local functions

// for example ... print difference of COGs between 2 images ...
int do_work(int argc, char* argv[])
{
  volume<float> vin, mask;
  read_volume(vin,inname.value());
  if (maskname.set()) {
    read_volume(mask,maskname.value());
  } else {
    mask = vin * 0.0f + 1.0f;
  }

  volume4D<float> lrgrad;
  string dir;
  dir=slicedir.value();
  if ( (dir=="x") || (dir=="x-") ) {
    lrgrad = lrxgrad(vin,mask);
  }
  if ( (dir=="y") || (dir=="y-") ) {
    lrgrad = lrygrad(vin,mask);
  }
  if ( (dir=="z") || (dir=="z-") ) {
    lrgrad = lrzgrad(vin,mask);
  }

  volume<float> sigloss;
  float gammabar = 1.0 / (2.0*M_PI);  // when working with input in rad/s
  sigloss = calc_sigloss(lrgrad,te.value(),gammabar);
  save_volume(sigloss,siglossname.value());

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
    options.add(maskname);
    options.add(siglossname);
    options.add(te);
    options.add(slicedir);
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
