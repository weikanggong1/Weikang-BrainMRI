/*  makerot.cc

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2003 University of Oxford  */

/*  CCOPYRIGHT  */

// Skeleton application framework for using newimage

#include <vector>
#include <string>
#include <iostream>

#include "utils/options.h"
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"


using namespace std;
using namespace Utilities;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;


// The two strings below specify the title and example usage that is
//  printed out as the help or usage message

string title="makerot \nCopyright(c) 2003, University of Oxford (Mark Jenkinson)";
string examples="makerot [options] --theta=angle";

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
Option<float> theta(string("-t,--theta"), 0.0,
		  string("angle of rotation (in degrees)"),
		  true, requires_argument);
Option<std::vector<float> > axis(string("-a,--axis"), vector<float>(3),
		  string("~<ax,ay,az>\tunnormalised axis vector (comma separated)"),
		  false, requires_argument);
Option<std::vector<float> > centre(string("-c,--centre"), vector<float>(3),
		  string("~<cx,cy,cz>\tcentre of rotation in mm (comma separated)"),
		  false, requires_argument);
Option<string> covopt(string("--cov"), string(""),
		  string("image filename used for centre of volume"),
		  false, requires_argument);
Option<string> outname(string("-o,--out"), string(""),
		  string("output filename for matrix"),
		  false, requires_argument);
int nonoptarg;

////////////////////////////////////////////////////////////////////////////

// Local functions


int do_work(int argc, char* argv[])
{
  ColumnVector ctr(3), ax(3);
  if (axis.unset()) {
    ax(1)=0; ax(2)=0; ax(3)=1;
  } else {
    ax(1) = axis.value()[0]; ax(2) = axis.value()[1]; ax(3) = axis.value()[2];
  }
  if (norm2(ax)==0.0) {
    cerr << "ERROR:: Axis cannot be zero" << endl;
    return 1;
  } else {
    ax = ax / sqrt(norm2(ax));
  }

  if (centre.unset()) {
    ctr(1)=0; ctr(2)=0; ctr(3)=0;
  } else {
    ctr(1) = centre.value()[0]; ctr(2) = centre.value()[1]; ctr(3) = centre.value()[2];
  }

  if (covopt.set()) {
    volume<float> v1;
    read_volume(v1,covopt.value());
    ctr(1) = (v1.xsize() -1.0) * v1.xdim() * 0.5;
    ctr(2) = (v1.ysize() -1.0) * v1.ydim() * 0.5;
    ctr(3) = (v1.zsize() -1.0) * v1.zdim() * 0.5;
  }

  Matrix mat(4,4);
  make_rot(ax*theta.value()*M_PI/180.0, ctr, mat);

  if (outname.set()) {
    write_ascii_matrix(mat,outname.value());
  } else {
    cout << mat << endl;
  }

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
    options.add(theta);
    options.add(axis);
    options.add(covopt);
    options.add(centre);
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
