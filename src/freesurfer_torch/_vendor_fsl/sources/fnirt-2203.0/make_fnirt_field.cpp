/*  make_fnirt_field.cpp

    Jesper Andersson, FMRIB Image Analysis Group

    Copyright (C) 2021 University of Oxford  */

/*  CCOPYRIGHT  */

#include "utils/options.h"
#include "miscmaths/miscmaths.h"
#include "warpfns/warpfns.h"
#include "warpfns/fnirt_file_reader.h"
#include "warpfns/fnirt_file_writer.h"

// COMMAND LINE OPTIONS

using namespace Utilities;

string title="make_fnirt_field\nCopyright(c) 2021, University of Oxford (Jesper Andersson)";
string examples="make_fnirt_field --inwarp=warpvol --outwarp=newwarpvol --std=1";

Option<bool> verbose(string("-v,--verbose"), false, 
		     string("switch on diagnostic messages"), 
		     false, no_argument);
Option<bool> help(string("-h,--help"), false,
		  string("display this message"),
		  false, no_argument);
Option<bool> debug(string("--debug"), false,
		  string("turn on debugging output"),
		  false, no_argument);
Option<string> inwarp(string("-i,--inwarp"), string(""),
		      string("filename for existing fnirt warp file (--cout)"),
		      true, requires_argument);
Option<string> outwarp(string("-o,--outwarp"), string(""),
		       string("filename for output fnirt warp file"),
		       true, requires_argument);
Option<float> stdevp(string("-s,--std"), 1.0,
		     string("Standard deviation (in mm) of coefficents"),
		     true, requires_argument);
Option<float> fwhm(string("-f,--fwhm"), 1.0,
		   string("FWHM (in coefficients) of smoothing of coefficients"),
		   false, requires_argument);
Option<float> jmin(string("--jmin"), 0.1,
			string("minimum acceptable Jacobian value for constraint (default 0.1)"),
			false, requires_argument);
Option<float> jmax(string("--jmax"), 10.0,
			string("maximum acceptable Jacobian value for constraint (default 10.0)"),
			false, requires_argument);

int make_fnirt_field()
{
  if (verbose.value()) std::cout << "Reading input warps" << std::endl << std::flush;
  // Read existing fnirt field
  NEWIMAGE::FnirtFileReader infile(inwarp.value(),NEWIMAGE::UnknownWarps,verbose.value());

  // Get the field as splinefield
  if (verbose.value()) std::cout << "Making splinefield from input" << std::endl << std::flush;
  std::vector<std::shared_ptr<BASISFIELD::splinefield> > spfield(3);
  for (unsigned int i=0; i<3; i++) {
    BASISFIELD::splinefield test(infile.FieldAsSplinefield(i));
    spfield[i] = std::make_shared<BASISFIELD::splinefield>(infile.FieldAsSplinefield(i));
  }
  // Generate random vectors to set as spline coefficients
  if (verbose.value()) std::cout << "Setting coefficients to a random vector" << std::endl << std::flush;
  NEWIMAGE::volume<float> coef(spfield[0]->CoefSz_x(),spfield[0]->CoefSz_y(),spfield[0]->CoefSz_z());
  std::default_random_engine dre;
  std::normal_distribution<double> ndist(0.0,stdevp.value());
  for (unsigned int i=0; i<3; i++) {
    for (auto it=coef.nsfbegin(); it!=coef.nsfend(); ++it) *it = ndist(dre);
    if (fwhm.value() != 0.0) {
      coef = NEWIMAGE::smooth(coef,fwhm.value()/2.355);
    }
    spfield[i]->SetCoef(coef.vec());
  }
  
  // Check that it hasn't folded over itself
  if (verbose.value()) std::cout << "Making sure field is within prescribed Jacobian range" << std::endl;
  std::vector<unsigned int> imsz = infile.FieldSize();
  std::vector<double> vxsz = infile.VoxelSize();
  NEWIMAGE::volume<float> jac(imsz[0],imsz[1],imsz[2]);
  jac.setxdim(vxsz[0]); jac.setydim(vxsz[1]); jac.setzdim(vxsz[2]); 
  NEWIMAGE::deffield2jacobian(*(spfield[0]),*(spfield[1]),*(spfield[2]),jac);
  float maxj = 0.1, minj = 10.0;
  for (auto it=jac.nsfbegin(); it!=jac.nsfend(); ++it) {
    maxj = std::max(maxj,*it);
    minj = std::min(minj,*it);
  }
  if (minj < jmin.value()) {
    std::cout << "Smallest Jacobian = " << minj << ", smaller than the allowed " << jmin.value() << std::endl;
    exit(EXIT_FAILURE);
  }
  if (maxj > jmax.value()) {
    std::cout << "Largest Jacobian = " << maxj << ", larger than the allowed " << jmax.value() << std::endl;
    exit(EXIT_FAILURE);
  }

  // Write out field
  if (verbose.value()) std::cout << "Writing field out as fnirt coefficent file" << std::endl;
  NEWIMAGE::FnirtFileWriter wrt(outwarp.value(),*(spfield[0]),*(spfield[1]),*(spfield[2]));

  return(EXIT_SUCCESS);
}

int main(int argc, char *argv[])
{
  OptionParser options(title, examples);

  try {
    options.add(inwarp);
    options.add(outwarp);
    options.add(stdevp);
    options.add(fwhm);
    options.add(jmin);
    options.add(jmax);
    options.add(debug);
    options.add(verbose);
    options.add(help);
    
    int nparsed = options.parse_command_line(argc, argv);
    if (nparsed < argc) {
      for (; nparsed<argc; nparsed++) {
        cerr << "Unknown input: " << argv[nparsed] << endl;
      }
      exit(EXIT_FAILURE);
    }

    if ( (help.value()) || (!options.check_compulsory_arguments(true)) )
      {
	options.usage();
	exit(EXIT_FAILURE);
      }
  } catch(X_OptionError& e) {
    options.usage();
    cerr << endl << e.what() << endl;
    exit(EXIT_FAILURE);
  } catch(std::exception &e) {
    cerr << e.what() << endl;
  } 

  return make_fnirt_field();
}
