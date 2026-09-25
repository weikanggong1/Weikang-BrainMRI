/*  applyxfm4D.cc

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2000 University of Oxford  */

/*  CCOPYRIGHT  */

#include <string>
#include <iostream>

#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"
#include "newimage/fmribmain.h"

using namespace std;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;

// Globals - needed by fmrib_main

string oname, iname, transname, refname, usrinterp="sinc", matprefix="/MAT_0";
bool singlematrix, fourd, verbose;
interpolation interpmethod = sinc;
//////////////////////////////////////////////////////////////////////

template <class T>
int fmrib_main(int argc, char* argv[])
{
  if (fourd) {
    // 4D mode
    volume4D<T> invol, outvol;
    volume<T> refvol, dummy;
    read_volume4D(invol,iname);
    for (int t=0; t<invol.tsize(); t++) {
      invol[t].setpadvalue(invol[t].backgroundval());
    }
    if (verbose) {
      cout << "using interpolation method (enum, string): " << interpmethod << ", " << usrinterp << endl;
    }
    invol.setextrapolationmethod(extraslice);
    invol.setinterpolationmethod(interpmethod);
    if (interpmethod == sinc) {
        invol.definesincinterpolation("b",7);
    }

    // old form used a volume number
    //    refvol = invol[atoi(refname.c_str())];

    read_volume(refvol,refname);

    Matrix affmat(4,4);
    string matname;
    if (singlematrix) { affmat = read_ascii_matrix(transname); }

    if (invol.maxt() - invol.mint() > 10000) {
      cerr << "WARNING:: More than 10000 volumes - only doing first 10000" << endl;
    }

    for (int m=invol.mint(); m<=Min(invol.maxt(),invol.mint()+10000); m++) {

      if (!singlematrix) {
	matname = transname + matprefix;
	char nc='0';
	int n = m;
	matname += (nc + (char) (n / 1000));
	n -= (n/1000)*1000;
	matname += (nc + (char) (n / 100));
	n -= (n/100)*100;
	matname += (nc + (char) (n / 10));
	n -= (n/10)*10;
	matname += (nc + (char) n);
	cout << matname << endl;
	affmat = read_ascii_matrix(matname);
      }

      dummy = refvol;
      affine_transform(invol[m],dummy,affmat);
      outvol.addvolume(dummy);
    }
    outvol.settdim(invol.tdim());
    outvol.setDisplayMaximumMinimum(0,0);
    save_volume4D(outvol,oname);

  } else {
    // 3D mode
    volume<T> invol, outvol;

    read_volume(invol,iname);
    read_volume(outvol,refname);
    invol.setextrapolationmethod(extraslice);
    invol.setinterpolationmethod(interpmethod);
    if (interpmethod == sinc) {
        invol.definesincinterpolation("r",9);
    }

    Matrix affmat(4,4);
    affmat = read_ascii_matrix(transname);

    affine_transform(invol,outvol,affmat);
    outvol.settdim(invol.tdim());
    outvol.setDisplayMaximumMinimum(0,0);
    save_volume(outvol,oname);
  }

  return 0;
}

interpolation get_interptype(string usrinterp) {
    if (usrinterp.compare("nearestneighbour") == 0 || usrinterp.compare("nn") == 0) {
      return nearestneighbour;
    }
    else if (usrinterp.compare("trilinear") == 0) {
      return trilinear;
    }
    else if (usrinterp.compare("sinc") == 0) {
      return sinc;
    }
    else if (usrinterp.compare("spline") == 0) {
      return spline;
    }
    else {
      // if no matches
      cerr << "interpolation method: " << usrinterp << " is not a valid choice. Please choose from nearestneighbour, trilinear, sinc, or spline." << endl;
      exit(EXIT_FAILURE);
    }
}


int main(int argc,char *argv[])
{

  Tracer tr("main");
  if (argc<5) {
    cerr << "Usage: " << argv[0] << " <input volume> <ref volume>"
    << " <output volume> <transformation matrix file/dir>\n\n" <<
    "\t--interp, -interp <nearestneighbour (or nn), trilinear, spline, sinc (default)>\n" <<
    "\t--singlematrix, -singlematrix (flag option, do not provide an argument)\n" <<
    "\t--fourdigit, -fourdigit (flag option, do not provide an argument)\n" <<
    "\t--userprefix, -userprefix <prefix>\n" << endl;
    return -1;
  }

  // NB: a hidden option (-3D) exists (must appear after singlematrix)
  // parse the command line
  singlematrix = false;
  fourd = true;
  verbose = false;
  // first four arguments are postional and must be in the correct order
  iname = argv[1];
  refname = argv[2];
  oname = argv[3];
  transname = argv[4];
  for (int i = 5; i < argc; ++i){
    string option = argv[i];
    if (option == "-singlematrix" || option == "--singlematrix") {
        singlematrix = true;
    }
    else if (option == "-fourdigit" || option == "--fourdigit") {
        matprefix = "/MAT_";
    }
    else if (option == "-userprefix" || option == "--userprefix") {
        if (i++ <= argc) {
            string uprefix = argv[i];
            matprefix = "/" + uprefix;
        }
    }
    else if (option == "-interp" || option == "--interp") {
        if (i++ <= argc) {
            usrinterp = argv[i];
            interpmethod = get_interptype(usrinterp);
        }
    }
    else if (option == "-3D" || option == "--3D" || option == "-3d" || option == "--3d") {
        fourd = false;
    }
    else if (option == "--verbose" || option == "-verbose" || option == "-v") {
        verbose = true;
    }
  }

  // call the templated main
  return call_fmrib_main(dtype(iname),argc,argv);

}
