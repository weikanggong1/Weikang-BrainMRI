/*  rmsdiff.cc

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 1999-2000 University of Oxford  */

/*  CCOPYRIGHT  */


#include <string>
#include <iostream>
#include <unistd.h>

#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"

using namespace std;
using namespace MISCMATHS;
using namespace NEWMAT;
using namespace NEWIMAGE;


////////////////////////////////////////////////////////////////////////////

int main(int argc,char *argv[])
{

  float rmax=80.0;

  if (argc<4) {
    cerr << "Usage: " << argv[0] << " matrixfile1 matrixfile2 refvol [mask]" << endl;
    cerr << "        Outputs rms deviation between matrices (in mm)" << endl;
    return -1;
  }

  Matrix affmat1(4,4), affmat2(4,4);
  affmat1 = read_ascii_matrix(argv[1]);
  if (affmat1.Nrows()<4) {
    cerr << "Could not read matrix " << argv[1] << endl;
    return -2;
  }
  affmat2 = read_ascii_matrix(argv[2]);
  if (affmat2.Nrows()<4) {
    cerr << "Could not read matrix " << argv[2] << endl;
    return -2;
  }

  if (fabs(affmat1.Determinant())<0.1) {
    cerr << "WARNING:: matrix 1 has low determinant" << endl;
    cerr << affmat1 << endl;
  }

  if (fabs(affmat2.Determinant())<0.1) {
    cerr << "WARNING:: matrix 2 has low determinant" << endl;
    cerr << affmat2 << endl;
  }

  ColumnVector centre(3);
  centre = 0;

  volume<float> refvol;
  read_volume(refvol,argv[3]);

  if (argc<5) {
    // do the RMS
    // compute the centre of gravity
    centre = refvol.cog("scaled_mm");
    float rms = rms_deviation(affmat1,affmat2,centre,rmax);
    cout << rms << endl;
  } else {
    // do the extreme distance
    double maxdist=0, dist=0, sumdistsq=0;
    ColumnVector cvec(4);
    cvec=0;  cvec(4)=1;
    long int nvox=0;
    volume<float> mask;
    read_volume(mask,argv[4]);
    for (int z=mask.minz(); z<=mask.maxz(); z++) {
      for (int y=mask.miny(); y<=mask.maxy(); y++) {
	for (int x=mask.minx(); x<=mask.maxx(); x++) {
	  if (mask(x,y,z)>0.5) {
	    cvec(1)=x*refvol.xdim();  cvec(2)=y*refvol.ydim();  cvec(3)=z*refvol.zdim();
	    dist = norm2((affmat1 -affmat2)*cvec);
	    maxdist=Max(dist,maxdist);
	    sumdistsq+=dist*dist;
	    nvox++;
	  }
	}
      }
    }
    cout << maxdist << endl;
    double rms = sqrt(sumdistsq/nvox);
    cout << rms << endl;
  }

  return 0;

}
