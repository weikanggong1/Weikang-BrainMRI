/*  testcode.cc

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 1999-2000 University of Oxford  */

/*  CCOPYRIGHT  */

#include <string>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <time.h>
#include <vector>
#include <algorithm>
#define WANT_STREAM
#define WANT_MATH

#include "armawrap/newmatio.h"
//#include "miscimfns.h"
#include "miscmaths/miscmaths.h"
//#include "interpolation.h"
//#include "mjimage.h"
#include "newimage/costfns.h"
#include "newimage/newimageall.h"
#include "globaloptions.h"

#ifndef NO_NAMESPACE
 using namespace MISCMATHS;
 using namespace MISCIMFNS;
 using namespace COSTFNS;
 using namespace INTERPOLATION;
 using namespace NEWMAT;
 using namespace MJIMAGE;
 using namespace GENERALIO;
#endif



int testfn_yy(int argc, char *argv[]) {
  volume gfimage;
  if (argc > 1) {
    read_volume(gfimage,argv[1]);
  } else {
    read_volume(gfimage,"gfimage.hdr");
  }
	cout << "Read File Successfully!" << endl;
	cout << "Width: " << gfimage.rows() << "\tHeight: "
	     << gfimage.columns() << "\tDepth: " << gfimage.slices() << endl;

	//ZEnumPixelOf<float>	pixel(&gfimage);

	//for(pixel.Reset(); pixel.FMore(); pixel.Next()) *pixel *= -0.5;

	float max=0.0, min=0.0;
	get_min_max(gfimage,min,max);
	cout << "Min and Max = " << min << " & " << max << endl;

	for (float val=0.3; val>-0.005; val-=0.0014) {
	  for (int z=0; z<gfimage.zsize(); z++) {
	    for (int y=0; y<gfimage.ysize(); y++) {
	      for (int x=0; x<gfimage.xsize(); x++) {
		gfimage(x,y,z) *= (1.0+val);
	      }
	    }
	  }
	  cerr << ".";
	}

	get_min_max(gfimage,min,max);
	cout << "Min and Max = " << min << " & " << max << endl;


	save_volume(gfimage,"result");

  return -1;
}



int main(void)
{


  //  if (testfn_yy(argc,argv)<0) exit(-1);


  volume ref, test, bindex;
  read_volume(ref,"/usr/people/flitney/src/cr/ref");
  ref.setvoxelsize(1.0,1.0,1.0);
  ref.setvoxelorigin(0,0,0);
  read_volume(test,"/usr/people/flitney/src/cr/test");
  test.setvoxelsize(1.0,1.0,1.0);
  test.setvoxelorigin(0,0,0);
  read_volume(bindex,"/usr/people/flitney/src/cr/ref");
  bindex.setvoxelsize(1.0,1.0,1.0);
  bindex.setvoxelorigin(0,0,0);

  float theta=0.0, thecr=0.0;
  Matrix xfm;

  imagepair imp(ref,test);
  imp.set_no_bins(256);
  globaloptions::get().impair = &imp;

  for(theta = 0; theta < 360.0; theta += 4.0)
    {
      Matrix tr1, tr2, rot(4,4);
      ColumnVector angl(3), trans(3);
      trans = 0;
      angl = 0;
      angl(1) = theta*M_PI/180.0;

      make_rot(angl,trans,rot);

      tr1=IdentityMatrix(4);
      tr1(1,4) = -(test.xsize())/2.0;
      tr1(2,4) = -(test.ysize())/2.0;
      tr1(3,4) = -(test.zsize())/2.0;
      tr2=IdentityMatrix(4);
      tr2(1,4) = (test.xsize())/2.0;
      tr2(2,4) = (test.ysize())/2.0;
      tr2(3,4) = (test.zsize())/2.0;

      xfm = tr2 * rot * tr1;

      thecr = corr_ratio(globaloptions::get().impair, xfm);
      cerr << 1.0 - thecr << endl;
    }

  return 0;
}
