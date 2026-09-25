//  
//  Declarations for helper class dilator.h
//
//  dilator.h
//
//  Implements a very bare-bones class for doing "mean dilation"
//  for filling out undefined values in inverse fields with "average
//  of surrounding defined values". Intended for use with new_invwarp.
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2010 University of Oxford 
//
/*  CCOPYRIGHT  */

#ifndef dilator_h
#define dilator_h

class Dilator
{
public:
  Dilator(const NEWIMAGE::volume<float>& ima) : _ima(ima) {}
  unsigned int Dilate(float nan = -999);
  const NEWIMAGE::volume<float>& Get() const { return(_ima); }

private:
  NEWIMAGE::volume<float> _ima;
};

unsigned int Dilator::Dilate(float nan)
{
  NEWIMAGE::volume<float> tmp=_ima;
  unsigned int cnt=0;
  for (int k=0; k<_ima.zsize(); k++) {
    for (int j=0; j<_ima.ysize(); j++) {
      for (int i=0; i<_ima.xsize(); i++) {
        if (tmp(i,j,k)==nan) { // If it is a zero
          float val=0.0;
          float sum=0.0;
          unsigned int n=0;
          if (i>1 && (val=tmp.value(i-1,j,k))!=nan) { sum+=val; n++; }
          if (i<(_ima.xsize()-1) && (val=tmp.value(i+1,j,k))!=nan) { sum+=val; n++; }
          if (j>1 && (val=tmp.value(i,j-1,k))!=nan) { sum+=val; n++; }
          if (j<(_ima.ysize()-1) && (val=tmp.value(i,j+1,k))!=nan) { sum+=val; n++; }
          if (k>1 && (val=tmp.value(i,j,k-1))!=nan) { sum+=val; n++; }
          if (k<(_ima.zsize()-1) && (val=tmp.value(i,j,k+1))!=nan) { sum+=val; n++; }
          if (n) {
            _ima(i,j,k) = sum/float(n);
            cnt++;
          }
	}
      }
    }
  }
  return(cnt);
}

#endif // End #ifndef dilator_h
