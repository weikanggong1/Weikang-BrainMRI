/*  warpfns.h

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2001 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(__warpfns_h)
#define __warpfns_h


#ifndef EXPOSE_TREACHEROUS
#define I_DEFINED_ET
#define EXPOSE_TREACHEROUS           // To allow us to use .sampling_mat()
#endif

#include <cstdlib>
#include <ctime>
#include <string>
#include <iostream>
#include <vector>
#include <thread>

#include "armawrap/newmat.h"
#include "NewNifti/NewNifti.h"
#include "utils/threading.h"
#include "newimage/newimageall.h"
#include "basisfield/basisfield.h"

namespace RGT_UTILS { // raw_general_transform utilities

/// Used to ensure validity and self-consistency of input to raw_general_transform
template <class T>
std::tuple<bool,std::string> validate_input(const NEWMAT::Matrix&             A,
					    const NEWMAT::Matrix              *TT,
					    const NEWMAT::Matrix              *M,
					    const NEWIMAGE::volume4D<float>&  d,
					    const std::vector<int>&           defdir,
					    const std::vector<int>&           derivdir,
					    const std::vector<unsigned int>&  slices,
					    const NEWIMAGE::volume<T>&        out,
					    const NEWIMAGE::volume4D<T>&      deriv,
					    const NEWIMAGE::volume<char>      *valid);

template <class T>
std::tuple<NEWIMAGE::extrapolation,NEWIMAGE::extrapolation,
	   std::vector<bool> > set_extrapolation(const NEWIMAGE::volume<T>&       f,
						 const NEWIMAGE::volume4D<float>& d);

template <class T>
std::tuple<NEWMAT::Matrix,bool> make_iT(const NEWIMAGE::volume4D<float>& d,
					const NEWIMAGE::volume<T>&       out,
					const NEWMAT::Matrix             *TT);

/// Returns a vector where v[i] and v[i+1] denotes the first and
/// (on past the) last row to be processed by the ith thread
std::vector<unsigned int> rows_per_thread(unsigned int nrows,
					  unsigned int nthr);

template <class T>
void affine_no_derivs(// Input
		      unsigned int                     first_j,
		      unsigned int                     last_j,
		      const NEWIMAGE::volume<T>&       f,
		      const std::vector<unsigned int>& slices,
		      const NEWMAT::Matrix&            A,
		      // Output
		      NEWIMAGE::volume<T>&             out,
		      NEWIMAGE::volume<char>           *valid);

template <class T>
void affine_with_derivs(// Input
			unsigned int                     first_j,
			unsigned int                     last_j,
			const NEWIMAGE::volume<T>&       f,
			const std::vector<unsigned int>& slices,
			const NEWMAT::Matrix&            A,
			const std::vector<int>&          derivdir,
			// Output
			NEWIMAGE::volume<T>&             out,
			NEWIMAGE::volume4D<T>&           deriv,
			NEWIMAGE::volume<char>           *valid);

template <class T>
void displacements_no_iT(// Input
			 unsigned int                     first_j,
			 unsigned int                     last_j,
			 const NEWIMAGE::volume<T>&       f,
			 const std::vector<unsigned int>& slices,
			 const NEWMAT::Matrix&            A,
			 const NEWIMAGE::volume4D<float>& d,
			 const std::vector<int>&          defdir,
			 const NEWMAT::Matrix&            M,
			 const std::vector<int>&          derivdir,
			 // Output
			 NEWIMAGE::volume<T>&             out,
			 NEWIMAGE::volume<T>&             deriv,
			 NEWIMAGE::volume<char>           *valid);

template <class T>
void displacements_with_iT(// Input
			   unsigned int                     first_j,
			   unsigned int                     last_j,
			   const NEWIMAGE::volume<T>&       f,
			   const std::vector<unsigned int>& slices,
			   const NEWMAT::Matrix&            iT,
			   const NEWMAT::Matrix&            A,
			   const NEWIMAGE::volume4D<float>& d,
			   const std::vector<int>&          defdir,
			   const NEWMAT::Matrix&            M,
			   const std::vector<int>&          derivdir,
			   // Output
			   NEWIMAGE::volume<T>&             out,
			   NEWIMAGE::volume<T>&             deriv,
			   NEWIMAGE::volume<char>           *valid);

template <class T>
void set_sqform(// Input
		const NEWIMAGE::volume<T>&        f,
		const NEWIMAGE::volume4D<float>&  d,
		const std::vector<int>&           defdir,
		NEWMAT::Matrix                    iA,      // Copy is intentional
		const NEWMAT::Matrix              *TT,
		const NEWMAT::Matrix              *M,
		// Output
		NEWIMAGE::volume<T>&              out,
		NEWIMAGE::volume4D<T>&            deriv);

} // End namespace RGT_UTILS

namespace NEWIMAGE {

class WarpFnsException: public std::exception
{
private:
  std::string m_msg;
public:
  WarpFnsException(const std::string& msg) noexcept: m_msg(std::string("warpfns::") + msg) {}
  virtual const char * what() const noexcept { return(m_msg.c_str()); }
  ~WarpFnsException() noexcept {}
};


  //
  // Here stars declarations of functions inherited from initial
  // version of warpfns.h.
  //
  int affine2warp(const NEWMAT::Matrix& affmat,
                  volume4D<float>&      warpvol,
                  const volume<float>&  outvol);

  int shift2warp(const volume<float>& shiftmap,
                 volume4D<float>&     warp,
                 const std::string&   shiftdir);

  int convertwarp_rel2abs(volume4D<float>& warpvol);
  int convertwarp_abs2rel(volume4D<float>& warpvol);

  int concat_warps(const volume4D<float>& prewarp,
                   const volume4D<float>& postwarp,
                   volume4D<float>&       totalwarp);

  // default value for gammabar is for rad/s units; te in seconds;
  //   lrgrad in rad/s/voxel
  volume<float> calc_sigloss(volume4D<float>& lrgrad, float te,
			     float gammabar=0.5/M_PI);


  // try to determine if the warp is stored in absolute (vs relative) convention
  bool is_abs_convention(const volume4D<float>& warpvol);

  void jacobian_check(volume4D<float>&      jvol,
                      NEWMAT::ColumnVector& jacobian_stats,
                      const volume4D<float>& warp,
                      float minJ, float maxJ, bool use_vol=true);

  volume4D<float> jacobian_check(NEWMAT::ColumnVector& jacobian_stats,
                                 const volume4D<float>& warp,
                                 float minJ, float maxJ);

  NEWMAT::ColumnVector jacobian_quick_check(const volume4D<float>& warp,
                                            float minJ, float maxJ);


  void constrain_topology(volume4D<float>& warp, float minJ, float maxJ);

  void constrain_topology(volume4D<float>& warp);

//////////////////////////////////////////////////////////////////////////
//
// Here starts declarations of coordinate-transform functions
// that will be defined below.
//
// The general format of the NewimageCoord2NewimageCoord (here
// abbreviated to N2N) is
//
// trgt_coord = N2N(some_transforms,src_vol,trgt_vol,src_coord)
//
// The purpose of the routines is to supply a voxel-coordinate in one
// space (e.g. example_func) and obtain the corresponding voxel-
// coordinate in another space (e.g. standard space).
//
// The set of transforms that are indicated in "some_transforms" above
// Are as follows. Imagine we have four spaces a, b, c and d and that we
// have a linear transform M1 mapping a onto b, a non-linear transform
// w mapping c onto b (N.B. the order here) and a linear transform M2
// mapping c onto d.
//
// -------       -------       -------       -------
// |     |   M1  |     |   w   |     |   M2  |     |
// |  a  |------>|  b  |<------|  c  |------>|  d  |
// |     |  lin  |     | non-  |     |  lin  |     |
// -------       ------- lin   -------       -------
//
// A common example would be that a=example_func, b=highres
// c=standard, M1=example_func2highres.mat and w=highres2standard_warp.nii.gz
// In this example there is nothing corresponding to D or M2.
// Note also that the affine part of the mapping between b and c (i.e.
// highres2standard.mat) is incorporated into w.
//
// Let us now say we have a coordinate xf in the space of example_func (a),
// and we want to map that to xs in standard space (c). We would then use a call
// that can be schematically described as
//
// xs = N2N(Matrix&             M1 = example_func2highres.mat,
//          volume4D<float>&    warps = highres2standard_warp.nii.gz,
//          bool                invert_warps = true
//          volume<D>&          src = example_func.nii.gz,
//          volume<S>&          dest = standard.nii.gz
//          ColumnVector&       xf)
//
// The important points to realise here is that first we use M1 to get
// from a->b, hence we pass M1 in first. Secondly we want to go from b->c,
// so we pass in w. *BUT* w maps c->b, which is why we have set the
// invert_warps flag.
//
// Let us now assume we have a coordinate xs in standard space (c) and that
// we want to map it to a coordinate xf in functional space (a). We would
// then use a call like
//
// xf = N2N(volume4D<float>&    warps = highres2standard_warp.nii.gz,
//          bool                invert_warps = false,
//          Matrix&             M1 = inverse(example_func2highres.mat,
//          volume<D>&          src = standard.nii.gz,
//          volume<S>&          dest = example_func.nii.gz,
//          ColumnVector&       xs)
//
// The points to note here are that we first map from c->b, so we
// pass in w first. And this time w goes in the right direction
// so we set the invert_warps flag to false. After that we go from
// b->a, so we pass in THE INVERSE OF M1 to take us that step.
//
// These examples should hopefully explain the thoughts behind the
// various overloaded versions of N2N below. But before I stop I will
// also give you the same example in "proper code".
//
// volume<float>    funcvol; read_volume(funcvol,"example_func");
// volume<float>    stdvol; read_volume(stdvol,"standard");
// Matrix           M1 = read_ascii_matrix("example_func2highres.mat");
// FnirtFileReader  reader("highres2standard_warp");
//
// ColumnVector     xf(3);
// ColumnVector     xs(3);
//
// xf << 36 << 42 << 23;    // Coordinate 36,42,23 in example_func
// xs = NewimageCoord2NewimageCoord(M1,reader.FieldAsNewimageVolume4D(true),
//                                  true,funcvol,stdvol,xf);
//
//
// xs << 63 << 69 << 41;    // Coordinate 63,69,41 in standard
// xf = NewimageCoord2NewimageCoord(reader.FieldAsNewimageVolume4D(true),
//                                  false,M1.i(),stdvol,funcvol,xs);
//
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//
//  Declarations
//
//////////////////////////////////////////////////////////////////////////

//
// For use e.g. when using affine mapping highres->standard
//
template <class D, class S>
NEWMAT::ReturnMatrix NewimageCoord2NewimageCoord(const NEWMAT::Matrix&        M,
                                                 const volume<D>&             srcvol,
                                                 const volume<S>&             destvol,
                                                 const NEWMAT::ColumnVector&  srccoord);

//
// For use e.g. when transforming (non-linearly) between highres and standard
//
template <class D, class S>
NEWMAT::ReturnMatrix NewimageCoord2NewimageCoord(const volume4D<float>&       warps,
                                                 bool                         inv_flag,
                                                 const volume<D>&             srcvol,
                                                 const volume<S>&             destvol,
                                                 const NEWMAT::ColumnVector&  srccoord);

//
// For use e.g. when transforming from standard space to functional space
//
template <class D, class S>
NEWMAT::ReturnMatrix NewimageCoord2NewimageCoord(const volume4D<float>&       warps,
                                                 bool                         inv_flag,
                                                 const NEWMAT::Matrix&        M,
                                                 const volume<D>&             srcvol,
                                                 const volume<S>&             destvol,
                                                 const NEWMAT::ColumnVector&  srccoord);

//
// For use e.g. when transforming from functional space to standard space
//
template <class D, class S>
NEWMAT::ReturnMatrix NewimageCoord2NewimageCoord(const NEWMAT::Matrix&        M,
                                                 const volume4D<float>&       warps,
                                                 bool                         inv_flag,
                                                 const volume<D>&             srcvol,
                                                 const volume<S>&             destvol,
                                                 const NEWMAT::ColumnVector&  srccoord);

//
// For use when doing the whole shabang. Whenever that might be.
//
template <class D, class S>
NEWMAT::ReturnMatrix NewimageCoord2NewimageCoord(const NEWMAT::Matrix&        M1,
                                                 const volume4D<float>&       warps,
                                                 bool                         inv_flag,
                                                 const NEWMAT::Matrix&        M2,
                                                 const volume<D>&             srcvol,
                                                 const volume<S>&             destvol,
                                                 const NEWMAT::ColumnVector&  srccoord);

//
// Internal function providing functionality for
// all the overloaded functions above.
//
template <class D, class S>
int raw_newimagecoord2newimagecoord(const NEWMAT::Matrix         *M1,
                                    const volume4D<float>        *warps,
                                    bool                         inv_flag,
                                    const NEWMAT::Matrix         *M2,
                                    const volume<D>&             src,
                                    const volume<S>&             trgt,
                                    NEWMAT::ColumnVector&        coord);

//
// Function to calculate the inverse lookup for a single coordinate
// Uses newimage voxel coordinates everywhere and takes relative, mm warps
//
template <class T>
NEWMAT::ColumnVector inv_coord(const volume4D<float>&      warp,
                               const volume<T>&            srcvol,
                               const NEWMAT::ColumnVector& coord);


//////////////////////////////////////////////////////////////////////////
//
//  Definitions
//
//////////////////////////////////////////////////////////////////////////

//
// For use e.g. when using affine mapping highres->standard
//
template <class D, class S>
NEWMAT::ReturnMatrix NewimageCoord2NewimageCoord(const NEWMAT::Matrix&        M,
                                                 const volume<D>&             srcvol,
                                                 const volume<S>&             destvol,
                                                 const NEWMAT::ColumnVector&  srccoord)
{
  // Check that matrix is kosher
  if (M.Nrows()!=4 || M.Ncols()!=4) imthrow("NewimageCoord2NewimageCoord: M must be a 4x4 matrix",11);
  // Allocate output coordinates
  NEWMAT::ColumnVector   tmp(4);
  tmp.Rows(1,3) = srccoord.Rows(1,3);
  tmp(4) = 1.0;
  // Do the conversion
  raw_newimagecoord2newimagecoord(&M,0,false,0,srcvol,destvol,tmp);
  // Return coordinate-vector of same size as incord
  NEWMAT::ColumnVector  destcord(srccoord);
  destcord.Rows(1,3) = tmp.Rows(1,3);
  destcord.Release();
  return(destcord);
}

//
// For use e.g. when transforming (non-linearly) between highres and standard
//
template <class D, class S>
NEWMAT::ReturnMatrix NewimageCoord2NewimageCoord(const volume4D<float>&       warps,
                                                 bool                         inv_flag,
                                                 const volume<D>&             srcvol,
                                                 const volume<S>&             destvol,
                                                 const NEWMAT::ColumnVector&  srccoord)
{
  // Check and prep warp-field
  if (warps.tsize() != 3) imthrow("NewimageCoord2NewimageCoord: warps must be a 4D volume with three points in fourth dimension",11);
  if (warps.nvoxels() <= 0) imthrow("NewimageCoord2NewimageCoord: warps must have a non-zero size",11);
  // Allocate output coordinates
  NEWMAT::ColumnVector   tmp(4);
  tmp.Rows(1,3) = srccoord.Rows(1,3);
  tmp(4) = 1.0;
  // Do the conversion
  raw_newimagecoord2newimagecoord(0,&warps,inv_flag,0,srcvol,destvol,tmp);
  // Return coordinate-vector of same size as incord
  NEWMAT::ColumnVector  destcord(srccoord);
  destcord.Rows(1,3) = tmp.Rows(1,3);
  destcord.Release();
  return(destcord);
}

//
// For use e.g. when transforming from standard space to functional space
//
template <class D, class S>
NEWMAT::ReturnMatrix NewimageCoord2NewimageCoord(const volume4D<float>&       warps,
                                                 bool                         inv_flag,
                                                 const NEWMAT::Matrix&        M,
                                                 const volume<D>&             srcvol,
                                                 const volume<S>&             destvol,
                                                 const NEWMAT::ColumnVector&  srccoord)
{
  // Check that matrix is kosher
  if (M.Nrows()!=4 || M.Ncols()!=4) imthrow("NewimageCoord2NewimageCoord: M must be a 4x4 matrix",11);
  // Check and prep warp-field
  if (warps.tsize() != 3) imthrow("NewimageCoord2NewimageCoord: warps must be a 4D volume with three points in fourth dimension",11);
  if (warps.nvoxels() <= 0) imthrow("NewimageCoord2NewimageCoord: warps must have a non-zero size",11);
  // Allocate output coordinates
  NEWMAT::ColumnVector   tmp(4);
  tmp.Rows(1,3) = srccoord.Rows(1,3);
  tmp(4) = 1.0;
  // Do the conversion
  raw_newimagecoord2newimagecoord(0,&warps,inv_flag,&M,srcvol,destvol,tmp);
  // Return coordinate-vector of same size as incord
  NEWMAT::ColumnVector  destcord(srccoord);
  destcord.Rows(1,3) = tmp.Rows(1,3);
  destcord.Release();
  return(destcord);
}

//
// For use e.g. when transforming from functional space to standard space
//
template <class D, class S>
NEWMAT::ReturnMatrix NewimageCoord2NewimageCoord(const NEWMAT::Matrix&        M,
                                                 const volume4D<float>&       warps,
                                                 bool                         inv_flag,
                                                 const volume<D>&             srcvol,
                                                 const volume<S>&             destvol,
                                                 const NEWMAT::ColumnVector&  srccoord)
{
  // Check that matrix is kosher
  if (M.Nrows()!=4 || M.Ncols()!=4) imthrow("NewimageCoord2NewimageCoord: M must be a 4x4 matrix",11);
  // Check and prep warp-field
  if (warps.tsize() != 3) imthrow("NewimageCoord2NewimageCoord: warps must be a 4D volume with three points in fourth dimension",11);
  if (warps.nvoxels() <= 0) imthrow("NewimageCoord2NewimageCoord: warps must have a non-zero size",11);
  // Allocate output coordinates
  NEWMAT::ColumnVector   tmp(4);
  tmp.Rows(1,3) = srccoord.Rows(1,3);
  tmp(4) = 1.0;
  // Do the conversion
  raw_newimagecoord2newimagecoord(&M,&warps,inv_flag,0,srcvol,destvol,tmp);
  // Return coordinate-vector of same size as incord
  NEWMAT::ColumnVector  destcord(srccoord);
  destcord.Rows(1,3) = tmp.Rows(1,3);
  destcord.Release();
  return(destcord);
}

//
// For use when doing the whole shabang. Whenever that might be.
//
template <class D, class S>
NEWMAT::ReturnMatrix NewimageCoord2NewimageCoord(const NEWMAT::Matrix&        M1,
                                                 const volume4D<float>&       warps,
                                                 bool                         inv_flag,
                                                 const NEWMAT::Matrix&        M2,
                                                 const volume<D>&             srcvol,
                                                 const volume<S>&             destvol,
                                                 const NEWMAT::ColumnVector&  srccoord)
{
  // Check that matrices are kosher
  if (M1.Nrows()!=4 || M1.Ncols()!=4) imthrow("NewimageCoord2NewimageCoord: M1 must be a 4x4 matrix",11);
  if (M2.Nrows()!=4 || M2.Ncols()!=4) imthrow("NewimageCoord2NewimageCoord: M2 must be a 4x4 matrix",11);
  // Check and prep warp-field
  if (warps.tsize() != 3) imthrow("NewimageCoord2NewimageCoord: warps must be a 4D volume with three points in fourth dimension",11);
  if (warps.nvoxels() <= 0) imthrow("NewimageCoord2NewimageCoord: warps must have a non-zero size",11);
  // Allocate output coordinates
  NEWMAT::ColumnVector   tmp(4);
  tmp.Rows(1,3) = srccoord.Rows(1,3);
  tmp(4) = 1.0;
  // Do the conversion
  raw_newimagecoord2newimagecoord(&M1,&warps,inv_flag,&M2,srcvol,destvol,tmp);
  // Return coordinate-vector of same size as incord
  NEWMAT::ColumnVector  destcord(srccoord);
  destcord.Rows(1,3) = tmp.Rows(1,3);
  destcord.Release();
  return(destcord);
}

template <class D, class S>
int raw_newimagecoord2newimagecoord(const NEWMAT::Matrix         *M1,
                                    const volume4D<float>        *warps,
                                    bool                         inv_flag,
                                    const NEWMAT::Matrix         *M2,
                                    const volume<D>&             src,
                                    const volume<S>&             trgt,
                                    NEWMAT::ColumnVector&        coord)
{
  int  rval = 1;
  //
  // First go from voxel-coordinate in src to mm-space of warps
  //
  coord = src.sampling_mat()*coord;
  if (M1) coord = (*M1)*coord;
  //
  // Then find displacement-vector at that point
  //
  if (warps) {
    if (inv_flag) {
      // Here we will use a fake volume to take us back and
      // forth between mm and newimage coordinates. This is
      // just so that we shall be able to pass voxel-coordinates
      // into inv_coord even though here we are really in mm-space.
      volume<float>  fake_vol = (*warps)[0];
      coord = fake_vol.sampling_mat().i() * coord;  // mm->vox in fake-space
      coord = inv_coord(*warps,fake_vol,coord);     // vox_in_fake->vox_in_ref_of_warps
      coord = warps->sampling_mat() * coord;        // vox->mm in ref_of_warps-space
    }
    else {
      NEWMAT::ColumnVector  vd_coord = warps->sampling_mat().i() * coord;
      extrapolation oldex = warps->getextrapolationmethod();
      if (oldex!=periodic) warps->setextrapolationmethod(extraslice);
      NEWMAT::ColumnVector  dvec(4);
      dvec = 0.0;
      dvec(1) = (*warps)[0].interpolate(vd_coord(1),vd_coord(2),vd_coord(3));
      dvec(2) = (*warps)[1].interpolate(vd_coord(1),vd_coord(2),vd_coord(3));
      dvec(3) = (*warps)[2].interpolate(vd_coord(1),vd_coord(2),vd_coord(3));
      coord += dvec;
      rval = (warps->in_bounds(float(vd_coord(1)),float(vd_coord(2)),float(vd_coord(3)))) ? 1 : -1;   // Indicate invalid warp-value
      warps->setextrapolationmethod(oldex);
    }
  }
  //
  // Finally go to voxel-space of trgt
  //
  if (M2) coord = (*M2)*coord;
  coord = trgt.sampling_mat().i()*coord;

  return(rval);
}


template <class T>
NEWMAT::ColumnVector inv_coord(const volume4D<float>&      warp,
                               const volume<T>&            srcvol,
                               const NEWMAT::ColumnVector& coord)
{
  // coord is in source (x1) space - same as srcvol
  // need the srcvol to work out the voxel<->mm conversion for the x1 space
  int N=5;
  int N_2=(int)(N/2);

  NEWMAT::ColumnVector coord4(4);
  coord4 << coord(1) << coord(2) << coord(3) << 1.0;

  volume<float> idx(N,N,N);  // NB: different coordinate conventions to matlab
  idx=0.0f;
  int c=1;
  for (int z=0; z<N; z++) {
    for (int y=0; y<N; y++) {
      for (int x=0; x<N; x++) {
	idx(x,y,z)=c++;
      }
    }
  }

  // Form matrix M which stores trilinear coefficients needed to interpolate inverse warpfield
  // The coordlist is the list of x2 coordinates mapping into the neighbourhood (in voxel coords)
  NEWMAT::Matrix M;
  NEWMAT::Matrix coordlist;
  float dx,dy,dz;
  float dist, mindist=MISCMATHS::Max(srcvol.maxx(),MISCMATHS::Max(srcvol.maxy(),srcvol.maxz()));
  mindist *= mindist;  // conservative estimate
  float minptx=-1, minpty=-1, minptz=-1;
  // loop around all voxels in ref space: x2
  for (int z2=0; z2<=warp.maxz(); z2++) {
    for (int y2=0; y2<=warp.maxy(); y2++) {
      for (int x2=0; x2<=warp.maxx(); x2++) {
	// if warp points towards voxel in the neighbour of the target coord
	dx=(warp(x2,y2,z2,0)+x2*warp.xdim())/srcvol.xdim()-coord4(1);
	dy=(warp(x2,y2,z2,1)+y2*warp.ydim())/srcvol.ydim()-coord4(2);
	dz=(warp(x2,y2,z2,2)+z2*warp.zdim())/srcvol.zdim()-coord4(3);
	dist=dx*dx+dy*dy+dz*dz;  // in voxels - might be better in mm?
	if (dist<mindist) { mindist=dist; minptx=x2; minpty=y2; minptz=z2; }
	if ((std::fabs(dx)<N_2) && (std::fabs(dy)<N_2) && (std::fabs(dz)<N_2)) {
	  MISCMATHS::addrow(M,N*N*N);
	  // add current voxel coord (x2) to coordlist
      MISCMATHS::addrow(coordlist,3);
	  coordlist(coordlist.Nrows(),1)=x2;
	  coordlist(coordlist.Nrows(),2)=y2;
	  coordlist(coordlist.Nrows(),3)=z2;
	  // loop around all voxels in target coords' neighbourhood: x1
	  for (int z1=-N_2; z1<=N_2; z1++) {
	    for (int y1=-N_2; y1<=N_2; y1++) {
	      for (int x1=-N_2; x1<=N_2; x1++) {
		if ((std::fabs(dx-x1)<1.0) && (std::fabs(dy-y1)<1.0) && (std::fabs(dz-z1)<1.0)) {
		  //   M(newrow,idx(x1)) = (1-fabs(dx))*(1-fabs(dy))*(1-fabs(dz))
		  M(M.Nrows(),MISCMATHS::round(idx(x1+N_2,y1+N_2,z1+N_2))) = (1.0-std::fabs(dx-x1))*(1.0-std::fabs(dy-y1))*(1.0-std::fabs(dz-z1));
		}
	      }
	    }
	  }
	}
      }
    }
  }

  int ncols=N*N*N;
  NEWMAT::Matrix L;
  // form regularisation matrix (L) - isotropic assumption for now - probably unimportant
  for (int z=0; z<N; z++) {
    for (int y=0; y<N; y++) {
      for (int x=0; x<N; x++) {
	if ((x>0) && (x<N-1)) {
	  MISCMATHS::addrow(L,ncols);
	  L(L.Nrows(),MISCMATHS::round(idx(x,y,z)))=2;
	  L(L.Nrows(),MISCMATHS::round(idx(x-1,y,z)))=-1;
	  L(L.Nrows(),MISCMATHS::round(idx(x+1,y,z)))=-1;
	}
	if ((y>0) && (y<N-1)) {
	  MISCMATHS::addrow(L,ncols);
	  L(L.Nrows(),MISCMATHS::round(idx(x,y,z)))=2;
	  L(L.Nrows(),MISCMATHS::round(idx(x,y-1,z)))=-1;
	  L(L.Nrows(),MISCMATHS::round(idx(x,y+1,z)))=-1;
	}
	if ((z>0) && (z<N-1)) {
	  MISCMATHS::addrow(L,ncols);
	  L(L.Nrows(),MISCMATHS::round(idx(x,y,z)))=2;
	  L(L.Nrows(),MISCMATHS::round(idx(x,y,z-1)))=-1;
	  L(L.Nrows(),MISCMATHS::round(idx(x,y,z+1)))=-1;
	}
      }
    }
  }

  // now calculate new coordinate (in voxel coords)
  NEWMAT::ColumnVector newcoord(4);
  if (M.Nrows()>8) {
    // normalise both M and L (on a per-voxel basis)
    M *= 1;
    //L *= 1*M.Nrows()/L.Nrows();
    float lambda=0.5;   // this is the default in invwarp.cc
    NEWMAT::ColumnVector coordsx, coordsy, coordsz;
    NEWMAT::CroutMatrix X = M.t()*M + lambda*L.t()*L;  // carries out LU decomposition - for efficiency
    coordsx = X.i()*M.t()*coordlist.SubMatrix(1,coordlist.Nrows(),1,1);
    coordsy = X.i()*M.t()*coordlist.SubMatrix(1,coordlist.Nrows(),2,2);
    coordsz = X.i()*M.t()*coordlist.SubMatrix(1,coordlist.Nrows(),3,3);
    newcoord << coordsx(MISCMATHS::round(idx(N_2,N_2,N_2)))
             << coordsy(MISCMATHS::round(idx(N_2,N_2,N_2)))
             << coordsz(MISCMATHS::round(idx(N_2,N_2,N_2))) << 1.0;
  } else {
    // just copy the nearest value of relative warp (wherever it was) and add that to the existing voxel
    //  this should deal with any constant translation and maybe even rotation...
    newcoord << (warp[0].interpolate(minptx,minpty,minptz) + coord4(1)*srcvol.xdim())/warp.xdim()
             << (warp[1].interpolate(minptx,minpty,minptz) + coord4(2)*srcvol.ydim())/warp.ydim()
             << (warp[2].interpolate(minptx,minpty,minptz) + coord4(3)*srcvol.zdim())/warp.zdim() << 1.0;
  }

  if (coord.Nrows()==3) {
    NEWMAT::ColumnVector nc3(3);  nc3 << newcoord(1) << newcoord(2) << newcoord(3);  newcoord=nc3;
  }
  return newcoord;
}

  ///////////////////////////////////////////////////////////////////////////
  // IMAGE PROCESSING ROUTINES
  ///////////////////////////////////////////////////////////////////////////

  // General Transform
  //
  // The routine "raw_general_transform" is the heart of the "warping"
  // functions. It provides functionality for calculating warped images
  // and partial derivatives in warped space for use by routines for
  // non-linear registration as well as distortion correction. In addition
  // it is also used for final resampling of images given a pre-determined
  // displacement field.
  //
  // In the most general case we might have registered some volume i to a
  // template s that already had an affine transformation matrix A mapping
  // i onto s. The non-linear mapping of i onto s is given by a displacement
  // field d. We may in addition also have a volume f that maps linearly onto
  // the volume in through a linear transform M. A typical example would be
  // that s is e.g. the avg152 (implementing the MNI space), i is a structural
  // from some subject and f is a functional volume from that same subject.
  // A is a matrix generated by flirt mapping i onto s, M is another matrix
  // generated by flirt mapping the functional onto the structural and d is
  // a displacement field calculated by fnirt.
  // Finally there is another volume out, which defines the space to which we
  // ultimately want to resample f (or i, if there is no f). There is an
  // affine transform T that maps s onto out. An example of out might be the
  // Talairach space and T might be a "known" matrix that effects an approximate
  // mapping MNI->Talairach. Another example of out might simply be a volume in
  // the space of s, but with a different voxel/matrix size. For example if one
  // wants to use a template with a 1mm isotropic resolution (for high resolution
  // nonlinear registration) but wants to resample the functional data to a
  // volume with 2mm resolution (because 1mm might be a little over the top
  // for functional data).
  //
  // In the code I have retained the notation I have sketched above. To recap
  //
  // volume<T>       f   // "Final" volume in chain. The volume that we want to map some
  //                     // cordinate x_out (in space of out) into so that we can can interpolate
  //                     // intensity values from s and write into out
  // Matrix          A;  // Affine mapping of i onto s
  // volume4D<float> d;  // Non-linear mapping of i onto s
  // volume<T>       s;  // Volume used as template. A coordinate x_i in volume i is related
  //                     // to a coordinate x_s in s as x_i = inv(B_i)*inv(A)*B_s*x_s + inv(B_i)*d(x_s)
  //                     // where B_i and B_s are voxel->mm transforms for i and s respectively.
  // volume<T>       out // Volume into which we ultimately want to map f (or i, if no f)
  // Matrix          T   // Affine (or subset of) mapping of t onto s
  // Matrix          M   // Affine (typically rigid sub-set of) mapping of in onto s
  //
  // Together with some other options (such as derivatives or no derivatives, 3D or 1D non-linear
  // transform etc) this all makes the calling interface a little messy. There is therefore a set
  // of alternative calls with a reduced interface that will be more convenient for many
  // applications.
  //

/////////////////////////////////////////////////////////////////////
//
// This is as raw as it gets, with pointers and all
//
/////////////////////////////////////////////////////////////////////


template <class T>
void raw_general_transform(// Input
			   const volume<T>&          f,        // Input volume
			   const NEWMAT::Matrix&     A,        // 4x4 affine transformation matrix
			   const volume4D<float>&    d,        // Displacement fields (also defines space of t).
			   // Note that these are "relative" fields in units of mm.
			   const std::vector<int>&   defdir,   // Directions of displacements.
			   const std::vector<int>&   derivdir, // Directions of derivatives
			   std::vector<unsigned int> slices,   // Vector of slices (in out) that should be resampled (N.B. copy is intentional)
			   const NEWMAT::Matrix      *TT,      // Mapping of out onto t
			   const NEWMAT::Matrix      *M,       // Mapping of in onto s
			   // Output
			   volume<T>&                out,      // Output volume
			   volume4D<T>&              deriv,    // Partial derivatives. Note that the derivatives are in units "per voxel"
			   volume<char>              *valid,   // Mask indicating what voxels fell inside original fov
			   // Optional input
			   Utilities::NoOfThreads    nthr=Utilities::NoOfThreads(1)) // No. of threads. N.B. threading in y-direction
{
  // Validate input
  auto [valinp,msg] = RGT_UTILS::validate_input(A,TT,M,d,defdir,derivdir,slices,out,deriv,valid);
  if (!valinp) throw WarpFnsException("NEWIMAGE::raw_general_transform: "+msg);

  // Assume we should resample all slices if slices vector is empty
  if (slices.size()==0) { slices.resize(out.zsize()); std::iota(slices.begin(),slices.end(),0); }

  // Save old extrapolation settings and set new
  auto [oldex,d_oldex,d_old_epvalidity] = RGT_UTILS::set_extrapolation(f,d);

  // Create a matrix iT mapping from voxel coordinates in volume out to voxel-coordinates in volume s
  // (same space as d).
  auto [iT,useiT] = RGT_UTILS::make_iT(d,out,TT);

  // Create a matrix iA mapping from voxel coordinates in volume t to
  // mm-coordinates in volume i.

  NEWMAT::Matrix iA = A.i();
  // if (f.left_right_order()==FSL_NEUROLOGICAL) {iA = f.swapmat(-1,2,3) * iA;}      // Swap if input neurological
  // if (out.left_right_order()==FSL_NEUROLOGICAL) {iA = iA * out.swapmat(-1,2,3);}  // Swap if output neurological
  if (defdir.size()) iA = iA * d[0].sampling_mat();  // If we have a displacement field
  else iA = iA * out.sampling_mat();                 // Else

  // Create a matrix mapping from mm-coordinates in volume i
  // to voxel coordinates in volume f. If the matrix M is empty
  // this is simply a mm->voxel mapping for volume f

  NEWMAT::Matrix iM(4,4);
  if (M) iM = f.sampling_mat().i() * M->i();
  else iM = f.sampling_mat().i();

  // Do the actual resampling. The code has been divided into four funcions
  // depending on if it is an affine transform, or if a displacement field
  // was supplied. In the latter case it is divided into the case where the
  // target volume is in the same space as the displacement field or not.
  //
  // It is parallelised using the >C++11 thread library. In the single thread
  // case only the final (non-threaded) call is made. The parallelisation is
  // done in the y-direction (rather than in the z-direction). The reason for
  // this is that in the slice-to-volume case the number of slices will often
  // be smaller than the available number of cores.

  std::vector<unsigned int> nrows = RGT_UTILS::rows_per_thread(static_cast<unsigned int>(out.ysize()),nthr._n);
  std::vector<std::thread> threads(nthr._n-1); // + main thread makes nthr

  if (!defdir.size()) { // If we have an affine only transform
    // Affine only means we can combine all three matrices into one
    NEWMAT::Matrix iB = iM*iA*iT;
    if (!derivdir.size()) { // If we don't need to calculate derivatives
      for (unsigned int i=0; i<nthr._n-1; i++) {
	threads[i] = std::thread(RGT_UTILS::affine_no_derivs<T>,nrows[i],nrows[i+1],std::ref(f),std::ref(slices),std::ref(iB),std::ref(out),valid);
      }
      RGT_UTILS::affine_no_derivs(nrows[nthr._n-1],nrows[nthr._n],f,slices,iB,out,valid);
    }
    else { // We need derivatives in at least one direction
      for (unsigned int i=0; i<nthr._n-1; i++) {
	threads[i] = std::thread(RGT_UTILS::affine_with_derivs<T>,nrows[i],nrows[i+1],std::ref(f),std::ref(slices),
				 std::ref(iB),std::ref(derivdir),std::ref(out),std::ref(deriv),valid);
      }
      RGT_UTILS::affine_with_derivs(nrows[nthr._n-1],nrows[nthr._n],f,slices,iB,derivdir,out,deriv,valid);
    }
  }
  else { // We have displacements in at least one direction
    if (!useiT) { // If the final space is the same as that of the displacement field
      for (unsigned int i=0; i<nthr._n-1; i++) {
	threads[i] = std::thread(RGT_UTILS::displacements_no_iT<T>,nrows[i],nrows[i+1],std::ref(f),std::ref(slices),std::ref(iA),
				 std::ref(d),std::ref(defdir),std::ref(iM),std::ref(derivdir),std::ref(out),std::ref(deriv),valid);
      }
      RGT_UTILS::displacements_no_iT(nrows[nthr._n-1],nrows[nthr._n],f,slices,iA,d,defdir,iM,derivdir,out,deriv,valid);
    }
    else { // If the final space is not the same as that of the displacement field
	for (unsigned int i=0; i<nthr._n-1; i++) {
	  threads[i] = std::thread(RGT_UTILS::displacements_with_iT<T>,nrows[i],nrows[i+1],std::ref(f),std::ref(slices),std::ref(iT),std::ref(iA),
				   std::ref(d),std::ref(defdir),std::ref(iM),std::ref(derivdir),std::ref(out),std::ref(deriv),valid);
	}
	RGT_UTILS::displacements_with_iT(nrows[nthr._n-1],nrows[nthr._n],f,slices,iT,iA,d,defdir,iM,derivdir,out,deriv,valid);
    }
  }
  std::for_each(threads.begin(),threads.end(),std::mem_fn(&std::thread::join)); // Join the threads

  // Make sure that the s/qform of the output volume (and derivs if calculated) is set sensibly
  RGT_UTILS::set_sqform(f,d,defdir,iA,TT,M,out,deriv);

  // Restore extrapolation settings and return
  f.setextrapolationmethod(oldex);
  if (d.tsize()) {
    d.setextrapolationmethod(d_oldex);
    d.setextrapolationvalidity(d_old_epvalidity[0],d_old_epvalidity[1],d_old_epvalidity[2]);
  }
  // All done!
}


/////////////////////////////////////////////////////////////////////
//
// The following two routines are slightly simplified interfaces
// such that users should not have to pass in zero-pointers when
// they do not want to have an "infov" mask output, or when there
// are only a template and an inout volume such that we have no
// need for M or TT.
//
/////////////////////////////////////////////////////////////////////


  template <class T>
  void raw_general_transform(// Input
			     const volume<T>&         vin,        // Input volume
			     const NEWMAT::Matrix&    A,          // 4x4 affine transformation matrix
			     const volume4D<float>&   d,          // Displacement fields
			     const std::vector<int>&  defdir,     // Directions of displacements.
			     const std::vector<int>&  derivdir,   // Directions of derivatives
			     // Output
			     volume<T>&               vout,       // Output volume
			     volume4D<T>&             deriv,      // Partial derivatives
			     // Optional input
			     Utilities::NoOfThreads   nthreads=Utilities::NoOfThreads(1))
  {
    std::vector<unsigned int> slices; // Means all slices will be resampled
    raw_general_transform(vin,A,d,defdir,derivdir,slices,nullptr,nullptr,vout,deriv,nullptr,nthreads);
  }

  template <class T>
  void raw_general_transform(// Input
			     const volume<T>&         vin,        // Input volume
			     const NEWMAT::Matrix&    A,          // 4x4 affine transformation matrix
			     const volume4D<float>&   d,          // Displacement fields
			     const std::vector<int>&  defdir,     // Directions of displacements.
			     const std::vector<int>&  derivdir,   // Directions of derivatives
			     // Output
			     volume<T>&               vout,       // Output volume
			     volume4D<T>&             deriv,      // Partial derivative directions
			     volume<char>&            invol,      // Mask indicating what voxels fell inside original volume
			     // Optional input
			     Utilities::NoOfThreads   nthreads=Utilities::NoOfThreads(1))
  {
    std::vector<unsigned int> slices; // Means all slices will be resampled

    raw_general_transform(vin,A,d,defdir,derivdir,slices,nullptr,nullptr,vout,deriv,&invol,nthreads);
  }

  // These routines supply a convenient interface for applywarp.

  template<class T>
  void apply_warp(// Input
		  const volume<T>&        vin,         // Input volume
		  const NEWMAT::Matrix&   A,           // 4x4 affine transform
		  const volume4D<float>   d,           // Displacement fields
		  const NEWMAT::Matrix&   TT,
		  const NEWMAT::Matrix&   M,
		  // Output
		  volume<T>&              vout,        // Resampled output volume
		  // Optional input
		  Utilities::NoOfThreads  nthreads=Utilities::NoOfThreads(1))
  {
    std::vector<int>          defdir = {0, 1, 2};
    std::vector<int>          derivdir;
    std::vector<unsigned int> slices; // Means all slices will be resampled
    volume4D<T>               deriv;
    const NEWMAT::Matrix      *Tptr = nullptr;
    const NEWMAT::Matrix      *Mptr = nullptr;

    if ((TT-NEWMAT::IdentityMatrix(4)).MaximumAbsoluteValue() > 1e-6) Tptr = &TT;
    if ((M-NEWMAT::IdentityMatrix(4)).MaximumAbsoluteValue() > 1e-6) Mptr = &M;

    raw_general_transform(vin,A,d,defdir,derivdir,slices,Tptr,Mptr,vout,deriv,NULL,nthreads);
  }

  template<class T>
  void apply_warp(// Input
		  const volume<T>&        vin,         // Input volume
		  const NEWMAT::Matrix&   A,           // 4x4 affine transform
		  const volume4D<float>   d,           // Displacement fields
		  const NEWMAT::Matrix&   TT,
		  const NEWMAT::Matrix&   M,
		  // Output
		  volume<T>&              vout,        // Resampled output volume
		  volume<char>&           mask,        // Set when inside original volume
		  // Optional input
		  Utilities::NoOfThreads  nthreads=Utilities::NoOfThreads(1))
  {
    std::vector<int>          defdir = {0, 1, 2};
    std::vector<int>          derivdir;
    std::vector<unsigned int> slices; // Means all slices will be resampled
    volume4D<T>               deriv;
    const NEWMAT::Matrix      *Tptr = NULL;
    const NEWMAT::Matrix      *Mptr = NULL;
    mask.reinitialize(vout.xsize(),vout.ysize(),vout.zsize());  // Just to be certain
    copybasicproperties(vout,mask);

    if ((TT-NEWMAT::IdentityMatrix(4)).MaximumAbsoluteValue() > 1e-6) Tptr = &TT;
    if ((M-NEWMAT::IdentityMatrix(4)).MaximumAbsoluteValue() > 1e-6) Mptr = &M;

    raw_general_transform(vin,A,d,defdir,derivdir,slices,Tptr,Mptr,vout,deriv,&mask,nthreads);
  }

  template<class T>
  void apply_warp(// Input
		  const volume<T>&                 vin,         // Input volume
		  const NEWMAT::Matrix&            A,           // 4x4 affine transform
		  const volume4D<float>            d,           // Displacement fields
		  const NEWMAT::Matrix&            TT,
		  const NEWMAT::Matrix&            M,
		  const std::vector<unsigned int>& slices,
		  // Output
		  volume<T>&                       vout,        // Resampled output volume
		  volume<char>&                    mask,        // Set when inside original volume
		  // Optional input
		  Utilities::NoOfThreads           nthreads=Utilities::NoOfThreads(1))
  {
    std::vector<int>          defdir = {0, 1, 2};
    std::vector<int>          derivdir;
    volume4D<T>               deriv;
    const NEWMAT::Matrix      *Tptr = NULL;
    const NEWMAT::Matrix      *Mptr = NULL;

    if (!samesize(vout,mask)) { // Just to be certain
      mask.reinitialize(vout.xsize(),vout.ysize(),vout.zsize());
      copybasicproperties(vout,mask);
      mask = 0;
    }

    if ((TT-NEWMAT::IdentityMatrix(4)).MaximumAbsoluteValue() > 1e-6) Tptr = &TT;
    if ((M-NEWMAT::IdentityMatrix(4)).MaximumAbsoluteValue() > 1e-6) Mptr = &M;

    raw_general_transform(vin,A,d,defdir,derivdir,slices,Tptr,Mptr,vout,deriv,&mask,nthreads);
  }


/////////////////////////////////////////////////////////////////////
//
// The following three routines are interfaces to mimick the old
// functions apply_warp and raw_apply_warp. These are used mainly
// for resampling of images that are typically related to the
// input to e.g. fnirt or fugue through some rigid matrix M.
// Examples of M would be a rigid mapping between a subjects
// functional volumes and his/her structural or between a subjects
// functional volumes and his/her field map.
//
/////////////////////////////////////////////////////////////////////

  template <class T>
  int apply_warp(const volume<T>&        invol,
		 volume<T>&              outvol,
		 const volume4D<float>&  warpvol,
		 // Optional input
		 Utilities::NoOfThreads  nthreads=Utilities::NoOfThreads(1))
  {
    NEWMAT::IdentityMatrix eye(4);
    return(apply_warp(invol,outvol,warpvol,eye,eye,nthreads));
  }

  template <class T>
  int apply_warp(const volume<T>&                invol,
		 volume<T>&                      outvol,
		 const volume4D<float>&          warpvol,
		 const NEWMAT::Matrix&           premat,
		 const NEWMAT::Matrix&           postmat,
		 // Optional input
		 Utilities::NoOfThreads          nthreads=Utilities::NoOfThreads(1))
  {
    // set the desired extrapolation settings
    extrapolation oldin = invol.getextrapolationmethod();
    extrapolation oldwarp = warpvol.getextrapolationmethod();
    warpvol.setextrapolationmethod(extraslice);
    invol.setextrapolationmethod(extraslice);
    float oldpad = invol.getpadvalue();
    invol.setpadvalue(invol.backgroundval());

    int retval = raw_apply_warp(invol,outvol,warpvol,premat,postmat,nthreads);

    // restore extrapolation settings
    warpvol.setextrapolationmethod(oldwarp);
    invol.setextrapolationmethod(oldin);
    invol.setpadvalue(oldpad);

    return retval;
  }

  template <class T>
  int raw_apply_warp(const volume<T>&                invol,
		     volume<T>&                      outvol,
		     const volume4D<float>&          warpvol,
		     const NEWMAT::Matrix&           premat,
		     const NEWMAT::Matrix&           postmat,
		     // Optional input
		     Utilities::NoOfThreads          nthreads=Utilities::NoOfThreads(1))
  {
    NEWMAT::IdentityMatrix    A(4);
    std::vector<int>          defdir = {0, 1, 2};
    std::vector<int>          derivdir;
    std::vector<unsigned int> slices; // Means all slices will be resampled
    volume4D<T>               deriv;

    raw_general_transform(invol,A,warpvol,defdir,derivdir,slices,&postmat,&premat,outvol,deriv,NULL,nthreads);

    return(0);
  }

  // The following handful of routines are simplified interfaces to
  // raw_general_transform that may be convenient for certain
  // specific applications.

  template <class T>
  void affine_transform(// Input
			const volume<T>&         vin,
			const NEWMAT::Matrix&    aff,
			// Output
			volume<T>&               vout,
			// Optional input
			Utilities::NoOfThreads   nthreads=Utilities::NoOfThreads(1))
  {
    volume4D<float>  pdf;
    std::vector<int> pdefdir;
    volume4D<float>  deriv;
    std::vector<int> pderivdir;

    raw_general_transform(vin,aff,pdf,pdefdir,pderivdir,vout,deriv,nthreads);
  }

  template <class T>
  void affine_transform(// Input
			const volume<T>&         vin,
			const NEWMAT::Matrix&    aff,
			// Output
			volume<T>&               vout,
			volume<char>&            inside_volume,
			// Optional input
			Utilities::NoOfThreads   nthreads=Utilities::NoOfThreads(1))
  {
    volume4D<float>  pdf;
    std::vector<int> pdefdir;
    volume4D<float>  deriv;
    std::vector<int> pderivdir;

    raw_general_transform(vin,aff,pdf,pdefdir,pderivdir,vout,deriv,inside_volume,nthreads);
  }

  template <class T>
  void affine_transform(// Input
			const volume<T>&                   vin,
			const NEWMAT::Matrix&              aff,
			const std::vector<unsigned int>&   slices,
			// Output
			volume<T>&                         vout,
			volume<char>&                      inside_volume,
			// Optional input
			Utilities::NoOfThreads             nthreads=Utilities::NoOfThreads(1))
  {
    volume4D<float>  pdf;
    std::vector<int> pdefdir;
    volume4D<float>  pderiv;
    std::vector<int> pderivdir;

    raw_general_transform(vin,aff,pdf,pdefdir,pderivdir,slices,nullptr,nullptr,vout,pderiv,&inside_volume,nthreads);
  }

  template <class T>
  void affine_transform_3partial(// Input
				 const volume<T>&             vin,
				 const NEWMAT::Matrix&        aff,
				 // Output
				 volume<T>&                   vout,
				 volume4D<T>&                 deriv,
				 // Optional input
				 Utilities::NoOfThreads       nthreads=Utilities::NoOfThreads(1))
  {
    volume4D<float>  pdf;
    std::vector<int> pdefdir;
    std::vector<int> pderivdir = {0, 1, 2};
    raw_general_transform(vin,aff,pdf,pdefdir,pderivdir,vout,deriv,nthreads);
  }

  template <class T>
  void affine_transform_3partial(// Input
				 const volume<T>&       vin,
				 const NEWMAT::Matrix&  aff,
				 // Output
				 volume<T>&             vout,
				 volume4D<T>&           deriv,
				 volume<char>&          invol,
				 // Optional input
				 Utilities::NoOfThreads nthreads=Utilities::NoOfThreads(1))
  {
    volume4D<float>  pdf;
    std::vector<int> pdefdir;
    std::vector<int> pderivdir = {0, 1, 2};
    raw_general_transform(vin,aff,pdf,pdefdir,pderivdir,vout,deriv,invol,nthreads);
  }

  template <class T>
  void displacement_transform_1D(// Input
				 const volume<T>&       vin,
				 const NEWMAT::Matrix&  aff,
				 const volume<float>&   df,
				 int                    defdir,
				 // Output
				 volume<T>&             vout,
				 // Optional input
				 Utilities::NoOfThreads nthreads=Utilities::NoOfThreads(1))
  {
    volume4D<float>  pdf;
    std::vector<int> pdefdir(1,defdir);
    std::vector<int> pderivdir;
    volume4D<T>      pderiv;

    pdf.addvolume(df);
    raw_general_transform(vin,aff,pdf,pdefdir,pderivdir,vout,pderiv,nthreads);
  }

  template <class T>
  void displacement_transform_1D(// Input
				 const volume<T>&       vin,
				 const NEWMAT::Matrix&  aff,
				 const volume<float>&   df,
				 int                    defdir,
				 // Output
				 volume<T>&             vout,
				 volume<char>&          invol,
				 // Optional input
				 Utilities::NoOfThreads nthreads=Utilities::NoOfThreads(1))
  {
    volume4D<float>  pdf;
    std::vector<int> pdefdir(1,defdir);
    std::vector<int> pderivdir;
    volume4D<T>      pderiv;

    pdf.addvolume(df);
    raw_general_transform(vin,aff,pdf,pdefdir,pderivdir,vout,pderiv,invol,nthreads);
  }

  template <class T>
  void displacement_transform_1D_3partial(// Input
					  const volume<T>&       vin,
					  const NEWMAT::Matrix&  aff,
					  const volume<float>&   df,
					  int                    dir,
					  // Output
					  volume<T>&             vout,
					  volume4D<T>&           deriv,
					  // Optional input
					  Utilities::NoOfThreads nthreads=Utilities::NoOfThreads(1))
  {
    volume4D<float>      pdf;
    std::vector<int>     pdefdir(1,dir);
    std::vector<int>     pderivdir = {0, 1, 2};

    pdf.addvolume(df);
    raw_general_transform(vin,aff,pdf,pdefdir,pderivdir,vout,deriv,nthreads);
  }

  template <class T>
  void displacement_transform_1D_3partial(// Input
					  const volume<T>&       vin,
					  const NEWMAT::Matrix&  aff,
					  const volume<float>&   df,
					  int                    dir,
					  // Output
					  volume<T>&             vout,
					  volume4D<T>&           deriv,
					  volume<char>&          invol,
					  // Optional input
					  Utilities::NoOfThreads nthreads=Utilities::NoOfThreads(1))
  {
    volume4D<float>      pdf;
    std::vector<int>     pdefdir(1,dir);
    std::vector<int>     pderivdir = {0, 1, 2};

    pdf.addvolume(df);
    raw_general_transform(vin,aff,pdf,pdefdir,pderivdir,vout,deriv,invol,nthreads);
  }

  template <class T>
  void general_transform(// Input
			 const volume<T>&         vin,
			 const NEWMAT::Matrix&    aff,
			 const volume4D<float>&   df,
			 // Output
			 volume<T>&               vout,
			 // Optional input
			 Utilities::NoOfThreads   nthreads=Utilities::NoOfThreads(1))
  {
    std::vector<int>  pdefdir = {0, 1, 2};
    std::vector<int>  pderivdir;
    volume4D<T>       pderiv;

    raw_general_transform(vin,aff,df,pdefdir,pderivdir,vout,pderiv,nthreads);
  }

  template <class T>
  void general_transform(// Input
			 const volume<T>&         vin,
			 const NEWMAT::Matrix&    aff,
			 const volume4D<float>&   df,
			 // Output
			 volume<T>&               vout,
			 volume<char>&            invol,
			 // Optional input
			 Utilities::NoOfThreads   nthreads=Utilities::NoOfThreads(1))
  {
    std::vector<int>  pdefdir = {0, 1, 2};
    std::vector<int>  pderivdir;
    volume4D<T>       pderiv;

    raw_general_transform(vin,aff,df,pdefdir,pderivdir,vout,pderiv,invol,nthreads);
  }

  template <class T>
  void general_transform_3partial(// Input
				  const volume<T>&         vin,
				  const NEWMAT::Matrix&    aff,
				  const volume4D<float>&   df,
				  // Output
				  volume<T>&               vout,
				  volume4D<T>&             deriv,
				  // Optional input
				  Utilities::NoOfThreads   nthreads=Utilities::NoOfThreads(1))
  {
    std::vector<int>   dir = {0, 1, 2};

    raw_general_transform(vin,aff,df,dir,dir,vout,deriv,nthreads);
  }

  template <class T>
  void general_transform_3partial(// Input
				  const volume<T>&         vin,
				  const NEWMAT::Matrix&    aff,
				  const volume4D<float>&   df,
				  // Output
				  volume<T>&               vout,
				  volume4D<T>&             deriv,
				  volume<char>&            invol,
				  // Optional input
				  Utilities::NoOfThreads   nthreads=Utilities::NoOfThreads(1))
  {
    std::vector<int>   dir = {0, 1, 2};

    raw_general_transform(vin,aff,df,dir,dir,vout,deriv,invol,nthreads);
  }

  //
  // This routine mimics some of the code in raw_general_transform.
  // Ideally they should be re-written to allow for code re-use
  // but this solution is the least "invasive" in the sense of
  // risking to introduce any new bugs.
  // It will return a 3D displacement field in mm.
  //
  template <class T>
  void get_displacement_fields(// Input
                                  const volume<T>&         vin,
                                  const NEWMAT::Matrix&    aff,
                                  const volume4D<float>&   dfin,
                                  // Output
                                  volume4D<float>&         dfout)
  {
    if (dfin.tsize() != 3) imthrow("NEWIMAGE::get_displacement_field: dfin.tsize() must be 3.",11);
    if (!NEWIMAGE::samesize(vin,dfin[0])) imthrow("NEWIMAGE::get_displacement_field: mismatch between vin and dfin.",11);
    if (!NEWIMAGE::samesize(dfin,dfout)) imthrow("NEWIMAGE::get_displacement_field: mismatch between dfin and dfout.",11);
    NEWMAT::Matrix v2w = vin.sampling_mat();  // voxel->world
    NEWMAT::Matrix iA = aff.i();              // world->transformed_world
    NEWMAT::ColumnVector vox(4); vox(4) = 1.0;
    NEWMAT::ColumnVector mm(4);
    NEWMAT::ColumnVector tmm(4);
    for (int k=0; k<vin.zsize(); k++) {
      vox(3) = k;
      for (int j=0; j<vin.ysize(); j++) {
	vox(2) = j;
	for (int i=0; i<vin.xsize(); i++) {
	  vox(1) = i;
	  mm = v2w*vox;
	  tmm = iA*mm;
          dfout(i,j,k,0) = tmm(1) + dfin(i,j,k,0) - mm(1);
          dfout(i,j,k,1) = tmm(2) + dfin(i,j,k,1) - mm(2);
          dfout(i,j,k,2) = tmm(3) + dfin(i,j,k,2) - mm(3);
	}
      }
    }
  }

} // End namespace NEWIMAGE

namespace RGT_UTILS { // raw_general_transform utilities

template <class T>
std::tuple<bool,std::string> validate_input(const NEWMAT::Matrix&             A,
					    const NEWMAT::Matrix              *TT,
					    const NEWMAT::Matrix              *M,
					    const NEWIMAGE::volume4D<float>&  d,
					    const std::vector<int>&           defdir,
					    const std::vector<int>&           derivdir,
					    const std::vector<unsigned int>&  slices,
					    const NEWIMAGE::volume<T>&        out,
					    const NEWIMAGE::volume4D<T>&      deriv,
					    const NEWIMAGE::volume<char>      *valid)
{
  if (A.Nrows() != 4 || A.Ncols() != 4) return(std::make_tuple(false,"A must be 4x4 matrix"));
  if (TT != nullptr && (TT->Nrows() != 4 || TT->Ncols() != 4)) return(std::make_tuple(false,"If specified TT must be a 4x4 matrix"));
  if (M != nullptr && (M->Nrows() != 4 || M->Ncols() != 4)) return(std::make_tuple(false,"If specified M must be a 4x4 matrix"));
  if (static_cast<int>(defdir.size()) != d.tsize()) return(std::make_tuple(false,"Mismatch in input. defdir.size() must equal d.tsize()"));
  if (std::any_of(defdir.begin(),defdir.end(),[](int dir){ return(dir<0 || dir>2); })) return(std::make_tuple(false,"Displacements can only be specified for directions 0,1 or 2"));
  if (static_cast<int>(derivdir.size()) != deriv.tsize()) return(std::make_tuple(false,"Mismatch in input. derivdir.size() must equal deriv.tsize()"));
  if (std::any_of(derivdir.begin(),derivdir.end(),[](int dir){ return(dir<0 || dir>2); })) return(std::make_tuple(false,"Derivatives can only be specified for directions 0,1 or 2"));
  if (out.nvoxels() <= 0) return(std::make_tuple(false,"Size of vout must be set"));
  if (slices.size()==0) { // If the slices vector isn't empty, check that it makes sense
    if (int(slices.size()) > out.zsize()) return(std::make_tuple(false,"Mismatch between slices and out input"));
    else {
      for (unsigned int i=0; i<slices.size(); i++) if (int(slices[i])>=out.zsize()) return(std::make_tuple(false,"Invalid element in slices"));
    }
  }
  if (derivdir.size() && !samesize(out,deriv[0])) return(std::make_tuple(false,"vout and deriv must have same dimensions"));
  if (valid && !samesize(out,*valid)) return(std::make_tuple(false,"vout and valid must have same dimensions"));
  return(std::make_tuple(true,"All good!"));
}

template <class T>
std::tuple<NEWIMAGE::extrapolation,NEWIMAGE::extrapolation,
	   std::vector<bool> > set_extrapolation(const NEWIMAGE::volume<T>&       f,
						 const NEWIMAGE::volume4D<float>& d)
{
  NEWIMAGE::extrapolation oldex = f.getextrapolationmethod();
  NEWIMAGE::extrapolation d_oldex = NEWIMAGE::extraslice;  // Assign arbitrary value to silence compiler
  std::vector<bool>  d_old_epvalidity;
  if ((oldex==NEWIMAGE::boundsassert) || (oldex==NEWIMAGE::boundsexception)) {f.setextrapolationmethod(NEWIMAGE::constpad);}
  if (d.tsize()) {
    d_oldex = d.getextrapolationmethod();
    if (oldex==NEWIMAGE::periodic) d.setextrapolationmethod(NEWIMAGE::periodic);
    else d.setextrapolationmethod(NEWIMAGE::extraslice);
    d_old_epvalidity = d.getextrapolationvalidity();
    std::vector<bool> epvalidity = f.getextrapolationvalidity();
    d.setextrapolationvalidity(epvalidity[0],epvalidity[1],epvalidity[2]);
  }
  return(std::make_tuple(oldex,d_oldex,d_old_epvalidity));
}

template <class T>
std::tuple<NEWMAT::Matrix,bool> make_iT(const NEWIMAGE::volume4D<float>& d,
					const NEWIMAGE::volume<T>&       out,
					const NEWMAT::Matrix             *TT)
{
  NEWMAT::Matrix iT(4,4);
  if (TT) {
    if (d.tsize()) iT = d.sampling_mat().i() * TT->i() * out.sampling_mat();
    else iT = out.sampling_mat().i() * TT->i() * out.sampling_mat();
  }
  else {
    if (d.tsize()) iT = d.sampling_mat().i() * out.sampling_mat();
    else iT = NEWMAT::IdentityMatrix(4);
  }
  // We should only use iT if it is different from the identity matrix.
  bool useiT = ((iT-NEWMAT::IdentityMatrix(4)).MaximumAbsoluteValue() > 1e-6) ? true: false;

  return(std::make_tuple(iT,useiT));
}

// This function is used for resampling using an affine transform
// only when no derivates are needed. The parallellisation is along
// the y-direction.
template <class T>
void affine_no_derivs(// Input
		      unsigned int                     first_j,
		      unsigned int                     last_j,
		      const NEWIMAGE::volume<T>&       f,
		      const std::vector<unsigned int>& slices,
		      const NEWMAT::Matrix&            A,
		      // Output
		      NEWIMAGE::volume<T>&             out,
		      NEWIMAGE::volume<char>           *valid)
{
  float A11=A(1,1), A12=A(1,2), A13=A(1,3), A14=A(1,4);
  float A21=A(2,1), A22=A(2,2), A23=A(2,3), A24=A(2,4);
  float A31=A(3,1), A32=A(3,2), A33=A(3,3), A34=A(3,4);

  for (unsigned int si=0; si<slices.size(); si++) {
    int k = static_cast<int>(slices[si]);
    for (int j=static_cast<int>(first_j); j<static_cast<int>(last_j); j++) {
      float x = j*A12 + k*A13 + A14;
      float y = j*A22 + k*A23 + A24;
      float z = j*A32 + k*A33 + A34;
      for (int i=0; i<out.xsize(); i++) {
	out(i,j,k) = static_cast<T>(f.interpolate(x,y,z));
	if (valid != nullptr) (*valid)(i,j,k) = f.valid(x,y,z) ? 1 : 0;
	x += A11; y += A21; z += A31;
      }
    }
  }

  return;
}

// This function is used for resampling using an affine transform
// only when at least one derivate is needed. The parallellisation
// is along the y-direction.
template <class T>
void affine_with_derivs(// Input
			unsigned int                     first_j,
			unsigned int                     last_j,
			const NEWIMAGE::volume<T>&       f,
			const std::vector<unsigned int>& slices,
			const NEWMAT::Matrix&            A,
			const std::vector<int>&          derivdir,
			// Output
			NEWIMAGE::volume<T>&             out,
			NEWIMAGE::volume4D<T>&           deriv,
			NEWIMAGE::volume<char>           *valid)
{
  float A11=A(1,1), A12=A(1,2), A13=A(1,3), A14=A(1,4);
  float A21=A(2,1), A22=A(2,2), A23=A(2,3), A24=A(2,4);
  float A31=A(3,1), A32=A(3,2), A33=A(3,3), A34=A(3,4);

  for (unsigned int si=0; si<slices.size(); si++) {
    int k = static_cast<int>(slices[si]);
    for (int j=static_cast<int>(first_j); j<static_cast<int>(last_j); j++) {
      float x = j*A12 + k*A13 + A14;
      float y = j*A22 + k*A23 + A24;
      float z = j*A32 + k*A33 + A34;
      for (int i=0; i<out.xsize(); i++) {
	if (derivdir.size() == 1) { // Derivative in just a single direction
	  float tmp;
	  out(i,j,k) = static_cast<T>(f.interp1partial(x,y,z,derivdir[0],&tmp));
	  deriv(i,j,k,0) = static_cast<T>(tmp);
	}
	else { // Derivative in more than one direction
	  float tmp[3];
	  out(i,j,k) = static_cast<T>(f.interp3partial(x,y,z,&tmp[0],&tmp[1],&tmp[2]));
	  for (unsigned int di=0; di<derivdir.size(); di++) deriv(i,j,k,di) = static_cast<T>(tmp[derivdir[di]]);
	}
	if (valid != nullptr) (*valid)(i,j,k) = static_cast<char>(f.valid(x,y,z));
	x += A11; y += A21; z += A31;
      }
    }
  }
  return;
}

// This function is used for resampling using a displacement field
// when the final target volume is in the same space as the displacement
// field.
// The parallellisation is along the y-direction.
template <class T>
void displacements_no_iT(// Input
			 unsigned int                     first_j,
			 unsigned int                     last_j,
			 const NEWIMAGE::volume<T>&       f,
			 const std::vector<unsigned int>& slices,
			 const NEWMAT::Matrix&            A,
			 const NEWIMAGE::volume4D<float>& d,
			 const std::vector<int>&          defdir,
			 const NEWMAT::Matrix&            M,
			 const std::vector<int>&          derivdir,
			 // Output
			 NEWIMAGE::volume<T>&             out,
			 NEWIMAGE::volume<T>&             deriv,
			 NEWIMAGE::volume<char>           *valid)
{
  float A11=A(1,1), A12=A(1,2), A13=A(1,3), A14=A(1,4);
  float A21=A(2,1), A22=A(2,2), A23=A(2,3), A24=A(2,4);
  float A31=A(3,1), A32=A(3,2), A33=A(3,3), A34=A(3,4);

  float M11=M(1,1), M12=M(1,2), M13=M(1,3), M14=M(1,4);
  float M21=M(2,1), M22=M(2,2), M23=M(2,3), M24=M(2,4);
  float M31=M(3,1), M32=M(3,2), M33=M(3,3), M34=M(3,4);

  float x[3], xx[3];
  for (unsigned int si=0; si<slices.size(); si++) {
    int k = static_cast<int>(slices[si]);
    for (int j=static_cast<int>(first_j); j<static_cast<int>(last_j); j++) {
      for (int i=0; i<out.xsize(); i++) {
	x[0] = i*A11 + j*A12 + k*A13 + A14;
	x[1] = i*A21 + j*A22 + k*A23 + A24;
	x[2] = i*A31 + j*A32 + k*A33 + A34;
	for (unsigned int di=0; di<defdir.size(); di++) x[defdir[di]] += d(i,j,k,di);
	xx[0] = x[0]*M11 + x[1]*M12 + x[2]*M13 + M14;
	xx[1] = x[0]*M21 + x[1]*M22 + x[2]*M23 + M24;
	xx[2] = x[0]*M31 + x[1]*M32 + x[2]*M33 + M34;
	if (derivdir.size() == 0) out(i,j,k) = static_cast<T>(f.interpolate(xx[0],xx[1],xx[2]));
	else if (derivdir.size() == 1) {
	  float tmp;
	  out(i,j,k) = static_cast<T>(f.interp1partial(xx[0],xx[1],xx[2],derivdir[0],&tmp));
	  deriv(i,j,k,0) = static_cast<T>(tmp);
	}
	else {
	  float tmp[3];
	  out(i,j,k) = static_cast<T>(f.interp3partial(xx[0],xx[1],xx[2],&tmp[0],&tmp[1],&tmp[2]));
	  for (unsigned int di=0; di<derivdir.size(); di++) deriv(i,j,k,di) = tmp[derivdir[di]];
	}
	if (valid != nullptr) (*valid)(i,j,k) = static_cast<char>(f.valid(xx[0],xx[1],xx[2]));
      }
    }
  }
  return;
}

// This function is used for resampling using a displacement field
// when the final target volume is in a different space to the displacement
// field. This can be for example if one has estimated the field in a 2mm
// space and wants to resample the image to a 1mm space.
// The parallellisation is along the y-direction.
template <class T>
void displacements_with_iT(// Input
			   unsigned int                     first_j,
			   unsigned int                     last_j,
			   const NEWIMAGE::volume<T>&       f,
			   const std::vector<unsigned int>& slices,
			   const NEWMAT::Matrix&            iT,
			   const NEWMAT::Matrix&            A,
			   const NEWIMAGE::volume4D<float>& d,
			   const std::vector<int>&          defdir,
			   const NEWMAT::Matrix&            M,
			   const std::vector<int>&          derivdir,
			   // Output
			   NEWIMAGE::volume<T>&             out,
			   NEWIMAGE::volume<T>&             deriv,
			   NEWIMAGE::volume<char>           *valid)
{
  float T11=iT(1,1), T12=iT(1,2), T13=iT(1,3), T14=iT(1,4);
  float T21=iT(2,1), T22=iT(2,2), T23=iT(2,3), T24=iT(2,4);
  float T31=iT(3,1), T32=iT(3,2), T33=iT(3,3), T34=iT(3,4);

  float A11=A(1,1), A12=A(1,2), A13=A(1,3), A14=A(1,4);
  float A21=A(2,1), A22=A(2,2), A23=A(2,3), A24=A(2,4);
  float A31=A(3,1), A32=A(3,2), A33=A(3,3), A34=A(3,4);

  float M11=M(1,1), M12=M(1,2), M13=M(1,3), M14=M(1,4);
  float M21=M(2,1), M22=M(2,2), M23=M(2,3), M24=M(2,4);
  float M31=M(3,1), M32=M(3,2), M33=M(3,3), M34=M(3,4);

  float x[3], xx[3];
  for (unsigned int si=0; si<slices.size(); si++) {
    int k = static_cast<int>(slices[si]);
    for (int j=static_cast<int>(first_j); j<static_cast<int>(last_j); j++) {
      for (int i=0; i<out.xsize(); i++) {
	x[0] = i*T11 + j*T12 + k*T13 + T14;
	x[1] = i*T21 + j*T22 + k*T23 + T24;
	x[2] = i*T31 + j*T32 + k*T33 + T34;
	xx[0] = x[0]*A11 + x[1]*A12 + x[2]*A13 + A14;
	xx[1] = x[0]*A21 + x[1]*A22 + x[2]*A23 + A24;
	xx[2] = x[0]*A31 + x[1]*A32 + x[2]*A33 + A34;
	for (unsigned int di=0; di<defdir.size(); di++) xx[defdir[di]] += d[di].interpolate(x[0],x[1],x[2]);
	if (valid != nullptr) (*valid)(i,j,k) = static_cast<char>(d.valid(x[0],x[1],x[2]));
	x[0] = xx[0]*M11 + xx[1]*M12 + xx[2]*M13 + M14;
	x[1] = xx[0]*M21 + xx[1]*M22 + xx[2]*M23 + M24;
	x[2] = xx[0]*M31 + xx[1]*M32 + xx[2]*M33 + M34;
	if (derivdir.size() == 0) out(i,j,k) = static_cast<T>(f.interpolate(x[0],x[1],x[2]));
	else if (derivdir.size() == 1) { // If we want a single partial
	  float tmp;
	  out(i,j,k) = static_cast<T>(f.interp1partial(x[0],x[1],x[2],derivdir[0],&tmp));
	  deriv(i,j,k,0) = static_cast<T>(tmp);
	}
	else { // If we want more than one derivative
	  float tmp[3];
	  out(i,j,k) = static_cast<T>(f.interp3partial(x[0],x[1],x[2],&tmp[0],&tmp[1],&tmp[2]));
	  for (unsigned int di=0; di<derivdir.size(); di++) deriv(i,j,k,di) = tmp[derivdir[di]];
	}
	if (valid != nullptr) (*valid)(i,j,k) = static_cast<char>(f.valid(x[0],x[1],x[2]) && (*valid)(i,j,k));
      }
    }
  }
  return;
}

//
// Set the sform and qform appropriately.
// 1. If the outvol has it's codes set, then leave it.
// 2. If outvol doesn't have the codes set AND there
//    is a warpfield AND the warpfield has the codes
//    set then copy codes warpfield->outvol.
// 3. If the outvol doesn't have its codes set AND
//    there is NO warpfield AND invol has its codes
//    set, then set the q and sform to the transformed
//    version of those in invol. Set codes to ALIGNED_ANAT.
//
template <class T>
void set_sqform(// Input
		const NEWIMAGE::volume<T>&        f,
		const NEWIMAGE::volume4D<float>&  d,
		const std::vector<int>&           defdir,
		NEWMAT::Matrix                    iA,      // Copy is intentional
		const NEWMAT::Matrix              *TT,
		const NEWMAT::Matrix              *M,
		// Output
		NEWIMAGE::volume<T>&              out,
		NEWIMAGE::volume4D<T>&            deriv)
{
  if ( (out.sform_code()==NiftiIO::NIFTI_XFORM_UNKNOWN) &&       // qform is set in outvol
       (out.qform_code()!=NiftiIO::NIFTI_XFORM_UNKNOWN) ) {
    out.set_sform(out.qform_code(), out.qform_mat());            // Copy qform->sform
  }
  else if ( (out.qform_code()==NiftiIO::NIFTI_XFORM_UNKNOWN) &&  // sform is set in outvol
	    (out.sform_code()!=NiftiIO::NIFTI_XFORM_UNKNOWN) ) {
    out.set_qform(out.sform_code(), out.sform_mat());            // Copy sform->qform
  }
  else if ( (out.qform_code()==NiftiIO::NIFTI_XFORM_UNKNOWN) &&  // Neither is set in outvol
	    (out.sform_code()==NiftiIO::NIFTI_XFORM_UNKNOWN) ) {
    if (defdir.size()) {                                         // If there is a warp-field
      if (d.sform_code()!=NiftiIO::NIFTI_XFORM_UNKNOWN) {        // If sform of warp-field is known
	out.set_sform(d.sform_code(),d.sform_mat());
	out.set_qform(d.sform_code(),d.sform_mat());
      }
      else if (d.qform_code()!=NiftiIO::NIFTI_XFORM_UNKNOWN) {
	out.set_sform(d.qform_code(),d.qform_mat());
	out.set_qform(d.qform_code(),d.qform_mat());
      }
    }
    else { // I there is no warp-field, i.e. an affine transform, and s/qform not set in outvol
      if (f.sform_code()!=NiftiIO::NIFTI_XFORM_UNKNOWN) {      // Combine input sform with affine matrix
	if (TT != nullptr) iA = iA*TT->i();
	if (M != nullptr) iA = M->i()*iA;
	iA = f.sform_mat()*iA;
	out.set_sform(NiftiIO::NIFTI_XFORM_ALIGNED_ANAT,iA);
	out.set_qform(NiftiIO::NIFTI_XFORM_ALIGNED_ANAT,iA);
      }
      else if (f.qform_code()!=NiftiIO::NIFTI_XFORM_UNKNOWN) { // Combine input qform with affine matrix
	if (TT != nullptr) iA = iA*TT->i();
	if (M != nullptr) iA = M->i()*iA;
	iA = f.qform_mat()*iA;
	out.set_sform(NiftiIO::NIFTI_XFORM_ALIGNED_ANAT,iA);
	out.set_qform(NiftiIO::NIFTI_XFORM_ALIGNED_ANAT,iA);
      }
    }
  }
  // Set s/qform in deriv to the same as in out
  if (deriv.tsize()) {
    deriv.set_sform(out.sform_code(), out.sform_mat());
    deriv.set_qform(out.qform_code(), out.qform_mat());
  }

}

} // End namespace RGT_UTILS

#ifdef I_DEFINED_ET
#undef I_DEFINED_ET
#undef EXPOSE_TREACHEROUS   // Avoid exporting dodgy routines
#endif

#endif // End #if !defined(__warpfns_h)
