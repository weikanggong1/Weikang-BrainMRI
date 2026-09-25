//
//  Declarations for tetrahedron class Tetrahedron
//
//  tetrahedron.h
//
//  Implements a tetrahedron class that can be used for inverting
//  3D warp-fields. It has functionality for finding the tetrahedron
//  (in warped space) that encompasses a point (given in original
//  space). Once that tetrahedron is found is has functionality
//  for calculating the exact location of that point in transformed
//  space.
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2010 University of Oxford
//
/*  CCOPYRIGHT  */

#ifndef tetrahedron_h
#define tetrahedron_h

#include <fstream>
#include <memory>
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"

namespace FNIRT {

class TetrahedronException: public std::exception
{
private:
  std::string m_msg;
public:
  TetrahedronException(const std::string& msg) throw(): m_msg(msg) {}

  virtual const char * what() const throw() {
    return std::string("Tetrahedron::" + m_msg).c_str();
  }

  ~TetrahedronException() throw() {}
};

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// This class is used for inverting warp-fields. Let us say we have
// a point x,y,z in original point and we want to know what point
// x',y',z' in warped space that corresponds to. We do so by finding
// the indicies of the tetrahedron in warped space for which x,y,z
// falls within when these 4 vertices of that tetrahedron in
// original space. Within this tetrahedron every point x',y',z' is
// given by equations
// x' = a1 + b1*x + c1*y + d1*z
// y' = a2 + b2*x + c2*y + d2*z
// z' = a3 + b3*x + c3*y + d3*z
// And the coefficients a1,b1,...,d3 can be calculated from the
// coordinates of the vertices of the tetrahedron in the two spaces.
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


class Tetrahedron
{
public:
  Tetrahedron(int x, int y, int z,
              unsigned int xs, unsigned int ys, unsigned int zs) :
    _i1(x), _j1(y), _k1(z), _is(xs), _js(ys), _ks(zs), _miter(1000), _hasaff(false)
  {
    populate_me();
    set_coordinates();
  }
  Tetrahedron(int x, int y, int z,
              const std::shared_ptr<NEWIMAGE::volume4D<float> >& def) :
              _i1(x), _j1(y), _k1(z), _is(def->xsize()), _js(def->ysize()),
              _ks(def->zsize()), _miter(1000), _hasaff(false), _def(def)
  {
    populate_me();
    set_coordinates();
  }
  Tetrahedron(int x, int y, int z,
              const std::shared_ptr<NEWIMAGE::volume4D<float> >& def,
              const NEWMAT::Matrix& aff) :
              _i1(x), _j1(y), _k1(z), _is(def->xsize()), _js(def->ysize()),
              _ks(def->zsize()), _miter(1000), _hasaff(true), _def(def)
  {
    populate_me();
    set_affine(aff);
    set_coordinates();
  }
  ~Tetrahedron() {}

  void SetFirstPoint(int x, int y, int z)
  {
    _i1 = x; _j1 = y; _k1 = z;
    populate_me();
    set_coordinates();
  }
  void SetDeformationField(const std::shared_ptr<NEWIMAGE::volume4D<float> >& def)
  {
    const std::shared_ptr<NEWIMAGE::volume4D<float> >  test = def;
    _def = def;
    set_coordinates();
  }
  void SetAffine(const NEWMAT::Matrix& aff)
  {
    set_affine(aff);
    set_coordinates();
  }
  void SetIgnoreFOV(bool set=true)
  {
    _ioob=set;
    _def->setextrapolationmethod(NEWIMAGE::zeropad);
  }
  bool FindPoint(double x, double y, double z, double& ox, double& oy, double& oz)
  {
    if (find_tetrahedron(x,y,z)) {
      get_point(x,y,z,ox,oy,oz);
      return(true);
    }
    else return(false);
  }
  void Print() { PrintIndicies(); PrintCoordinates(); }
  void PrintIndicies()
  {
    std::cout << _i1 << "  " << _j1 << "  " << _k1 << std::endl;
    std::cout << _i2 << "  " << _j2 << "  " << _k2 << std::endl;
    std::cout << _i3 << "  " << _j3 << "  " << _k3 << std::endl;
    std::cout << _i4 << "  " << _j4 << "  " << _k4 << std::endl;
  }
  void PrintCoordinates()
  {
    std::cout << _x1 << "  " << _y1 << "  " << _z1 << std::endl;
    std::cout << _x2 << "  " << _y2 << "  " << _z2 << std::endl;
    std::cout << _x3 << "  " << _y3 << "  " << _z3 << std::endl;
    std::cout << _x4 << "  " << _y4 << "  " << _z4 << std::endl;
  }
private:
  void populate_me();
  void set_affine(const NEWMAT::Matrix& aff);
  void set_coordinates();
  void set_coordinates(int indx);
  bool is_point_in_tetrahedron(double x, double y, double z, int& indx);
  bool is_point_on_right_side_of_plane(double x, double y, double z, unsigned int indx);
  bool mirror_tetrahedron(int indx);
  bool find_tetrahedron(double x, double y, double z);
  void get_point(double x, double y, double z, double& ox, double& oy, double& oz);
  int           _i1, _j1, _k1;  // Index (in warped space) of first point of tetrahedron
  int           _i2, _j2, _k2;  // Index (in warped space) of second point of tetrahedron
  int           _i3, _j3, _k3;  // You get it?
  int           _i4, _j4, _k4;
  unsigned int _is, _js, _ks;  // Size of space it resides in.
  double       _x1, _y1, _z1;  // Coordinates (in original space) of first point of tetrahedron
  double       _x2, _y2, _z2;  // Coordinates (in original space) of second point of tetrahedron
  double       _x3, _y3, _z3;
  double       _x4, _y4, _z4;
  int          _miter;         // Max iterations, used to detect singularities.
  bool         _ioob;          // If set means we should ignore out-of-bounds
  bool         _hasaff;        // If set means that there is an explicit affine component
  double       _a11, _a12, _a13, _a14;  // First row of affine matrix
  double       _a21, _a22, _a23, _a24;  // Second row of affine matrix
  double       _a31, _a32, _a33, _a34;  // Third row of affine matrix
  std::shared_ptr<const NEWIMAGE::volume4D<float> >   _def; // Deformation field.
};

/////////////////////////////////////////////////////////////////////
//
// Returns true if the point given by x,y,z (in original space)
// is inside the tetrahedron. If it returns false it will also
// return an index to a point that defines which plane it was
// on the outside of. This will be the point of the tetrahedron
// _not_ on that plane.
// Note that it may well be "outside" several of the planes, but
// only the first encountered will be reported in indx.
// To decrease the changes of a tetrahedron getting stuck
// in an area of singularity (in which case the tetrahedron
// gets turned inside-out) the order in which the planes will
// be random.
// If it finds it is outside a plane, but that a step in that
// direction would take it outside the valid FOV it will go to
// the next plane and test that. If it turns out that there is
// direction in which it can go it will return false and set
// indx to -1, indicating that the point out of bounds.
// The FOV checking can be turned off by setting the _ioob (ignore
// out of bounds) flag.
//
/////////////////////////////////////////////////////////////////////
bool Tetrahedron::is_point_in_tetrahedron(double x, double y, double z, int& indx)
{
  indx = (rand() % 4) + 1;  // Random value in range 1-4
  bool one_is_on_right_side = false;
  bool four_is_on_right_side = false;
  bool one_is_ok = false;
  bool four_is_ok = false;

  for (int i=0; i<4; i++) {
    switch (indx) {
    case 1:
      if (!_ioob) {
        if (_i1 != _i2) {
	  int tmp = _i1 + 2*(_i2-_i1);
	  if (tmp>0 && tmp<int(_is)) one_is_ok=true;
	}
	else if (_j1 != _j2) {
	  int tmp = _j1 + 2*(_j2-_j1);
	  if (tmp>0 && tmp<int(_js)) one_is_ok=true;
	}
	else if (_k1 != _k2) {
	  int tmp = _k1 + 2*(_k2-_k1);
	  if (tmp>0 && tmp<int(_ks)) one_is_ok=true;
	}
      }
      else one_is_ok = true;
      one_is_on_right_side = is_point_on_right_side_of_plane(x,y,z,indx);
      if (!one_is_on_right_side && one_is_ok) return(false);
      break;
    case 2:
      if (!is_point_on_right_side_of_plane(x,y,z,indx)) return(false);
      break;
    case 3:
      if (!is_point_on_right_side_of_plane(x,y,z,indx)) return(false);
      break;
    case 4:
      if (!_ioob) {
	if (_i4 != _i3) {
	  int tmp = _i4 + 2*(_i3-_i4);
	  if (tmp>0 && tmp<int(_is)) four_is_ok=true;
	}
	else if (_j4 != _j3) {
	  int tmp = _j4 + 2*(_j3-_j4);
	  if (tmp>0 && tmp<int(_js)) four_is_ok=true;
	}
	else if (_k4 != _k3) {
	  int tmp = _k4 + 2*(_k3-_k4);
	  if (tmp>0 && tmp<int(_ks)) four_is_ok=true;
	}
      }
      else four_is_ok = true;
      four_is_on_right_side = is_point_on_right_side_of_plane(x,y,z,4);
      if (!four_is_on_right_side && four_is_ok) {indx=4; return(false); }
      break;
    default:
      break;
    }
    if (indx==4) indx=1;
    else indx++;
  }

  // If we get here we are either inside, or in a pickle.
  if (one_is_on_right_side && four_is_on_right_side) return(true);
  else indx = -1;
  return(false);
}

/////////////////////////////////////////////////////////////////////
//
// Will "mirror" the point given by index thereby causing the
// tetrahedron to "take a step". The mirroring is done in a line
// for indicies 2 and 3 and in a point for indicies 1 and 4. This
// is so that the tetrahedron shall always retain its shape and
// always have all its vertices on voxel centers.
// If the mirroring would cause the tetrahedron to go outside
// the valid FOV it will _not_ be done and a return value
// of false will be used.
// The FOV checking can be ignored by setting the _ioob flag.
//
/////////////////////////////////////////////////////////////////////
bool Tetrahedron::mirror_tetrahedron(int indx)
{
  switch(indx) {
  case 1:
    if (_i1 != _i2) {
      int tmp = _i1 + 2*(_i2-_i1);
      if (!_ioob && (tmp < 0 || tmp > (int(_is)-1))) return(false);
      else _i1 = tmp;
    }
    else if (_j1 != _j2) {
      int tmp = _j1 + 2*(_j2-_j1);
      if (!_ioob && (tmp < 0 || tmp > (int(_js)-1))) return(false);
      else _j1 = tmp;
    }
    else if (_k1 != _k2) {
      int tmp = _k1 + 2*(_k2-_k1);
      if (!_ioob && (tmp < 0 || tmp > (int(_ks)-1))) return(false);
      else _k1 = tmp;
    }
    else throw TetrahedronException("mirror_tetrahedron::Nämenvanudå 1");
    break;
  case 2:
    if (_i2==_i1 && _i2==_i3) { // If in yz-plane
      if (_j2 == _j1) { _j2=_j3; _k2=_k1; }
      else if (_k2 == _k1) { _k2=_k3; _j2=_j1; }
      else throw TetrahedronException("mirror_tetrahedron::Nämenvanudå 2-x");
    }
    else if (_j2==_j1 && _j2==_j3) { // If in xz-plane
      if (_i2 == _i1) { _i2=_i3; _k2=_k1; }
      else if (_k2 == _k1) { _k2=_k3; _i2=_i1; }
      else throw TetrahedronException("mirror_tetrahedron::Nämenvanudå 2-y");
    }
    else if (_k2==_k1 && _k2==_k3) { // If in xy-plane
      if (_i2 == _i1) { _i2=_i3; _j2=_j1; }
      else if (_j2 == _j1) { _j2=_j3; _i2=_i1; }
      else throw TetrahedronException("mirror_tetrahedron::Nämenvanudå 2-z");
    }
    else throw TetrahedronException("mirror_tetrahedron::Nämenvanudå 2");
    break;
  case 3:
    if (_i3==_i2 && _i3==_i4) { // If in yz-plane
      if (_j3 == _j2) { _j3=_j4; _k3=_k2; }
      else if (_k3 == _k2) { _k3=_k4; _j3=_j2; }
      else throw TetrahedronException("mirror_tetrahedron::Nämenvanudå 3-x");
    }
    else if (_j3==_j2 && _j3==_j4) { // If in xz-plane
      if (_i3 == _i2) { _i3=_i4; _k3=_k2; }
      else if (_k3 == _k2) { _k3=_k4; _i3=_i2; }
      else throw TetrahedronException("mirror_tetrahedron::Nämenvanudå 3-y");
    }
    else if (_k3==_k2 && _k3==_k4) { // If in xy-plane
      if (_i3 == _i2) { _i3=_i4; _j3=_j2; }
      else if (_j3 == _j2) { _j3=_j4; _i3=_i2; }
      else throw TetrahedronException("mirror_tetrahedron::Nämenvanudå 3-z");
    }
    else throw TetrahedronException("mirror_tetrahedron::Nämenvanudå 3");
    break;
  case 4:
    if (_i4 != _i3) {
      int tmp = _i4 + 2*(_i3-_i4);
      if (!_ioob && (tmp < 0 || tmp > (int(_is)-1))) return(false);
      else _i4 = tmp;
    }
    else if (_j4 != _j3) {
      int tmp = _j4 + 2*(_j3-_j4);
      if (!_ioob && (tmp < 0 || tmp > (int(_js)-1))) return(false);
      else _j4 = tmp;
    }
    else if (_k4 != _k3) {
      int tmp = _k4 + 2*(_k3-_k4);
      if (!_ioob && (tmp < 0 || tmp > (int(_ks)-1))) return(false);
      else _k4 = tmp;
    }
    else throw TetrahedronException("mirror_tetrahedron::Nämenvanudå 4");
    break;
  default:
    throw TetrahedronException("mirror_tetrahedron::Invalid index");
    break;
  }
  set_coordinates(indx);
  return(true);
}
/////////////////////////////////////////////////////////////////////
//
// Returns true if the point given by x,y,z is on the "right" side
// of the plane spanned by the three vertices that are _not_
// indicated by indx. N.B. that indx is one-offset.
// It first calculates a normal to the plane and then calculates
// the dot-product of that normal and a vector from one of the
// vertices to the point given by x,y,z. If that dot-product has
// the same sign as the dot-product between the normal and the
// vector to the remaining vertex the point is on the "right" side.
// A dot-product of zero (point is on the plane) counts as "in".
//
/////////////////////////////////////////////////////////////////////
bool Tetrahedron::is_point_on_right_side_of_plane(double x, double y, double z, unsigned int indx)
{
  double nx, ny, nz;   // Normal to plane
  double udot=0.0;     // Dot product with "unknown" point
  double kdot=0.0;     // Dot product with "known" point
  switch(indx) {
  case 1:
    nx=(_y3-_y2)*(_z4-_z2)-(_z3-_z2)*(_y4-_y2);
    ny=(_z3-_z2)*(_x4-_x2)-(_x3-_x2)*(_z4-_z2);
    nz=(_x3-_x2)*(_y4-_y2)-(_y3-_y2)*(_x4-_x2);
    udot=nx*(x-_x2)+ny*(y-_y2)+nz*(z-_z2);
    kdot=nx*(_x1-_x2)+ny*(_y1-_y2)+nz*(_z1-_z2);
    break;
  case 2:
    nx=(_y3-_y1)*(_z4-_z1)-(_z3-_z1)*(_y4-_y1);
    ny=(_z3-_z1)*(_x4-_x1)-(_x3-_x1)*(_z4-_z1);
    nz=(_x3-_x1)*(_y4-_y1)-(_y3-_y1)*(_x4-_x1);
    udot=nx*(x-_x1)+ny*(y-_y1)+nz*(z-_z1);
    kdot=nx*(_x2-_x1)+ny*(_y2-_y1)+nz*(_z2-_z1);
    break;
  case 3:
    nx=(_y2-_y1)*(_z4-_z1)-(_z2-_z1)*(_y4-_y1);
    ny=(_z2-_z1)*(_x4-_x1)-(_x2-_x1)*(_z4-_z1);
    nz=(_x2-_x1)*(_y4-_y1)-(_y2-_y1)*(_x4-_x1);
    udot=nx*(x-_x1)+ny*(y-_y1)+nz*(z-_z1);
    kdot=nx*(_x3-_x1)+ny*(_y3-_y1)+nz*(_z3-_z1);
    break;
  case 4:
    nx=(_y2-_y1)*(_z3-_z1)-(_z2-_z1)*(_y3-_y1);
    ny=(_z2-_z1)*(_x3-_x1)-(_x2-_x1)*(_z3-_z1);
    nz=(_x2-_x1)*(_y3-_y1)-(_y2-_y1)*(_x3-_x1);
    udot=nx*(x-_x1)+ny*(y-_y1)+nz*(z-_z1);
    kdot=nx*(_x4-_x1)+ny*(_y4-_y1)+nz*(_z4-_z1);
    break;
  default:
    break;
  }
  return(!udot || ((udot<0)&&(kdot<0)) || ((udot>0)&&(kdot>0)));
}

/////////////////////////////////////////////////////////////////////
//
// Given a point x,y,z in original space and given that this
// point falls inside the current location of the tetrahedron (in
// original space) it will return the corresponing point xx,yy,zz
// in transformed space.
//
/////////////////////////////////////////////////////////////////////
void Tetrahedron::get_point(double x, double y, double z, double& xx, double& yy, double& zz)
{
  NEWMAT::Matrix X(4,4);           // Coordinates of tetrahedron in original space
  NEWMAT::Real a[] = {1.0,_x1,_y1,_z1,1.0,_x2,_y2,_z2,1.0,_x3,_y3,_z3,1.0,_x4,_y4,_z4};
  X << a;
  NEWMAT::CroutMatrix XX = X;      // Looks weird, but allegedly carries out LU-decomp.

  NEWMAT::Matrix Xp(4,3);          // Coordinates of tetrahedron in transformed space
  NEWMAT::Real b[] = {static_cast<NEWMAT::Real>(_i1), static_cast<NEWMAT::Real>(_j1), static_cast<NEWMAT::Real>(_k1),
                      static_cast<NEWMAT::Real>(_i2), static_cast<NEWMAT::Real>(_j2), static_cast<NEWMAT::Real>(_k2),
                      static_cast<NEWMAT::Real>(_i3), static_cast<NEWMAT::Real>(_j3), static_cast<NEWMAT::Real>(_k3),
                      static_cast<NEWMAT::Real>(_i4), static_cast<NEWMAT::Real>(_j4), static_cast<NEWMAT::Real>(_k4)};
  Xp << b;

  NEWMAT::Matrix B = XX.i() * Xp;  // x' = B(1,1) + B(2,1)*x + B(3,1)*y + B(4,1)*z
  NEWMAT::ColumnVector xo(4);              // 1-augmented coordinates of point of interest in original space
  NEWMAT::Real c[] = {1.0,x,y,z};
  xo << c;
  NEWMAT::ColumnVector xt = B.t()*xo;      // Point of interest in transformed space
  xx = xt(1); yy = xt(2); zz = xt(3);

  return;
}

/////////////////////////////////////////////////////////////////////
//
// Move tetrahedron around until point (in original space) falls
// within the coordinates of the tetrahedron (when warped into
// original space).
//
/////////////////////////////////////////////////////////////////////
bool Tetrahedron::find_tetrahedron(double x, double y, double z)
{
  int indx = 0;
  int iter = 0;
  while (iter < _miter && !is_point_in_tetrahedron(x,y,z,indx)) {
    if (indx < 0) return(false); // Point outside allowed volume
    mirror_tetrahedron(indx);
    iter++;
  }
  if (iter == _miter) return(false);
  return(true);
}

/////////////////////////////////////////////////////////////////////
//
// Translate indicies in warped space to coordinates in original
// space.
//
/////////////////////////////////////////////////////////////////////
void Tetrahedron::set_coordinates(int indx)
{
  switch(indx) {
  case 1:
    if (_hasaff) {
      _x1 = _a11*_i1+_a12*_j1+_a13*_k1+_a14;
      _y1 = _a21*_i1+_a22*_j1+_a23*_k1+_a24;
      _z1 = _a31*_i1+_a32*_j1+_a33*_k1+_a34;
    }
    else { _x1=_i1; _y1=_j1; _z1=_k1; }
    if (_def) { _x1 += (*_def)(_i1,_j1,_k1,0); _y1 += (*_def)(_i1,_j1,_k1,1); _z1 += (*_def)(_i1,_j1,_k1,2); }
    break;
  case 2:
    if (_hasaff) {
      _x2 = _a11*_i2+_a12*_j2+_a13*_k2+_a14;
      _y2 = _a21*_i2+_a22*_j2+_a23*_k2+_a24;
      _z2 = _a31*_i2+_a32*_j2+_a33*_k2+_a34;
    }
    else { _x2=_i2; _y2=_j2; _z2=_k2; }
    if (_def) { _x2 += (*_def)(_i2,_j2,_k2,0); _y2 += (*_def)(_i2,_j2,_k2,1); _z2 += (*_def)(_i2,_j2,_k2,2); }
    break;
  case 3:
    if (_hasaff) {
      _x3 = _a11*_i3+_a12*_j3+_a13*_k3+_a14;
      _y3 = _a21*_i3+_a22*_j3+_a23*_k3+_a24;
      _z3 = _a31*_i3+_a32*_j3+_a33*_k3+_a34;
    }
    else { _x3=_i3; _y3=_j3; _z3=_k3; }
    if (_def) { _x3 += (*_def)(_i3,_j3,_k3,0); _y3 += (*_def)(_i3,_j3,_k3,1); _z3 += (*_def)(_i3,_j3,_k3,2); }
    break;
  case 4:
    if (_hasaff) {
      _x4 = _a11*_i4+_a12*_j4+_a13*_k4+_a14;
      _y4 = _a21*_i4+_a22*_j4+_a23*_k4+_a24;
      _z4 = _a31*_i4+_a32*_j4+_a33*_k4+_a34;
    }
    else { _x4=_i4; _y4=_j4; _z4=_k4; }
    if (_def) { _x4 += (*_def)(_i4,_j4,_k4,0); _y4 += (*_def)(_i4,_j4,_k4,1); _z4 += (*_def)(_i4,_j4,_k4,2); }
    break;
  default:
    break;
  }
}
void Tetrahedron::set_coordinates()
{
  set_coordinates(1);
  set_coordinates(2);
  set_coordinates(3);
  set_coordinates(4);
}
/////////////////////////////////////////////////////////////////////
//
// Populates tetrahedron from an initial point.
// Can be made more clever by allowing population in arbitrary
// directions (all positive now).
//
/////////////////////////////////////////////////////////////////////
void Tetrahedron::populate_me()
{
  if (!_ioob && (_i1<0 || _i1 > int(_is-2) || _j1<0 || _j1 > int(_js-2) || _k1<0 || _k1 > int(_ks-2))) throw TetrahedronException("populate_me::Invalid initial point");
  _i2 = _i1; _j2 = _j1; _k2 = _k1+1;
  _i3 = _i1; _j3 = _j1+1; _k3 = _k1+1;
  _i4 = _i1+1; _j4 = _j1+1; _k4 = _k1+1;
  set_coordinates();
}

void Tetrahedron::set_affine(const NEWMAT::Matrix& aff)
{
  _hasaff = true;
  _a11=aff(1,1); _a12=aff(1,2); _a13=aff(1,3); _a14=aff(1,4);
  _a21=aff(2,1); _a22=aff(2,2); _a23=aff(2,3); _a24=aff(2,4);
  _a31=aff(3,1); _a32=aff(3,2); _a33=aff(3,3); _a34=aff(3,4);
}
} // End namespace FNIRT

#endif // End #ifndef tetrahedron_h
