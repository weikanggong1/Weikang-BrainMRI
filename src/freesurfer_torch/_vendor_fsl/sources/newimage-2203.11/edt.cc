// Matthew Webster, WIN Analysis Group
//Copyright (C) 2022 University of Oxford
/*  CCOPYRIGHT  */

#include "newimagefns.h"
#include "newimageio.h"
#include "edt.h"

using namespace std;

namespace NEWIMAGE {
  //Euclidean distance transform of a 1D function f
  //All elements of f should be real, but safety features have been added to handle Inf elements
  //For images, weight should be pixdim^2
  //aux is auxillary function ( e.g. for sparse interpolation )
  template<class AUX>
  vector<double> euclideanDistanceTransform(const vector<double>& f, const float weight, AUX&& aux) {
    int64_t n(f.size());
    vector<double> d(n), newAuxillary(n); //distance metric, aux output
    vector<int64_t> v(n,0);   //Only v[0] _needs_ to be 0
    vector<double> z(n+1,numeric_limits<double>::infinity()); //Only z[1] _needs_ to be inf
    z[0] = -numeric_limits<double>::infinity();
    for (int64_t q(1), k(0); q < n; q++) {
      double s(intersection(f,q,v[k],weight));
      while ( k > 0 && s <= z[k] ) //Safety addition: test k > 0
        s = intersection(f,q,v[--k],weight);
      v[++k] = q;
      z[k] = s;
      z[k+1] = numeric_limits<float>::infinity();
    }
    for (int64_t q(0), k(0); q < n; q++) {
      while (z[k+1] < q)
        k++;
      d[q] = ( weight*(q - v[k])*(q - v[k]) ) + f[v[k]];
      if ( !aux.empty() )
        newAuxillary[q]=aux[v[k]];
    }
    if ( !aux.empty() )
      aux=newAuxillary;
    return d;
  }

  template <class T>
  volume<double> euclideanDistanceTransform(const NEWIMAGE::volume<T>& input) {
    auto[ distanceMap, temp ] = euclideanDistanceTransform(input,input);
    return distanceMap;
  }

  template <typename T>
  tuple<volume<double>,volume<T>> euclideanDistanceTransform(const volume<T>& binary, const volume<T>& data) {
    binary.throwsIfNot3D();
    volume<double> distanceMap;
    volume<T> nearestData(data);
    copyconvert(binary,distanceMap);

    //"0D" initialisation
    for (int z=0; z<=distanceMap.maxz(); z++)
      for (int y=0; y<=distanceMap.maxy(); y++)
        for (int x=0; x<=distanceMap.maxx(); x++)
          distanceMap(x,y,z) = ( distanceMap(x,y,z) == 0) ? std::numeric_limits<float>::infinity() : 0;

    //1D:X
    vector<double> f(distanceMap.xsize());
    vector<double> aux(distanceMap.xsize());
    for (int z=0; z<=distanceMap.maxz(); z++)
      for (int y=0; y<=distanceMap.maxy(); y++) {
        for (int x=0; x<=distanceMap.maxx(); x++) {
          f[x] = distanceMap(x,y,z);
          aux[x] = nearestData(x,y,z);
        }
        vector<double> dvec( euclideanDistanceTransform(f,distanceMap.xdim()*distanceMap.xdim(),aux));
        for (int x=0; x<=distanceMap.maxx(); x++) {
          distanceMap(x,y,z) = dvec[x];
          nearestData(x,y,z) = aux[x];
        }
      }

    //2D:Y
    f.resize(distanceMap.ysize());
    aux.resize(distanceMap.ysize());
    for (int z=0; z<=distanceMap.maxz(); z++)
      for (int x=0; x<=distanceMap.maxx(); x++) {
        for (int y=0; y<=distanceMap.maxy(); y++) {
          f[y] = distanceMap(x,y,z);
          aux[y] = nearestData(x,y,z);
        }
        vector<double> dvec( euclideanDistanceTransform(f,distanceMap.ydim()*distanceMap.ydim(),aux));
        for (int y=0; y<=distanceMap.maxy(); y++) {
          distanceMap(x,y,z) = dvec[y];
          nearestData(x,y,z) = aux[y];
        }
      }

    //3D:Z
    f.resize(distanceMap.zsize());
    aux.resize(distanceMap.zsize());
    for (int x=0; x<=distanceMap.maxx(); x++)
      for (int y=0; y<=distanceMap.maxy(); y++) {
        for (int z=0; z<=distanceMap.maxz(); z++) {
          f[z] = distanceMap(x,y,z);
          aux[z] = nearestData(x,y,z);
        }
        vector<double> dvec( euclideanDistanceTransform(f,distanceMap.zdim()*distanceMap.zdim(),aux) );
        for (int z=0; z<=distanceMap.maxz(); z++) {
          distanceMap(x,y,z) = dvec[z];
          nearestData(x,y,z) = aux[z];
        }
      }

    return { distanceMap, nearestData };
  }

  //From a set of sparse data, defined by isSample > 0
  template<class T>
  volume<T> sparseInterpolate(const volume<T>& sparseData, const volume<T>& isSample, const double sigma) {

    if ( sparseData.tsize() > 1 ) {
      volume<T> iData(sparseData,TEMPLATE);
      for (int64_t t=0; t < sparseData.tsize(); t++)
        iData[t] = sparseInterpolate(sparseData[t],isSample[t],sigma);
      return iData;
    }

    volume<T> dmap, iData;
    tie(dmap,iData) = euclideanDistanceTransform(isSample,sparseData);

    auto kernel=gaussian_kernel3D(sigma,iData.xdim(),iData.ydim(),iData.zdim());
    if ( sigma > 0 )
      iData=generic_convolve(iData,kernel,true,true);
    for (int64_t z(0); z < iData.zsize(); z++)
      for (int64_t y(0); y < iData.ysize(); y++)
        for (int64_t x(0); x < iData.xsize(); x++)
              iData(x,y,z) = isSample(x,y,z) ? sparseData(x,y,z) : iData(x,y,z);
    return iData;
  }

  template<typename... Ts>
  auto edtInstantiate() {
    static auto instantiatedFunctions = std::tuple_cat(std::make_tuple(
      static_cast< volume<double>(*)(const volume<Ts>&)>(euclideanDistanceTransform<Ts>),
      static_cast< tuple<volume<double>,volume<Ts>>(*)(const volume<Ts>&, const volume<Ts>&)>(euclideanDistanceTransform<Ts>)
    )...);
    return &instantiatedFunctions;
  }

  template auto __attribute__((visibility("hidden"))) edtInstantiate<char,short,int,float,double>();

  template volume<double> sparseInterpolate(const volume<double>& data, const volume<double>& weights, const double sigma);
}