// Matthew Webster, WIN Analysis Group
//Copyright (C) 2022 University of Oxford
/*  CCOPYRIGHT  */

#include <limits>
#include <utility>
#include <vector>

#include "newimage.h"

namespace NEWIMAGE {

  inline double intersection(const std::vector<double>& f, const int64_t& q, const int64_t& vk, const double weight=1) {
    double diff = (q - vk) * weight;
    double sum = q + vk;
    double intersection = (f[q] - f[vk] + (diff * sum)) / (2 * diff);
    //Safety addition: isnan(intersection)
    if ( isnan(intersection) )
      return std::numeric_limits<double>::infinity();
    return intersection;
  }

  template<class AUX=std::vector<double>>
  std::vector<double> edt(const std::vector<double>& f, const float weight=1, AUX&& aux={});

  template <class T>
  NEWIMAGE::volume<double> euclideanDistanceTransform(const NEWIMAGE::volume<T>& input);

  template <typename T>
  std::tuple<NEWIMAGE::volume<double>,NEWIMAGE::volume<T>> euclideanDistanceTransform(const NEWIMAGE::volume<T>& binary, const NEWIMAGE::volume<T>& data);

  template<class T>
  NEWIMAGE::volume<T> sparseInterpolate(const NEWIMAGE::volume<T>& data, const NEWIMAGE::volume<T>& isSample, const double sigma=3);
}