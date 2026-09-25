/*  unwarpfns.h

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2000 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(__unwarpfns_h)
#define __unwarpfns_h

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"

namespace FUGUE {

  //////////////////////////////////////////////////////////////////////////

  int maximum_position(const NEWMAT::ColumnVector& vec);

  int maximum_position(const NEWMAT::Matrix& vec);

  void delete_row(NEWMAT::Matrix& mat, int r);

  void delete_column(NEWMAT::Matrix& mat, int c);

  void insert_row(NEWMAT::Matrix& mat, int r, const NEWMAT::Matrix& nrow);

  void insert_column(NEWMAT::Matrix& mat, int c, const NEWMAT::Matrix& ncol);

  NEWMAT::ColumnVector pava(const NEWMAT::ColumnVector& data, const NEWMAT::ColumnVector& weights);

  NEWMAT::ColumnVector pava(const NEWMAT::ColumnVector& data);

  //////////////////////////////////////////////////////////////////////////

  template <class T>
  void simple_erode(NEWIMAGE::volume<T>& binaryvol);

  template <class T>
  void simple_dilate(NEWIMAGE::volume<T>& binaryvol);

  NEWIMAGE::volume<float> despike_filter2D(const NEWIMAGE::volume<float>& vol, float threshold);

  NEWIMAGE::volume<float> median_filter2D(const NEWIMAGE::volume<float>& vol);

  NEWIMAGE::volume<float> masked_despike_filter2D(const NEWIMAGE::volume<float>& vol,
                                                  const NEWIMAGE::volume<float>& mask,
                                                  float threshold);

  NEWIMAGE::volume<float> masked_median_filter2D(const NEWIMAGE::volume<float>& vol,
                                                 const NEWIMAGE::volume<float>& mask);

  float basic_mask_threshold(const NEWIMAGE::volume<float>& absmap);

  NEWIMAGE::volume<float> make_basic_head_mask(const NEWIMAGE::volume<float>& absmap,
                                               float thresh);

  NEWIMAGE::volume<float> make_head_mask(const NEWIMAGE::volume<float>& absmap,
                                         float thresh);
  NEWIMAGE::volume<float> make_head_mask2D(const NEWIMAGE::volume<float>& absmap,
                                           float thresh);

  NEWIMAGE::volume<float> make_filled_head_mask(const NEWIMAGE::volume<float>& absmap);

  NEWIMAGE::volume<float> fill_head_mask(const NEWIMAGE::volume<float>& origmask);

  void fill_holes(NEWIMAGE::volume<float>& vol,
                  const NEWIMAGE::volume<float>& holemask,
                  const NEWIMAGE::volume<float>& mask);

  NEWIMAGE::volume<float> extrapolate_volume(const NEWIMAGE::volume<float>& datavol,
                                             const NEWIMAGE::volume<float>& origmask,
                                             const NEWIMAGE::volume<float>& enlargedmask);

  NEWIMAGE::volume<float> polynomial_extrapolate(const NEWIMAGE::volume<float>& datavol,
                                                 const NEWIMAGE::volume<float>& validmask,
                                                 int n, bool verbose=false);

  NEWIMAGE::volume<float> fourier_extrapolate(const NEWIMAGE::volume<float>& datavol,
                                              const NEWIMAGE::volume<float>& validmask,
                                              int n, bool verbose=false);

  void connection_matrices(const NEWIMAGE::volume<float>& phasemap,
                           const NEWIMAGE::volume<int>& label,
                           NEWMAT::Matrix& Qab,
                           NEWMAT::Matrix& Pab,
                           NEWMAT::Matrix& Nab,
                           double& K);

  void connection_matrices(const NEWIMAGE::volume<float>& phasemap,
                           const NEWIMAGE::volume<int>& label,
                           NEWMAT::Matrix& Kab);

  float calc_cost(const NEWMAT::Matrix& Mab,
                  const NEWMAT::Matrix& Qab,
                  const NEWMAT::Matrix& Pvec,
                  float K);

  float calc_delta_cost(const NEWMAT::Matrix& Mab,
                        const NEWMAT::Matrix& Qab,
                        const NEWMAT::Matrix& Pvec,
                        int j, float d);

  float calc_delta_cost(const NEWMAT::Matrix& Mab,
                        const NEWMAT::Matrix& Qab,
                        const NEWMAT::Matrix& Pvecon2PI,
                        int signedj);

  float wrap(float theta);

  void remove_linear_ramps(const NEWMAT::ColumnVector& ramp,
                           NEWIMAGE::volume<float>& ph,
                           const NEWIMAGE::volume<float>& mask);

  void restore_linear_ramps(const NEWMAT::ColumnVector& ramp,
                            NEWIMAGE::volume<float>& uph,
                            const NEWIMAGE::volume<float>& mask);

  NEWMAT::ColumnVector estimate_linear_ramps(const NEWIMAGE::volume<float>& ph,
                                             const NEWIMAGE::volume<float>& mask);

  ////////////////////////////////////////////////////////////////////////////

  NEWIMAGE::volume<float> apply_unwrapping(const NEWIMAGE::volume<float>& phasemap,
                                           const NEWIMAGE::volume<int>& label,
                                           const NEWMAT::Matrix& label_offsets);

  NEWIMAGE::volume<int> find_phase_labels(const NEWIMAGE::volume<float>& phasemap,
                                          const NEWIMAGE::volume<float>& mask,
                                          int n=4);

  NEWIMAGE::volume<int> find_phase_labels2D(const NEWIMAGE::volume<float>& phasemap,
                                            const NEWIMAGE::volume<float>& mask,
                                            int n, bool unique3Dlabels);

  NEWIMAGE::volume<float> unwrap(const NEWIMAGE::volume<float>& phasemap,
                                 const NEWIMAGE::volume<int>& label,
                                 bool verbose=false);

  NEWIMAGE::volume<float> unwrap2D(const NEWIMAGE::volume<float>& phasemap,
                                   const NEWIMAGE::volume<int>& label,
                                   bool verbose=false);

  NEWIMAGE::volume<float> calc_fmap(const NEWIMAGE::volume<float>& phase1,
                                    const NEWIMAGE::volume<float>& phase2,
                                    const NEWIMAGE::volume<float>& mask1,
                                    const NEWIMAGE::volume<float>& mask2,
                                    float asym_time);

  float fmap2pixshift_factor(const NEWIMAGE::volume<float>& fmap,
                             float pe_dwell_time,
                             const std::string& dir);

  NEWIMAGE::volume<float> yderiv(const NEWIMAGE::volume<float>& vol);

  NEWIMAGE::volume<float> limit_pixshift(const NEWIMAGE::volume<float>& pixshift,
                                         const NEWIMAGE::volume<float>& mask,
                                         float mindiff);

  NEWIMAGE::volume<float> apply_pixshift(const NEWIMAGE::volume<float>& epivol,
                                         const NEWIMAGE::volume<float>& pixshiftmap);

  ////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////


  // TEMPLATE BODIES

  template <class T>
  void simple_erode(NEWIMAGE::volume<T>& binaryvol)
  {
    binaryvol.setpadvalue(1.0);
    binaryvol.setextrapolationmethod(NEWIMAGE::constpad);
    NEWMAT::ColumnVector structelement(3);
    structelement = 1.0/3.0;
    binaryvol = convolve_separable(binaryvol,structelement,
                                   structelement,structelement);
    binaryvol.binarise(1.0 - 0.5/27.0);
  }

  template <class T>
  void simple_dilate(NEWIMAGE::volume<T>& binaryvol)
  {
    binaryvol.setpadvalue(0.0);
    binaryvol.setextrapolationmethod(NEWIMAGE::constpad);
    NEWMAT::ColumnVector structelement(3);
    structelement = 1.0/3.0;
    binaryvol = convolve_separable(binaryvol,structelement,
                                   structelement,structelement);
    binaryvol.binarise(0.5/27.0,100.0);
  }

}
#endif
