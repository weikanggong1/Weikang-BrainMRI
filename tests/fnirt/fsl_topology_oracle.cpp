// Optional FSL 6.0.7.4 oracle for the PyTorch topology projection tests.
#include <cstdlib>
#include <iostream>

#include "newimage/newimageall.h"
#include "warpfns/warpfns.h"

namespace NEWIMAGE {
void grad_calc(volume4D<float>& grad, const volume4D<float>& warp);
void limit_grad(
    volume4D<float>& grad, const volume4D<float>& jacobians,
    float min_jacobian, float max_jacobian);
void integrate_gradient_field(
    volume4D<float>& warp, const volume4D<float>& grad,
    float mean_x, float mean_y, float mean_z);
}

int main(int argc, char** argv) {
  if (argc != 5 && argc != 6) {
    std::cerr << "usage: fsl_topology_oracle input output min_jac max_jac "
                 "[intermediate_prefix]\n";
    return 2;
  }
  NEWIMAGE::volume4D<float> warp;
  NEWIMAGE::read_volume4D(warp, argv[1]);
  const float minimum = std::strtof(argv[3], nullptr);
  const float maximum = std::strtof(argv[4], nullptr);
  if (argc == 6) {
    NEWIMAGE::volume4D<float> gradient;
    NEWIMAGE::grad_calc(gradient, warp);
    NEWIMAGE::volume4D<float> jacobians;
    NEWMAT::ColumnVector stats;
    NEWIMAGE::jacobian_check(
        jacobians, stats, warp, minimum, maximum, true);
    NEWIMAGE::save_volume4D(gradient, std::string(argv[5]) + "_gradient");
    NEWIMAGE::save_volume4D(jacobians, std::string(argv[5]) + "_jacobians");
    NEWIMAGE::limit_grad(gradient, jacobians, minimum, maximum);
    NEWIMAGE::save_volume4D(gradient, std::string(argv[5]) + "_limited");
    NEWIMAGE::volume4D<float> once = warp;
    NEWIMAGE::integrate_gradient_field(
        once, gradient, warp[0].mean(), warp[1].mean(), warp[2].mean());
    NEWIMAGE::save_volume4D(once, std::string(argv[5]) + "_once");
  }
  NEWIMAGE::constrain_topology(warp, minimum, maximum);
  NEWIMAGE::save_volume4D(warp, argv[2]);
  return 0;
}
