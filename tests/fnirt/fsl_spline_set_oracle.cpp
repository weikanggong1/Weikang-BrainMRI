// Optional FSL 6.0.7.4 oracle for splinefield::Set dense-to-coefficient fit.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "basisfield/splinefield.h"
#include "newimage/newimageall.h"

int main() {
  std::vector<unsigned int> shape{8, 9, 7};
  std::vector<double> voxels{2.0, 2.5, 3.0};
  std::vector<unsigned int> spacing{3, 3, 3};
  NEWIMAGE::volume<float> dense(shape[0], shape[1], shape[2]);
  dense.setdims(voxels[0], voxels[1], voxels[2]);
  for (unsigned int z = 0; z < shape[2]; ++z) {
    for (unsigned int y = 0; y < shape[1]; ++y) {
      for (unsigned int x = 0; x < shape[0]; ++x) {
        dense(x, y, z) = static_cast<float>(
            std::sin(0.13 * (x + 1)) + std::cos(0.09 * (y + 2))
            - 0.04 * z + 0.002 * x * y);
      }
    }
  }
  BASISFIELD::splinefield field(shape, voxels, spacing, 3);
  field.Set(dense);
  std::shared_ptr<NEWMAT::ColumnVector> coefficients = field.GetCoef();
  std::cout << field.CoefSz_x() << " " << field.CoefSz_y() << " "
            << field.CoefSz_z() << "\n";
  for (int index = 0; index < coefficients->Nrows(); ++index) {
    std::cout << std::setprecision(17) << coefficients->element(index) << "\n";
  }
  return 0;
}
