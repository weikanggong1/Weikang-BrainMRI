// Numerical fixture generator for FSL 6.0.7.4 basisfield 2203.1,
// commit 9588bbe8eb8aa0939ddefd00df756aeb80d2305b.
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "basisfield/splinefield.h"
#include "warpfns/fnirt_file_writer.h"

using BASISFIELD::splinefield;

static NEWMAT::ColumnVector make_coefficients(unsigned int count) {
  NEWMAT::ColumnVector values(count);
  for (unsigned int i = 0; i < count; ++i) {
    values.element(i) = std::sin(0.17 * static_cast<double>(i + 1))
                      + 0.03 * static_cast<double>(static_cast<int>(i % 7) - 3);
  }
  return values;
}

static void print_vector(const char* name, const NEWMAT::ColumnVector& values) {
  std::cout << name << " " << values.Nrows();
  for (int i = 0; i < values.Nrows(); ++i) {
    std::cout << " " << std::setprecision(17) << values.element(i);
  }
  std::cout << "\n";
}

int main() {
  {
    std::vector<unsigned int> shape{7, 8, 6};
    std::vector<double> voxels{4.0, 4.0, 4.0};
    std::vector<unsigned int> spacing{3, 3, 3};
    splinefield old_field(shape, voxels, spacing, 3);
    NEWMAT::ColumnVector coefficients = make_coefficients(old_field.CoefSz());
    old_field.SetCoef(coefficients);
    std::vector<unsigned int> new_shape{13, 15, 11};
    std::vector<double> new_voxels{2.0, 2.0, 2.0};
    std::shared_ptr<BASISFIELD::basisfield> zoomed =
        old_field.ZoomField(new_shape, new_voxels);
    std::cout << "ZOOM_SHAPE " << zoomed->CoefSz_x() << " "
              << zoomed->CoefSz_y() << " " << zoomed->CoefSz_z() << "\n";
    print_vector("ZOOM_COEF", *zoomed->GetCoef());
  }
  {
    std::vector<unsigned int> shape{8, 9, 7};
    std::vector<double> voxels{2.0, 2.5, 3.0};
    std::vector<unsigned int> spacing{3, 3, 3};
    splinefield field(shape, voxels, spacing, 3);
    NEWMAT::ColumnVector coefficients = make_coefficients(field.CoefSz());
    field.SetCoef(coefficients);
    std::cout << "BEND_ENERGY " << std::setprecision(17)
              << field.BendEnergy() << "\n";
    print_vector("BEND_GRAD", field.BendEnergyGrad());
  }
  {
    std::vector<unsigned int> shape{9, 10, 8};
    std::vector<double> voxels{2.0, 2.5, 3.0};
    std::vector<unsigned int> spacing{3, 2, 2};
    splinefield field_x(shape, voxels, spacing, 3);
    splinefield field_y(shape, voxels, spacing, 3);
    splinefield field_z(shape, voxels, spacing, 3);
    NEWMAT::ColumnVector coefficients = make_coefficients(field_x.CoefSz());
    field_x.SetCoef(coefficients);
    field_y.SetCoef(2.0 * coefficients);
    field_z.SetCoef(-0.5 * coefficients);
    NEWMAT::Matrix affine = NEWMAT::IdentityMatrix(4);
    affine(1, 1) = 1.1;
    affine(2, 2) = 0.9;
    affine(3, 3) = 1.2;
    affine(1, 4) = 4.0;
    affine(2, 4) = -3.0;
    affine(3, 4) = 2.0;
    NEWIMAGE::FnirtFileWriter(
        "/tmp/fsl_coefficient_2203_0_oracle.nii.gz",
        field_x,
        field_y,
        field_z,
        affine
    );
  }
  return 0;
}
