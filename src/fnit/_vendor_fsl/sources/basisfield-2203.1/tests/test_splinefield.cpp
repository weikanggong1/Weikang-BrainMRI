#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <armadillo>
#include "utils/threading.h"
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "basisfield/fsl_splines.h"
#include "basisfield/splinefield.h"

using namespace std;
using namespace Utilities;
using namespace NEWMAT;
using namespace MISCMATHS;
using namespace NEWIMAGE;
using namespace BASISFIELD;

std::string format_mp(const NEWMAT::ColumnVector& mp)
{
  char str[256];
  sprintf(str,"xt = %5.2fmm, yt = %5.2fmm, zt = %5.2fmm, xr = %5.2fdeg, yr = %5.2fdeg, zr = %5.2fdeg",mp(1),mp(2),mp(3),180*mp(4)/M_PI,180*mp(5)/M_PI,180*mp(6)/M_PI);
  return(std::string(str));
}

int main(int    argc,
         char   *argv[])
{

  int seed = 0;
  if ((argc) > 1)  {
    seed = atoi(argv[1]);
  }
  /*
  NEWMAT::ColumnVector vec(6);
  vec(1) = 3.54567;
  vec(2) = 7.52345;
  vec(3) = 13.67524;
  vec(4) = 0.027534;
  vec(5) = 0.09472548;
  vec(6) = 0.13057463;
  cout << "vec = " << format_mp(vec) << endl;
  */

  NEWIMAGE::volume<float> full_vol, vol;
  string fsldir = getenv("FSLDIR");
  NEWIMAGE::read_volume(full_vol, fsldir + "/data/standard/MNI152_T1_2mm.nii.gz");

  // use a ROI to keep RAM/runtime down
  vol = full_vol.ROI(25, 30, 25, 0, 65, 70, 65, 0);

  std::vector<unsigned int> isz(3,0);
  isz[0]=static_cast<unsigned int>(vol.xsize());
  isz[1]=static_cast<unsigned int>(vol.ysize());
  isz[2]=static_cast<unsigned int>(vol.zsize());
  std::vector<double> vxs(3,0.0);
  vxs[0]=static_cast<double>(vol.xdim());
  vxs[1]=static_cast<double>(vol.ydim());
  vxs[2]=static_cast<double>(vol.zdim());
  std::vector<unsigned int> pksp(3,4);

  std::chrono::duration<double> tse1 = std::chrono::system_clock::now().time_since_epoch();
  BASISFIELD::splinefield spf(isz,vxs,pksp,3);
  std::chrono::duration<double> tse2 = std::chrono::system_clock::now().time_since_epoch();
  BASISFIELD::splinefield pspf(isz,vxs,pksp,3,NoOfThreads(8));
  BASISFIELD::splinefield pspf4(isz,vxs,pksp,3,NoOfThreads(4));
  BASISFIELD::splinefield pspf16(isz,vxs,pksp,3,NoOfThreads(16));

  std::vector<unsigned int> deriv1 = {0,0,0};
  std::vector<unsigned int> deriv2 = {1,0,0};

  // Testing asymmetric JtJ (the relatively easy problem)

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  std::shared_ptr<BFMatrix> my_matrix = spf.JtJ(deriv1,vol,deriv2,vol,nullptr,BFMatrixDoublePrecision);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of asymmetric JtJ took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  my_matrix->Print("matrix_asym_single_thread.txt");

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  std::shared_ptr<BFMatrix> my_para_matrix = pspf.JtJ(deriv1,vol,deriv2,vol,nullptr,BFMatrixDoublePrecision);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Multi threaded version of asymmetric JtJ took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  my_para_matrix->Print("matrix_asym_8_thread.txt");

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  std::shared_ptr<BFMatrix> my_para_matrix_4 = pspf4.JtJ(deriv1,vol,deriv2,vol,nullptr,BFMatrixDoublePrecision);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Multi threaded version of asymmetric JtJ 4 took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  my_para_matrix_4->Print("matrix_asym_4_thread.txt");

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  std::shared_ptr<BFMatrix> my_para_matrix_16 = pspf16.JtJ(deriv1,vol,deriv2,vol,nullptr,BFMatrixDoublePrecision);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Multi threaded version of asymmetric JtJ 16 took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  my_para_matrix_16->Print("matrix_asym_16_thread.txt");


  // Testing symmetric JtJ (the relatively harder problem)

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  std::shared_ptr<BFMatrix> my_sym_matrix = spf.JtJ(vol,nullptr,BFMatrixDoublePrecision);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of symmetric JtJ took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  std::cout << "my_sym_matrix->Nrows() = " << my_sym_matrix->Nrows() << ", my_sym_matrix->Ncols() = " << my_sym_matrix->Ncols() << ", my_sym_matrix->NZ() = " << my_sym_matrix->NZ() << endl;
  cout << "Writing my_sym_matrix" << endl;
  my_sym_matrix->Print("matrix_sym_single_thread.txt");
  cout << "Finished writing my_sym_matrix" << endl;

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  std::shared_ptr<BFMatrix> my_para_sym_matrix = pspf.JtJ(vol,nullptr,BFMatrixDoublePrecision);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Multi threaded version of symmetric JtJ took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  std::cout << "my_para_sym_matrix->Nrows() = " << my_para_sym_matrix->Nrows() << ", my_para_sym_matrix->Ncols() = " << my_para_sym_matrix->Ncols() << ", my_para_sym_matrix->NZ() = " << my_para_sym_matrix->NZ() << endl;
  cout << "Writing my_para_sym_matrix" << endl;
  my_para_sym_matrix->Print("matrix_sym_multi_thread_8.txt");
  cout << "Finished writing my_para_sym_matrix" << endl;

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  std::shared_ptr<BFMatrix> my_para_sym_matrix_4 = pspf4.JtJ(vol,nullptr,BFMatrixDoublePrecision);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Multi threaded version of symmetric JtJ 4 took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  my_para_sym_matrix_4->Print("matrix_sym_multi_thread_4.txt");

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  std::shared_ptr<BFMatrix> my_para_sym_matrix_16 = pspf16.JtJ(vol,nullptr,BFMatrixDoublePrecision);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Multi threaded version of symmetric JtJ 16 took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  my_para_sym_matrix_16->Print("matrix_sym_multi_thread_16.txt");

  // Doing some timing tests on other functions of splinefield

  std::vector<unsigned int> deriv = {1,0,0};

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector jte1 = spf.Jte(vol,vol,nullptr);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of jte1 took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;

  MISCMATHS::write_ascii_matrix(jte1, "jte1.txt");

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector jte2 = spf.Jte(deriv,vol,vol,nullptr);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of jte2 took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix(jte2, "jte2.txt");

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector jte3 = spf.Jte(vol,nullptr);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of jte3 took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix(jte3, "jte3.txt");

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector jte4 = spf.Jte(deriv,vol,nullptr);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of jte4 took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix(jte4, "jte4.txt");

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector pjte1 = pspf.Jte(vol,vol,nullptr);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of jte1 took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix(pjte1, "pjte1.txt");

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector pjte2 = pspf.Jte(deriv,vol,vol,nullptr);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of jte2 took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix(pjte2, "pjte2.txt");

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector pjte3 = pspf.Jte(vol,nullptr);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of jte3 took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix(pjte3, "pjte3.txt");

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector pjte4 = pspf.Jte(deriv,vol,nullptr);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of jte4 took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix(pjte4, "pjte4.txt");

  for (int i=0; i<jte1.Nrows(); i++) {
    if (jte1(i+1) != pjte1(i+1)) std::cout << "error in jte1" << std::endl;
    if (jte2(i+1) != pjte2(i+1)) std::cout << "error in jte1" << std::endl;
    if (jte3(i+1) != pjte3(i+1)) std::cout << "error in jte1" << std::endl;
    if (jte4(i+1) != pjte4(i+1)) std::cout << "error in jte1" << std::endl;
  }

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  double my_mem_energy = spf.MemEnergy();
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of MemEnergy took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  double my_bend_energy = spf.BendEnergy();
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of BendEnergy took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector my_mem_energy_grad = spf.MemEnergyGrad();
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of MemEnergyGrad took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector my_bend_energy_grad = spf.BendEnergyGrad();
  tse2 = std::chrono::system_clock::now().time_since_epoch();


  tse1 = std::chrono::system_clock::now().time_since_epoch();
  std::shared_ptr<BFMatrix> my_mem_energy_hess = spf.MemEnergyHess(BFMatrixDoublePrecision);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of MemEnergyHess took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  my_mem_energy_hess->Print("Mem_energy_hess_single_thread.txt");

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  std::shared_ptr<BFMatrix> my_bend_energy_hess = spf.BendEnergyHess(BFMatrixDoublePrecision);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of BendEnergyHess took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  my_bend_energy_hess->Print("Bend_energy_hess_single_thread.txt");

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  std::shared_ptr<BFMatrix> p_my_mem_energy_hess = pspf.MemEnergyHess(BFMatrixDoublePrecision);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Multi threaded version of MemEnergyHess took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  p_my_mem_energy_hess->Print("Mem_energy_hess_multi_thread.txt");

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  std::shared_ptr<BFMatrix> p_my_bend_energy_hess = pspf.BendEnergyHess(BFMatrixDoublePrecision);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Multi threaded version of BendEnergyHess took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  p_my_bend_energy_hess->Print("Bend_energy_hess_multi_thread.txt");

  if (seed == 0) {
    arma::arma_rng::set_seed_random();
  }
  else {
     arma::arma_rng::set_seed(seed);
  }
  arma::Col<double> ab(my_matrix->Ncols(),arma::fill::randu);
  NEWMAT::ColumnVector b = ab;

  MISCMATHS::write_ascii_matrix("xdata.txt",b);

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector x1 = my_matrix->MulByVec(b);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of matrix-by-vector multiplication took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix("x1.txt",x1);

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector x2 = my_para_matrix->MulByVec(b);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Multi threaded version of matrix-by-vector multiplication took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix("x2.txt",x2);

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector x3 = my_sym_matrix->MulByVec(b);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of sym-matrix-by-vector multiplication took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix("x3.txt",x3);

  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector x4 = my_para_sym_matrix->MulByVec(b);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Multi threaded version of sym-matrix-by-vector multiplication took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix("x4.txt",x4);

  my_sym_matrix->SetNthreads(4);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector x5 = my_sym_matrix->MulByVec(b);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Multi threaded version of single threaded sym-matrix-by-vector multiplication took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix("x5.txt",x5);

  my_para_sym_matrix->SetNthreads(1);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector x6 = my_para_sym_matrix->MulByVec(b);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of multi threaded sym-matrix-by-vector multiplication took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix("x6.txt",x6);

  MISCMATHS::SparseBFMatrix<double> *sdp = dynamic_cast<SparseBFMatrix<double> *>(&(*my_para_sym_matrix));
  MISCMATHS::SpMat<double> spmat = sdp->AsSpMat();

  spmat.SetNthreads(1);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector x7 = spmat.TransMult(b);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded version of sym-matrix-transpose-by-vector multiplication took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix("x7.txt",x7);

  spmat.SetNthreads(2);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector x8 = spmat.TransMult(b);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Double threaded version of sym-matrix-transpose-by-vector multiplication took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix("x8.txt",x8);

  spmat.SetNthreads(4);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector x9 = spmat.TransMult(b);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Four threaded version of sym-matrix-transpose-by-vector multiplication took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix("x9.txt",x9);

  spmat.SetNthreads(8);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  NEWMAT::ColumnVector x10 = spmat.TransMult(b);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Eight threaded version of sym-matrix-transpose-by-vector multiplication took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  MISCMATHS::write_ascii_matrix("x10.txt",x10);

  MISCMATHS::SpMat<double> spmat_t = spmat.t();
  spmat.SetNthreads(1);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  MISCMATHS::SpMat<double> prod = spmat_t * spmat;
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded spmat.t()*spmat took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  prod.Print("prod.txt");

  spmat.SetNthreads(2);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  MISCMATHS::SpMat<double> prod2 = spmat_t * spmat;
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Double threaded spmat.t()*spmat took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  prod2.Print("prod2.txt");

  spmat.SetNthreads(4);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  MISCMATHS::SpMat<double> prod4 = spmat_t * spmat;
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Four threaded spmat.t()*spmat took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  prod4.Print("prod4.txt");

  spmat.SetNthreads(7);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  MISCMATHS::SpMat<double> prod7 = spmat_t * spmat;
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Seven threaded spmat.t()*spmat took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  prod7.Print("prod7.txt");
  if (prod == prod7) cout << "prod == prod7" << endl;
  prod7.Set(1,1,1.0);
  if (prod == prod7) cout << "prod == prod7" << endl;

  // Get two SpMat objects with same sparsity
  MISCMATHS::SparseBFMatrix<double> *sdp2 = dynamic_cast<SparseBFMatrix<double> *>(&(*my_sym_matrix));
  MISCMATHS::SpMat<double> sym_matrix_spmat = sdp2->AsSpMat();
  MISCMATHS::SparseBFMatrix<double> *sdp3 = dynamic_cast<SparseBFMatrix<double> *>(&(*my_bend_energy_hess));
  MISCMATHS::SpMat<double> bend_energy_hess_spmat = sdp3->AsSpMat();

  // Do some testing
  MISCMATHS::SpMat<double> sym_matrix_spmat_1 = sym_matrix_spmat;
  sym_matrix_spmat_1.SetNthreads(1);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  sym_matrix_spmat_1 += bend_energy_hess_spmat;
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded sym_matrix_spmat_1 += bend_energy_hess_spmat; took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;

  MISCMATHS::SpMat<double> sym_matrix_spmat_4 = sym_matrix_spmat;
  sym_matrix_spmat_4.SetNthreads(4);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  sym_matrix_spmat_4 += bend_energy_hess_spmat;
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Four threaded sym_matrix_spmat_4 += bend_energy_hess_spmat; took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;

  if (sym_matrix_spmat_1 == sym_matrix_spmat_4) cout << "sym_matrix_spmat_1 == sym_matrix_spmat_4" << endl;


  bend_energy_hess_spmat.Set(1,500,1.0);
  sym_matrix_spmat_1 = sym_matrix_spmat;
  sym_matrix_spmat_1.SetNthreads(1);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  sym_matrix_spmat_1 += bend_energy_hess_spmat;
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Single threaded sym_matrix_spmat_1 += bend_energy_hess_spmat; took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;

  sym_matrix_spmat_4 = sym_matrix_spmat;
  sym_matrix_spmat_4.SetNthreads(4);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  sym_matrix_spmat_4 += bend_energy_hess_spmat;
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "Four threaded sym_matrix_spmat_4 += bend_energy_hess_spmat; took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;
  sym_matrix_spmat_4.AddTo(1,500,0.0001);

  if (sym_matrix_spmat_1 == sym_matrix_spmat_4) cout << "sym_matrix_spmat_1 == sym_matrix_spmat_4" << endl;


  my_bend_energy_hess->Set(1,500,10.0);
  my_sym_matrix->SetNthreads(1);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  my_sym_matrix->AddToMe(*my_bend_energy_hess);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "my_sym_matrix->AddToMe(*my_bend_energy_hess) took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;



  my_bend_energy_hess->Set(1,501,10.0);
  my_sym_matrix->SetNthreads(2);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  my_sym_matrix->AddToMe(*my_bend_energy_hess);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "my_sym_matrix(2)->AddToMe(*my_bend_energy_hess) took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;

  my_bend_energy_hess->Set(1,502,10.0);
  my_sym_matrix->SetNthreads(4);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  my_sym_matrix->AddToMe(*my_bend_energy_hess);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "my_sym_matrix(4)->AddToMe(*my_bend_energy_hess) took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;

  my_bend_energy_hess->Set(1,503,10.0);
  my_sym_matrix->SetNthreads(8);
  tse1 = std::chrono::system_clock::now().time_since_epoch();
  my_sym_matrix->AddToMe(*my_bend_energy_hess);
  tse2 = std::chrono::system_clock::now().time_since_epoch();
  std::cout << "my_sym_matrix(8)->AddToMe(*my_bend_energy_hess) took " << tse2.count() - tse1.count() << " seconds" << std::endl << std::flush;


  exit(EXIT_SUCCESS);
}
