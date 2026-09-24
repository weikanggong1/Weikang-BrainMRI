/*  FAST4 - FMRIB's Automated Segmentation Tool v4

    John Vickers, Mark Jenkinson and Steve Smith
    FMRIB Image Analysis Group

    Copyright (C) 2005-2007 University of Oxford  */

/*  CCOPYRIGHT  */

#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "utils/options.h"


class ZMRIMULTISegmentation
{
 public:

  ZMRIMULTISegmentation();
  ~ZMRIMULTISegmentation()
    {
    }
  void TanakaCreate(const NEWIMAGE::volume<float>* images, int nclasses, bool b2d, int nbiter,float nblowpass, float fbeta, int bapused, bool pveboolean, int nchan, bool bbias, int initfixity, bool verb,int biterationspve, int winit, float mixelpB, float Hyp,std::string mansegfle,int finalT2file);
  int TanakaMain(NEWIMAGE::volume<float>& pcsf, NEWIMAGE::volume<float>& pgm, NEWIMAGE::volume<float>& pwm);


  NEWIMAGE::volume4D<float> m_post;
  NEWIMAGE::volume4D<float> members;
  NEWIMAGE::volume4D<float> m_pve;
  NEWIMAGE::volume<float>* m_Finalbias;
  NEWIMAGE::volume<int> m_Segment;
  NEWIMAGE::volume<int> m_pveSegment;
  NEWIMAGE::volume<int> hardPV;
  NEWMAT::Matrix coord_minus_mean;
  float maximum, minimum;


private:
  NEWIMAGE::volume<float> Convolve(NEWIMAGE::volume<float>& resfieldimage);
  NEWIMAGE::volume4D<float> InitclassAlt(int);
  NEWIMAGE::volume4D<float>  Initclass(int noclasses);
  NEWMAT::Matrix covariancematrix(int classid, NEWIMAGE::volume4D<float> probability);
  float M_2PI(int numberofchan);
  float logpveGaussian(int x, int y, int z, NEWMAT::Matrix mu, NEWMAT::Matrix sig, float detsig);
  float PVEnergy(int x, int y, int z, NEWMAT::Matrix mu, NEWMAT::Matrix sigma, float sigdet);
  float logGaussian(int classnumber, int x, int y, int z);
  float pvmeans(int clas);
  float pvvar(int clas);
  void Volumesquant(const NEWIMAGE::volume4D<float>& probs);
  void Initialise();
  void takeexpo();
  float  MRFWeightsTotal();
  double MRFWeightsInner(const int x, const int y, const int z,const int c);
  double MRFWeightsAM(const int l, const int m, const int n);
  void InitWeights();
  void UpdateWeights();
  void InitSimple(const NEWIMAGE::volume<float>& pcsf, const NEWIMAGE::volume<float>& pgm, const NEWIMAGE::volume<float>& pwm);
  void InitAprioriKMeans();
  void BiasRemoval();
  void MeansVariances(int numberofclasses);
  void Dimensions();
  void WeightedKMeans();
  void mresmatrix();
  void UpdateMembers(NEWIMAGE::volume4D<float>& probability);
  void PVClassificationStep();
  void ICMPV();
  void PVMoreSophisticated();
  void PVestimation();
  void PVMeansVar(const NEWIMAGE::volume4D<float>& m_posttemp);
  void Classification(int x, int y, int z);
  void Classification();
  void InitKernel();
  void pveClassification(int x, int y, int z);
  void pveClassification();
  int qsort();
  void TanakaHyper();
  void TanakaPriorHyper();
  void TanakaIterations();
  NEWMAT::Matrix*  m_inv_co_variance;
  NEWMAT::Matrix* m_co_variance;
  NEWMAT::Matrix m_mean;
  NEWIMAGE::volume4D<float> m_prob;
  NEWIMAGE::volume4D<float>PVprob;
  NEWIMAGE::volume4D<float> talpriors;
  NEWIMAGE::volume<float>* m_Mri;
  NEWIMAGE::volume<float>* m_Mricopy;
  NEWIMAGE::volume<float>* p_bias;
  NEWIMAGE::volume<float>* m_meaninvcov;
  NEWIMAGE::volume<float>* m_resmean;
  NEWIMAGE::volume<float> pve_eng;
  NEWIMAGE::volume<float> m_maskc;
  NEWIMAGE::volume<int> m_mask;
  NEWMAT::ColumnVector kernelx, kernely, kernelz;
  float amx, amy, amz, amxy, amzx, amzy;
  double* volumequant;
  float* rhs;
  float* weight;
  int imagetype;
  float m_nbLowpass;
  float m_nxdim, m_nydim, m_nzdim;
  float beta, Hyper;
  float pveBmixeltype;
  int noclasses;
  int numberofchannels;
  int m_nSlicesize, m_nWidth, m_nHeight, m_nDepth, m_nbIter, initfixed, inititerations, iterationspve;
  int bapusedflag;
  bool verboseusage;
  bool biasfieldremoval;
  std::string mansegfile;

class kmeansexception: public std::exception
{
 public:
  virtual const char* what() const throw ()
  {
    return "Exception: Not enough classes detected to init KMeans";
  }
} kmeansexc;

};
