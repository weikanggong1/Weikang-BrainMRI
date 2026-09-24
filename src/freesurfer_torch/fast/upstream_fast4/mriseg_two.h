/*  FAST4 - FMRIB's Automated Segmentation Tool v4

    John Vickers, Mark Jenkinson, Steve Smith and Matthew Webster
    FMRIB Image Analysis Group

    Copyright (C) 2005-2009 University of Oxford  */

/*  CCOPYRIGHT  */

#include <vector>

#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "utils/options.h"


class ZMRISegmentation
{
 public:
   ZMRISegmentation();
  ~ZMRISegmentation()
    {
    }
  void TanakaCreate(const NEWIMAGE::volume<float>& image, float fbeta, int nclasses, float nblowpass, bool bbias, int biterationspve, float mixeltypeMRF, int nbiter, int initinitfixed, int winitfixed, int bapused, float hyp, bool verb,std::string mansegfle,int typeoffile);
  int TanakaMain(NEWIMAGE::volume<float>& pcsf, NEWIMAGE::volume<float>& pgm, NEWIMAGE::volume<float>& pwm);

  NEWIMAGE::volume4D<float> members;
  NEWIMAGE::volume4D<float> m_post;
  NEWIMAGE::volume4D<float> m_pve;
  NEWIMAGE::volume<float> m_BiasField;
  NEWIMAGE::volume<int> m_Segment;
  NEWIMAGE::volume<int> m_pveSegment;
  NEWIMAGE::volume<int>hardPV;


private:
  void BiasRemoval();
  void Classification();
  void Classification(int x, int y, int z);
  NEWIMAGE::volume<float> Convolve(NEWIMAGE::volume<float>& resfieldimage);
  void Dimensions();
  void Initclass();
  float MRFWeightsTotal();
  double MRFWeightsInner(const int x, const int y, const int z,const int c);
  double MRFWeightsAM(const int l, const int m, const int n);
  float logGaussian(float y_i, float mu, float sigma);
  float PVEnergy(int x, int y, int z, float mu, float sigma);
  void qsort();
  void TanakaIterations();
  void TanakaHyper();
  void TanakaPriorHyper();
  void Initialise();
  void InitSimple(const NEWIMAGE::volume<float>& pcsf, const NEWIMAGE::volume<float>& pgm, const NEWIMAGE::volume<float>& pwm);
  void UpdateMembers(NEWIMAGE::volume4D<float>& probability);
  void WeightedKMeans();
  void pveIterations();
  void pveInitialize();
  void takeexpo();
  void InitKernel(float kernalsize);
  void pveClassification(int x, int y, int z);
  void pveClassification();
  void Probabilities(int index);
  void calculateVolumes(const NEWIMAGE::volume4D<float>& probs);
  void printVolumeTotals();
  void PVClassificationStep();
  void ICMPV();
  void PVEnergyInit();
  void PVestimation();
  void MeansVariances(int numberofclasses, NEWIMAGE::volume4D<float>& probability);

  NEWIMAGE::volume4D<float>PVprob;
  NEWIMAGE::volume4D<float> pvprobsinit;
  NEWIMAGE::volume4D<float> talpriors;
  NEWIMAGE::volume4D<float> m_prob;
  NEWIMAGE::volume <float>  m_Mricopy;
  NEWIMAGE::volume<float> m_Mri;
  NEWIMAGE::volume<float> m_mask;
  NEWMAT::ColumnVector kernelx, kernely, kernelz;
  std::vector<double> volumequant;
  std::vector<float> m_mean;
  std::vector<float> m_variance;
  std::vector<float> weight;
  std::vector<float> rhs;
  double m_nxdim, m_nydim, m_nzdim;
  float m_nbLowpass;
  float beta;
  float pveBmixeltype;
  float nvoxel;
  float Hyper;
  float amx, amy, amz, amxy, amzy, amzx;
  int imagetype,m_nWidth, m_nHeight, m_nDepth;
  int neighbour[27];
  int m_nbIter;
  int bapusedflag;
  int nClasses;
  int inititerations;
  int usingmeanfield;
  int iterationspve;
  int initfixed;
  bool biasfieldremoval;
  bool verboseusage;
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
