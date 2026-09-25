#define EXPOSE_TREACHEROUS

#include "newimage/newimageall.h"
#include "armawrap/newmat.h"
#include <stdlib.h>
#include <vector>

#include <boost/version.hpp>
#if BOOST_VERSION < 108000
#define BOOST_NO_CXX98_FUNCTION_BASE
#endif

#include <boost/test/unit_test.hpp>

//  #  define EPS 1e-10
#define EPS 0.0000000001


BOOST_AUTO_TEST_SUITE(test_volume_masked_operations)


using namespace NEWIMAGE;
using namespace NEWMAT;
using namespace std;


#define assert_equals(V1,V2) BOOST_CHECK(abs(V1 - V2) < EPS)


BOOST_AUTO_TEST_CASE(volume_3d_mask_3d)
{
  float voldata[]  = {1, 1, 1, 1, 0, 0, 0, 0};
  float maskdata[] = {0, 1, 1, 0, 1, 0, 0, 0};

  volume<float> vol;  vol .reinitialize(2, 2, 2, 1, voldata,  false);
  volume<float> mask; mask.reinitialize(2, 2, 2, 1, maskdata, false);

  assert_equals(vol.sum(),          4);
  assert_equals(vol.mean(),         0.5);
  assert_equals(vol.variance(),     0.2857142857142857);
  assert_equals(vol.sum(mask),      2);
  assert_equals(vol.mean(mask),     0.66666666667);
  assert_equals(vol.variance(mask), 0.33333333333);
}


BOOST_AUTO_TEST_CASE(volume_4d_mask_3d)
{
  float voldata[]  = {1, 1, 1, 1, 0, 0, 0, 0,
                      0, 0, 1, 1, 1, 1, 0, 0};
  float maskdata[] = {0, 1, 1, 0, 1, 1, 0, 0};

  volume<float> vol;  vol .reinitialize(2, 2, 2, 2, voldata,  false);
  volume<float> mask; mask.reinitialize(2, 2, 2, 1, maskdata, false);

  assert_equals(vol.sum(),          8);
  assert_equals(vol.mean(),         0.5);
  assert_equals(vol.variance(),     0.26666666667);

  assert_equals(vol.sum(mask),      5);
  assert_equals(vol.mean(mask),     0.625);
  assert_equals(vol.variance(mask), 0.26785714285714285);
}

// applying a 4D mask to a 3D volume should error
BOOST_AUTO_TEST_CASE(volume_3d_mask_4d)
{
  float voldata[]  = {1, 1, 1, 1, 0, 0, 0, 0};
  float maskdata[] = {0, 1, 1, 0, 1, 1, 0, 0,
                      0, 0, 0, 0, 1, 1, 1, 1};

  volume<float> vol;  vol .reinitialize(2, 2, 2, 1, voldata,  false);
  volume<float> mask; mask.reinitialize(2, 2, 2, 2, maskdata, false);

  assert_equals(vol.sum(),          4);
  assert_equals(vol.mean(),         0.5);
  assert_equals(vol.variance(),     0.2857142857142857);

  BOOST_CHECK_THROW(vol.sum(     mask), std::runtime_error);
  BOOST_CHECK_THROW(vol.mean(    mask), std::runtime_error);
  BOOST_CHECK_THROW(vol.variance(mask), std::runtime_error);
}


BOOST_AUTO_TEST_CASE(volume_4d_mask_4d)
{
  float voldata[]  = {1, 1, 1, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 1, 1, 1, 1};
  float maskdata[] = {0, 1, 1, 0, 1, 1, 0, 0,
                      0, 0, 0, 0, 0, 1, 1, 0};

  volume<float> vol;  vol .reinitialize(2, 2, 2, 2, voldata,  false);
  volume<float> mask; mask.reinitialize(2, 2, 2, 2, maskdata, false);

  assert_equals(vol.sum(),          8);
  assert_equals(vol.mean(),         0.5);
  assert_equals(vol.variance(),     0.26666666667);

  assert_equals(vol.sum(mask),      4);
  assert_equals(vol.mean(mask),     0.66666666667);
  assert_equals(vol.variance(mask), 0.26666666667);
}


BOOST_AUTO_TEST_SUITE_END()
