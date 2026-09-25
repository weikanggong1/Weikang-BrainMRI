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


BOOST_AUTO_TEST_SUITE(test_complexvolume)


using namespace NEWIMAGE;
using namespace NEWMAT;
using namespace std;

// Test data used by these tests is generated in the
// feedsRun script before execution.

// Test that neurologically and radiologically stored images
// (which are otherwise identical) are loaded correctly
BOOST_AUTO_TEST_CASE(read_complexvolume_handle_neuro_radio)
{
  complexvolume neuro, radio;

  BOOST_CHECK(read_complexvolume(neuro, string("complex_neuro")) == 0);
  BOOST_CHECK(read_complexvolume(radio, string("complex_radio")) == 0);

  BOOST_CHECK(neuro.xsize() == radio.xsize());
  BOOST_CHECK(neuro.ysize() == radio.ysize());
  BOOST_CHECK(neuro.zsize() == radio.zsize());

  for (int x = 0; x < neuro.xsize(); x++) {
  for (int y = 0; y < neuro.ysize(); y++) {
  for (int z = 0; z < neuro.zsize(); z++) {
    BOOST_CHECK(neuro.re(x, y, z) == radio.re(x, y, z));
    BOOST_CHECK(neuro.im(x, y, z) == radio.im(x, y, z));
  }}}
}


BOOST_AUTO_TEST_CASE(read_save_complexvolume_round_trip)
{
  volume<float> rneuro,  ineuro,  rradio,  iradio,
                rneuro2, ineuro2, rradio2, iradio2;

  // defined in feedsRun
  Matrix neuromat(4, 4), radiomat(4, 4);
  neuromat <<  3 << 0 << 0 << 10
           <<  0 << 3 << 0 << 20
           <<  0 << 0 << 3 << 30
           <<  0 << 0 << 0 << 1;
  radiomat << -3 << 0 << 0 << 37
           <<  0 << 3 << 0 << 20
           <<  0 << 0 << 3 << 30
           <<  0 << 0 << 0 << 1;

  BOOST_CHECK(read_complexvolume(rneuro,  ineuro,  string("complex_neuro"))  == 0);
  BOOST_CHECK(read_complexvolume(rradio,  iradio,  string("complex_radio"))  == 0);
  BOOST_CHECK(save_complexvolume(rneuro,  ineuro,  string("complex_neuro2")) == 0);
  BOOST_CHECK(save_complexvolume(rradio,  iradio,  string("complex_radio2")) == 0);
  BOOST_CHECK(read_complexvolume(rneuro2, ineuro2, string("complex_neuro2")) == 0);
  BOOST_CHECK(read_complexvolume(rradio2, iradio2, string("complex_radio2")) == 0);

  BOOST_CHECK(rneuro.sform_code() == rneuro2.sform_code());
  BOOST_CHECK(rneuro.qform_code() == rneuro2.qform_code());
  BOOST_CHECK(ineuro.sform_code() == ineuro2.sform_code());
  BOOST_CHECK(ineuro.qform_code() == ineuro2.qform_code());
  BOOST_CHECK(rradio.sform_code() == rradio2.sform_code());
  BOOST_CHECK(rradio.qform_code() == rradio2.qform_code());

  // All volumes are presented as radiological - the
  // L/R flip on neurological volumes is also applied
  // to the s/qform affines, so we compare all volumes
  // against the known radiological affine.
  BOOST_CHECK(rradio .sform_mat() == radiomat);
  BOOST_CHECK(rradio2.sform_mat() == radiomat);
  BOOST_CHECK(iradio .sform_mat() == radiomat);
  BOOST_CHECK(iradio2.sform_mat() == radiomat);
  BOOST_CHECK(rneuro .sform_mat() == radiomat);
  BOOST_CHECK(rneuro2.sform_mat() == radiomat);
  BOOST_CHECK(ineuro .sform_mat() == radiomat);
  BOOST_CHECK(ineuro2.sform_mat() == radiomat);
  BOOST_CHECK(rradio .qform_mat() == radiomat);
  BOOST_CHECK(rradio2.qform_mat() == radiomat);
  BOOST_CHECK(iradio .qform_mat() == radiomat);
  BOOST_CHECK(iradio2.qform_mat() == radiomat);
  BOOST_CHECK(rneuro .qform_mat() == radiomat);
  BOOST_CHECK(rneuro2.qform_mat() == radiomat);
  BOOST_CHECK(ineuro .qform_mat() == radiomat);
  BOOST_CHECK(ineuro2.qform_mat() == radiomat);

  for (int x = 0; x < rneuro.xsize(); x++) {
  for (int y = 0; y < rneuro.ysize(); y++) {
  for (int z = 0; z < rneuro.zsize(); z++) {
    BOOST_CHECK(rneuro(x, y, z) == rneuro2(x, y, z));
    BOOST_CHECK(ineuro(x, y, z) == ineuro2(x, y, z));
    BOOST_CHECK(rradio(x, y, z) == rradio2(x, y, z));
    BOOST_CHECK(iradio(x, y, z) == iradio2(x, y, z));
  }}}
}

BOOST_AUTO_TEST_SUITE_END()
