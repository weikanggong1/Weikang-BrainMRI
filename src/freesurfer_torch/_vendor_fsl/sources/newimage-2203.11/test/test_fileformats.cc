
#include "newimage/newimageall.h"
#include <stdlib.h>
#include <vector>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(test_newimage_file_formats)

using namespace std;

BOOST_AUTO_TEST_CASE(newimage_file_formats)
{
  auto formats = NEWIMAGE::fileFormats::allFormats();

  int c = 0;

  for (pair<string, int> &fmt : formats) {

    switch (fmt.second) {

      case FSL_TYPE_ANALYZE:        BOOST_CHECK(0 == fmt.first.compare("ANALYZE"));        c++; break;
      case FSL_TYPE_ANALYZE_GZ:     BOOST_CHECK(0 == fmt.first.compare("ANALYZE_GZ"));     c++; break;
      case FSL_TYPE_NIFTI:          BOOST_CHECK(0 == fmt.first.compare("NIFTI"));          c++; break;
      case FSL_TYPE_NIFTI_GZ:       BOOST_CHECK(0 == fmt.first.compare("NIFTI_GZ"));       c++; break;
      case FSL_TYPE_NIFTI_PAIR:     BOOST_CHECK(0 == fmt.first.compare("NIFTI_PAIR"));     c++; break;
      case FSL_TYPE_NIFTI_PAIR_GZ:  BOOST_CHECK(0 == fmt.first.compare("NIFTI_PAIR_GZ"));  c++; break;
      case FSL_TYPE_NIFTI2:         BOOST_CHECK(0 == fmt.first.compare("NIFTI2"));         c++; break;
      case FSL_TYPE_NIFTI2_GZ:      BOOST_CHECK(0 == fmt.first.compare("NIFTI2_GZ"));      c++; break;
      case FSL_TYPE_NIFTI2_PAIR:    BOOST_CHECK(0 == fmt.first.compare("NIFTI2_PAIR"));    c++; break;
      case FSL_TYPE_NIFTI2_PAIR_GZ: BOOST_CHECK(0 == fmt.first.compare("NIFTI2_PAIR_GZ")); c++; break;
    }
  }

  // make sure all of the
  // cases above were triggered
  BOOST_CHECK(c == 10);
}


BOOST_AUTO_TEST_SUITE_END()
