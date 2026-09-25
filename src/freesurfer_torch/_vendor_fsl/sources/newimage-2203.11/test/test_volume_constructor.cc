#include "newimage/newimageall.h"
#include <stdlib.h>
#include <vector>

#include <boost/version.hpp>
#if BOOST_VERSION < 108000
#define BOOST_NO_CXX98_FUNCTION_BASE
#endif

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(test_volume_constructor)


using namespace NEWIMAGE;

BOOST_AUTO_TEST_CASE(volume_constructor_no_args)
{
    // volume constructor with no arguments
    // xsize should be zero
    volume<float> v;
    BOOST_TEST_MESSAGE("volume<float> v should have v.xsize() == 0");
    BOOST_CHECK(v.xsize() == 0);
}

BOOST_AUTO_TEST_CASE(volume_constructor_test_3_dim_args)
{
    // volume constructor with 3 dimensions, each dimension having length 1
    int x = 1;
    int y = 1;
    int z = 1;
    volume<float> v(x, y, z);
    BOOST_TEST_MESSAGE("volume<float> v(1, 1, 1); should have v.xsize(), v.ysize(), v.zsize() == 1");
    BOOST_CHECK(v.xsize() == 1);
    BOOST_CHECK(v.ysize() == 1);
    BOOST_CHECK(v.zsize() == 1);
}

BOOST_AUTO_TEST_CASE(volume_constructor_test_4_dim_args)
{
    // volume constructor with 4 dimensions. first 3 dimensions
    // are same length and 4th is 10
    int x = 1;
    int y = 1;
    int z = 1;
    int t = 10;
    volume<float> v(x, y, z, t);
    BOOST_TEST_MESSAGE("volume<float> v(1, 1, 1, 10); should have v.xsize(), v.ysize(), v.zsize() == 1, and v.tsize() == 10");
    BOOST_CHECK(v.xsize() == 1);
    BOOST_CHECK(v.ysize() == 1);
    BOOST_CHECK(v.zsize() == 1);
    BOOST_CHECK(v.tsize() == 10);
}


BOOST_AUTO_TEST_CASE(volume_4D_reinitialize)
{
    // create a volume from data in an array
    int x  = 4;
    int y  = 5;
    int z  = 6;
    int t  = 7;
    int sz = x * y * z * t;

    std::vector<float> data(sz);

    for (int i = 0; i < sz; i++) {
        data[i] = i;
    }

    volume<float> v;
    v.reinitialize(x, y, z, t, data.data(), false);
    BOOST_TEST(v.xsize() == 4);
    BOOST_TEST(v.ysize() == 5);
    BOOST_TEST(v.zsize() == 6);
    BOOST_TEST(v.tsize() == 7);

    for (int i = 0; i < sz; i++) {
        auto coords = v.ptrToCoord(i);
        int xi      = coords[0];
        int yi      = coords[1];
        int zi      = coords[2];
        int ti      = coords[3];
        BOOST_TEST(v(xi, yi, zi, ti) == i);
    }
}

BOOST_AUTO_TEST_SUITE_END()
