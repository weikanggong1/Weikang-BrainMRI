#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE newimage

#include <boost/version.hpp>
#if BOOST_VERSION < 108000
#define BOOST_NO_CXX98_FUNCTION_BASE
#endif

#include <boost/test/unit_test.hpp>
