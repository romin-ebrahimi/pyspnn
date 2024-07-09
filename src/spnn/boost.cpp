#include <boost/python.hpp>
#include <string>

using namespace boost::python;

// Hello world test that Boost Python C++ works.
std::string hello_world() {
   return std::string("Hello World.");
}

// Test that passing boost::python::list to C++ works.
list test_list(list input_list) {
    ssize_t size = len(input_list);
    list output_list;
    output_list.append(object()); // Append Python None type.
    output_list *= size;
    double value;

    for (unsigned int i = 0; i < size; i++) {
        value = extract<double>(input_list[i]);
        output_list[i] = value;
    }

    return output_list;
}

BOOST_PYTHON_MODULE(boost_cpp) {
    using namespace boost::python;
    def("hello_world", hello_world);
    def("test_list", test_list);
}