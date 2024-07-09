from setuptools import setup
from setuptools import Extension


# Boost Python C++ extension for test boost.cpp methods.
boost_cpp = Extension(
    name="boost_cpp",
    sources=["./src/spnn/boost.cpp"],
    include_dirs=["/usr/include"],  # C++ header files.
    library_dirs=["/usr/lib"],  # C++ libs.
    libraries=["boost_python39"],
    extra_compile_args=[
        "-shared",
        "-export-dynamic",
    ],
)

# Boost Python C++ extension for spnn.cpp.
spnn_cpp = Extension(
    name="spnn_cpp",
    sources=["./src/spnn/spnn.cpp"],
    include_dirs=["/usr/include"],  # C++ header files.
    library_dirs=["/usr/lib"],  # C++ libs.
    libraries=["armadillo", "boost_python39"],
    extra_compile_args=[
        "-shared",
        "-export-dynamic",
    ],
)

# Build of Boost C++ and Python extensions of "spnn" library.
setup(
    name="spnn",
    version="0.1",
    packages=["spnn"],
    package_dir={"": "src"},
    ext_modules=[boost_cpp, spnn_cpp],
    py_modules=["spnn"],
)
