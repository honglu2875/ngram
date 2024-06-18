import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import Extension, setup

code_root = "src/ngram"
include_root = "csrc/include"
cpp_root = "csrc"
openmp_arg = "-fopenmp"

extensions = [
    Extension(
        "ngram", 
        sources=[cpp_root + "/nodes.cpp", code_root + "/ngram.pyx",], 
        include_dirs=[numpy.get_include(), include_root, cpp_root], 
        extra_compile_args=["-O3", openmp_arg], 
        language="c++",
        extra_link_args=[openmp_arg],
        cmdclass = {'build_ext': build_ext},
    ),
]
setup(
    name='ngram',
    ext_modules = cythonize(extensions),
)
