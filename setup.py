# Copyright 2024 Honglu Fan (https://github.com/honglu2875).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import platform

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# setuptools requires every path handed to setup() to be relative and
# /-separated, so keep them that way rather than absolutising.
CSRC = "csrc"
SAIS = "csrc/libsais"

IS_WINDOWS = platform.system() == "Windows"
USE_OPENMP = os.environ.get("NGRAM_OPENMP", "1") != "0"
USE_NATIVE = os.environ.get("NGRAM_NATIVE", "0") == "1"

# libsais is C; the engine is C++.  The two need different flags, so keep the
# C++-only ones in a separate list and strip them per source below.
common_compile_args = ["-O3", "-fno-strict-aliasing"]
cxx_only_args = ["-std=c++17"]
link_args = []
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

if USE_OPENMP:
    common_compile_args.append("-fopenmp")
    link_args.append("-fopenmp")
    define_macros.append(("LIBSAIS_OPENMP", "1"))
if USE_NATIVE:
    common_compile_args.append("-march=native")
if IS_WINDOWS:  # pragma: no cover - not exercised here
    common_compile_args = ["/O2"]
    cxx_only_args = ["/std:c++17"]
    link_args = []

sais_sources = [
    SAIS + "/" + name
    for name in ("libsais.c", "libsais16.c", "libsais16x64.c", "libsais64.c")
]

extensions = [
    Extension(
        "ngram._core",
        sources=["src/ngram/_core.pyx"] + sais_sources,
        include_dirs=[numpy.get_include(), CSRC + "/include", SAIS],
        extra_compile_args=common_compile_args + cxx_only_args,
        extra_link_args=link_args,
        define_macros=define_macros,
        language="c++",
    )
]


class BuildExt(build_ext):
    """Strip C++-only flags when the compiler is invoked on a ``.c`` source."""

    def build_extensions(self):
        original = self.compiler._compile

        def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith(".c"):
                extra_postargs = [a for a in extra_postargs if a not in cxx_only_args]
            return original(obj, src, ext, cc_args, extra_postargs, pp_opts)

        self.compiler._compile = _compile
        try:
            super().build_extensions()
        finally:
            self.compiler._compile = original


setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "embedsignature": True,
        },
        annotate=os.environ.get("NGRAM_ANNOTATE", "0") == "1",
    ),
    cmdclass={"build_ext": BuildExt},
)
