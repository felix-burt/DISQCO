# setup.py
# Package metadata and dependencies live in pyproject.toml; this file only
# declares the optional C++ FM extension. If it fails to compile, the package
# still installs and FiducciaMattheyses falls back to the pure-Python FM pass.
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "disqco._fm_cpp",
        sources=["src/disqco/parti/FM/cpp/bindings.cpp"],
        extra_compile_args=["-O3"],
        cxx_std=17,
        optional=True,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
