# setup.py
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import numpy as np

ext_modules = [
    Pybind11Extension(
        "disqco._fm_cpp",
        sources=["src/disqco/parti/FM/cpp/bindings.cpp"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-std=c++17"],
        cxx_std=17,
    ),
]

setup(
    name="disqco",
    version="0.0.5",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "numpy==2.2.4",
        "qiskit==1.2.4",
        "qiskit-aer==0.15.1",
        "qiskit-qasm3-import==0.5.1",
        "networkx",
        "matplotlib",
        "pylatexenc",
        "jupyter-tikz",
        "ipykernel",
        "pytest",
        "tqdm",
    ],
    python_requires='>=3.11',
)
