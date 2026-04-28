# setup.py
from setuptools import setup, find_packages

setup(
    name="disqco",
    version="0.0.5",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy>=2.2",
                      "qiskit",
                      "qiskit-aer",
                      "bosonic-model",
                      "bosonic-converters",
                      "networkx",
                      "matplotlib",
                      "pylatexenc",
                      "jupyter-tikz",
                      "ipykernel",
                      "pytest",
                      "tqdm"],
    python_requires='>=3.11',
)