# setup.py
from setuptools import setup, find_packages

setup(
    name="bosonic-disqco",
    version="0.0.7",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy>=1.26,<2.3.4",
                      "qiskit>=2.2.2",
                      "qiskit-aer>=0.14.0",
                      "qiskit-qasm3-import>=0.6.0",
                      "networkx", 
                      "matplotlib", 
                      "pylatexenc", 
                      "jupyter-tikz", 
                      "ipykernel", 
                      "pytest", 
                      "tqdm"],
    python_requires='>=3.10',
)
