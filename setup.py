# setup.py
from setuptools import setup, find_packages

setup(
    name="disqco",
    version="0.0.3",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["numpy==2.2.4", "qiskit==1.2.4", "qiskit-aer==0.15.1", "qiskit-qasm3-import==0.5.1", "networkx", "matplotlib", "pylatexenc", "jupyter-tikz", "ipykernel"],
    python_requires='>=3.11',
)