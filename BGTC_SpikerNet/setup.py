#------------------------------------------------------------------------------------------------------------------
# Original Author: Brandon S Coventry     Purdue University/Wisconsin Institute for Translational Neuroengineering
# Updated by Markson Chen
#------------------------------------------------------------------------------------------------------------------

from setuptools import setup, find_packages

version_reqs = [
                "gymnasium==0.29.1",
                "numpy>=1.23",
                "scipy>=1.10",
                "matplotlib>=3.8",
                "docopt>=0.6.2",
                "tabulate>=0.9.0",
                "scikit-learn>=1.7.0",
                "pyNN==0.12.3",
                "neuron>=8.2.4",
                "stable-baselines3==2.3.0",
                ]

setup(
    name="gym_spiker",
    version="0.1.0",
    author="Brandon S Coventry",
    author_email="bscoventry@gmail.com",
    description="Basal-Ganglia–Thalamo-Cortical simulated backend for SpikerNet",
    packages=find_packages(),
    install_requires=version_reqs,
    python_requires=">=3.9",
    url="https://github.com/bscoventry/SpikerNet_StarProtocols",
    download_url="https://github.com/bscoventry/SpikerNet_StarProtocols"   # v_0.0 is not stable
)