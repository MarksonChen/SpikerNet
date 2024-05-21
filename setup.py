#------------------------------------------------------------------------------------------------------------------
# Author: Brandon S Coventry     Purdue University/Wisconsin Institute for Translational Neuroengineering
# Date: 05/21/24                 Nice Cloudy Day today
# Purpose: Setup file for SpikerNet
#------------------------------------------------------------------------------------------------------------------
from setuptools import setup, find_packages
# TODO: Figure out how to get Matlab added
# TODO: update requirements.txt
version_reqs = ['atari-py==0.2.6',
                'box2d-py==2.3.8',
                'clang',
                'cudatoolkit==10.1.243',
                'cython==0.29.32',
                'seaborn',
                'numba',
                'genrl==0.0.2',
                'gym==0.17.1',
                'numpy==1.19.5',
                'opencv-python==4.2.0.34',
                'cachey',
                'cached_property',
                'pytorch==1.4.0',
                'ipympl',
                'scikit-learn==0.24.2',
                'h5py',
                'tensorboard==2.6.0', 
                'tensorflow==2.6.2',
                'pillow>=7.1.0',
                'torchvision==0.5.0',
                'tqdm', # Only used in example notebooks at this point
                ]

setup(
    name="SpikerNet",
    version="0.0.1",
    author="Brandon S Coventry",
    author_email="bscoventry@gmail.com",
    packages=find_packages(),
    install_requires=version_reqs,
    url="https://github.com/bscoventry/SpikerNet_StarProtocols",
    download_url="https://github.com/bscoventry/SpikerNet_StarProtocols"   # v_0.0 is not stable
)