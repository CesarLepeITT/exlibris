# setup.py
from setuptools import setup, find_packages

setup(
    name="exlibris",
    version="0.1",
    packages=find_packages(),
    description="This library generates statistics for the assigned binary classification problems.",
    author="Lepe Garcia Cesar",
    author_email="l22212360@tijuana.teccnm.mx",
     install_requires=['pandas>=1.0.5',
                    'numpy>=1.21',
                    'scikit-learn>=0.22.2',
                   ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
