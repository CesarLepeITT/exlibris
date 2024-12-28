# setup.py
from setuptools import setup, find_packages

setup(
    name="exlibris",
    version="0.1",
    packages=find_packages(),
    description="Exlibris is a Python library that provides statistical comparisons between different binary classification models. This library is specifically designed to compare the GSGP classifier with other models.",
    url='https://github.com/CesarLepeITT/exlibris',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",  # Asegúrate de que sea markdown
    author="Lepe Garcia Cesar",
    author_email="l22212360@tijuana.tecnm.mx",
    install_requires=[
        'matplotlib==3.10.0',
        'pandas==2.2.3',
        'scikit-learn==1.6.0'],
    package_data={'exlibris': ['datasets/*.csv']}, 
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
