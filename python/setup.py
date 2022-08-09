from setuptools import setup

with open("../README.md", "r") as fh:
    ld = fh.read()

with open("requirements.txt","r") as r_file:
    install_r = r_file.readlines()

setup(
    name='proteolizardalgo',
    version='0.1.0',
    description='python interface to scalable algorithms based on bruker timsTOF data',
    packages=['proteolizardalgo'],
    package_data={'proteolizardalgorithm': ['proteolizardalgorithm.so']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 or later (GPLv3+)",
        "Operating System :: Linux"
    ],
    long_description=ld,
    long_description_content_type="text/markdown",
    install_requires=install_r
)
