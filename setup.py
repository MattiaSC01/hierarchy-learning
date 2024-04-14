from setuptools import setup, find_packages

setup(
    name="rhm",
    version="0.1",
    packages=find_packages(where='rhm'),
    package_dir={'': 'rhm'}, 
    description="Random Hierarchy Model, based on code from https://github.com/pcsl-epfl/hierarchy-learning",
    install_requires=[
        'torch',
        'numpy',
    ],
)
