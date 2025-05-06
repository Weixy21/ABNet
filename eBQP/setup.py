#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

requirements = [ ]

setup_requirements = [ ]

test_requirements = ['torch']

setup(
    author="Wei Xiao",
    author_email='weixy@mit.edu',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="An explicit solution for the CBF-based QP in PyTorch",
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    name='eBQP',
    packages=find_packages(include=['eBQP']),
    setup_requires=setup_requirements,
    test_requires = test_requirements,
    url='https://github.com/Weixy21/ABNet',
    version='0.0.0',
    zip_safe=False,
)