# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
#   This file is part of the rnasamba package, available at:
#   https://github.com/apcamargo/RNAsamba
#
#   Rnasamba is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#   Contact: antoniop.camargo@gmail.com

from setuptools import setup, find_packages

setup(
    name='rnasamba',
    version='0.1.0',
    packages=find_packages(),
    license='GNU General Public License v3.0',
    description='A tool for computing the coding potential of RNA transcript sequences using deep learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['biopython', 'numpy', 'keras >= 2.1.0', 'tensorflow>=1.5.0,<2.0'],
    python_requires= '>=3',
    entry_points = {
        'console_scripts': [
            'rnasamba-classify=rnasamba.cli:classify_cli',
            'rnasamba-train=rnasamba.cli:train_cli'
            ],
    },
    url='https://github.com/apcamargo/RNAsamba',
    keywords=['bioinformatics', 'coding potential', 'transcriptomics', 'machine learning', 'neural networks'],
    author='Antonio Pedro Camargo, Vsevolod Sourkov',
    author_email='antoniop.camargo@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Programming Language :: Python :: 3',
    ],
)