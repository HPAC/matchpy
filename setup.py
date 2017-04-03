# -*- coding: utf-8 -*-
import os.path

from setuptools import setup, find_packages

root = os.path.dirname(__file__)

with open(os.path.join(root, 'README.rst')) as f:
    readme = f.read()


setup(
    name="matchpy",
    use_scm_version=True,
    description="A pattern matching library.",
    long_description=readme,
    author="Manuel Krebber",
    author_email="admin@wheerd.de",
    url='https://github.com/HPAC/matchpy',
    license='MIT',
    zip_safe=True,
    packages=find_packages(exclude=('tests', )),
    test_suite='tests',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    setup_requires=[
        'setuptools_scm >= 1.7.0',
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
        'hypothesis',
    ],
    install_requires=[
        'hopcroftkarp>=1.2,<2.0',
        'multiset>=2.0,<3.0',
    ],
    extras_require={
        'graphs': ['graphviz>=0.5,<0.6'],
    },
)

