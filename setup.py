# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="patternmatcher",
    version='0.0.1',
    description="A pattern matching library",
    long_description=readme,
    author="Manuel Krebber",
    author_email="admin@wheerd.de",
    url="TODO",
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

