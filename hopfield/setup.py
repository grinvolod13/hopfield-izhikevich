from setuptools import setup, find_packages

setup(
    name='hopfield',
    version='0.1.0',
    packages=find_packages(include=['hopfield', 'hopfield.*'])
)