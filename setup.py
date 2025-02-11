from setuptools import setup

setup(
    name='BMI_HW4',
    version='0.1.0',    
    description='My homework 4 submission',
    url='https://github.com/R-Reidjr/HW4',
    author='Rashad Reid Jr',
    author_email='rashad.reid@ucsf.edu',
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy',
                      'pytest', 
                      'sklearn'
    ]
    python_requires="=3.13.1"
)