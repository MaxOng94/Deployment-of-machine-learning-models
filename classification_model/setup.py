from setuptools import setup,find_packages
from pathlib import Path



# Package meta-data.
NAME = "classification_model"
DESCRIPTION = 'Train and deploy classification model.'
URL = 'your github project'
EMAIL = 'your_email@email.com'
AUTHOR = 'Your name'
REQUIRES_PYTHON = '>=3.6.0'

# What packages are required for this module to be executed?
def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()

# get the version for our model

ROOT_DIR = Path.cwd()
PACKAGE_ROOT = ROOT_DIR / NAME
# here its the classification_model package root
about = {}
with open(PACKAGE_ROOT/ VERSION) as file:
    _version = file.read().strip()
about["__version__"] = _version

# go to the directory where we position the README.md file.
tpp_pipeline = ROOT_DIR.parent

# import the README and use it as the long-description.
# because in our manifest.in file, where we specifiy to include files with .md extensions, we can use this.

try:
    with io.open(tpp_pipeline / "README.md",encoding = "utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION




setup(
name = "",
version = "",
author = "",
author_email = "",
description = "",
long_description = long_description,
author = AUTHOR,
author_email = EMAIL,
python_requires = REQUIRES_PYTHON,
url = URL,
packages = find_packages(exclude = ("tests", )),
package_data = {"classification_model":["VERSION"]},
install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
