from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """Read requirements from file."""
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='house-price-prediction',
    version='0.1.0',
    author='Soumyajyoti',
    author_email='soumyajyoti@example.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    description='End-to-end California House Price Prediction ML Project',
)
