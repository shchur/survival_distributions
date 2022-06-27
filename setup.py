from setuptools import find_packages, setup

setup(
    name="survival_distributions",
    author="Oleksandr Shchur",
    description="Extended functionality for univariate probability distributions in PyTorch",
    version="0.0.2",
    license="MIT",
    author_email="oleks.shchur@gmail.com",
    url="https://github.com/shchur/survival_distributions",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=["torch>=1.10"],
)
