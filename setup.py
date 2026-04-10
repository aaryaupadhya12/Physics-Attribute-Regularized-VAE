"""
PAR-VAE: Physics-Attribute-Regularized VAE
A physics-constrained generative model for COVID-19 CT severity classification
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="parvae",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@institution.edu",
    description="Physics-Attribute-Regularized VAE for COVID-19 CT analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PAR-VAE",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/PAR-VAE/issues",
        "Documentation": "https://github.com/yourusername/PAR-VAE/tree/main/docs",
        "Source Code": "https://github.com/yourusername/PAR-VAE",
    },
    packages=find_packages(include=["src", "src.*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "isort>=5.0",
            "pylint>=2.0",
            "mypy>=0.990",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    include_package_data=True,
    keywords=[
        "medical-imaging",
        "covid-19",
        "variational-autoencoder",
        "physics-informed",
        "ct-scans",
        "deep-learning",
    ],
)
