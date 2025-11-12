#!/usr/bin/env python3
"""
Setup script for the reach package.

Time-free quantum reachability analysis using spectral, moment, and Krylov criteria.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Time-free quantum reachability analysis"

# Core dependencies
INSTALL_REQUIRES = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "matplotlib>=3.5.0",
    "pandas>=1.3.0",
    "qutip>=4.7.0",
]

# Optional dependencies for development and testing
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "black>=22.0.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
    ],
    "performance": [
        "tqdm>=4.60.0",  # Progress bars for long runs
    ],
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
}

# All optional dependencies
EXTRAS_REQUIRE["all"] = sum(EXTRAS_REQUIRE.values(), [])

setup(
    name="reach",
    version="0.1.0",
    author="Tomasz Andrzejewski",
    author_email="tomasz.andrzejewski@example.com",
    description="Time-free quantum reachability analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TomaszAnd/reachability",
    project_urls={
        "Bug Reports": "https://github.com/TomaszAnd/reachability/issues",
        "Source": "https://github.com/TomaszAnd/reachability",
        "Documentation": "https://github.com/TomaszAnd/reachability#readme",
    },
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "archive_*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "reach=reach.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="quantum reachability control hamiltonians random-matrix-theory",
)
