"""Setup configuration for the heart rate data processing pipeline."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="heart-rate-pipeline",
    version="1.0.0",
    author="Heart Rate Data Pipeline Team",
    author_email="your-email@domain.com",
    description="A modular pipeline for processing wearable heart rate data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/heart-rate-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.3.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0",
            "flake8>=3.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hr-pipeline=src.run_pipeline:main",
        ],
    },
)
