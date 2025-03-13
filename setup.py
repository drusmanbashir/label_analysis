from setuptools import setup, find_packages
import os
with open(os.path.join(os.path.dirname(__file__), "requirements.txt"), encoding="utf-8") as f:
    all_requirements = f.read().splitlines()
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

development_only = ["gudhi", "nurbspy"]

# Filter out development-only requirements using partial string matching
requirements = [
    req for req in all_requirements 
    if not any(dev_req in req for dev_req in development_only)
]
setup(
    name="label_analysis",
    version="0.1.0",
    author="Usman Bashir",
    author_email="usman.bashir@example.com",
    description="A Python package for analyzing and processing labeled data.",
    url="https://github.com/yourusername/label_analysis",
    packages=find_packages(),
    install_requires= requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
)

