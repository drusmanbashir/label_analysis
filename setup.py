from setuptools import setup, find_packages

setup(
    name="label_analysis",
    version="0.1.0",
    author="Usman Bashir",
    author_email="usman.bashir@example.com",
    description="A Python package for analyzing and processing labeled data.",
    url="https://github.com/yourusername/label_analysis",  # Update this with the correct repository link
    packages=find_packages(),
    install_requires=[
        "batchgenerators==0.25",
        "fastcore==1.6.3",
        "ipdb==0.13.13",
        "litq==0.1.0",
        "matplotlib==3.10.1",
        "networkx==3.3",
        "numpy",
        "pandas==2.2.3",
        "pytest==8.3.3",
        "scipy==1.15.2",
        "SimpleITK==2.4.1",
        "vtk==9.3.1",
    ],
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
