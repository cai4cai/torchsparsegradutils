import setuptools


def readme():
    with open("README.md") as f:
        return f.read()


setuptools.setup(
    name="torchsparsegradutils",
    version="0.2.0",
    description="A collection of utility functions to work with PyTorch sparse tensors",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    keywords="sparse torch utility",
    url="https://github.com/cai4cai/torchsparsegradutils",
    author="CAI4CAI research group",
    author_email="contact@cai4cai.uk",
    license="Apache-2.0",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "torch>=2.5",
        "scipy",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
    extras_require={
        "extras": ["jax", "cupy"],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-copybutton>=0.5.0",
            "myst-parser>=2.0.0",
            "sphinx-autobuild>=2021.3.14",
            "matplotlib>=3.5.0",
            "sphinx-autodoc-typehints>=1.24.0",
        ],
    },
    zip_safe=False,
    include_package_data=True,
)
