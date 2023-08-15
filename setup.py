import setuptools


def readme():
    with open("README.md") as f:
        return f.read()


setuptools.setup(
    name="torchsparsegradutils",
    version="0.1.2",
    description="A collection of utility functions to work with PyTorch sparse tensors",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8, <3.11",
    keywords="sparse torch utility",
    url="https://github.com/cai4cai/torchsparsegradutils",
    author="CAI4CAI research group",
    author_email="contact@cai4cai.uk",
    license="Apache-2.0",
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=[
        "torch>=1.13",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    test_suite="tests",
    extras_require={
        "extras": ["jax", "cupy"],
    },
    zip_safe=False,
    include_package_data=True,
)
