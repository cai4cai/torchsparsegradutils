import setuptools


def readme():
    with open("README.md") as f:
        return f.read()


setuptools.setup(
    name="torchsparsegradutils",
    version="0.0.1",
    description="A collection of utility functions to work with PyTorch sparse tensors",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    keywords="sparse torch utility",
    url="https://github.com/cai4cai/torchsparsegradutils",
    author="CAI4CAI research group",
    author_email="contact@cai4cai.uk",
    license="Apache-2.0",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch",
    ],
    extras_require={
        "extras": ["jax", "cupy"],
    },
    zip_safe=False,
)
