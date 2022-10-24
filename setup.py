import setuptools

def readme():
    with open('README.md') as f:
        return f.read()

setuptools.setup(name='torchsparseutils',
      version='0.0.1',
      description='A collection of utility functions to work with PyTorch sparse tensors',
      long_description=readme(),
      long_description_content_type="text/markdown",
      classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
      ],
      keywords='sparse torch utility',
      url='https://github.com/cai4cai/sparsitypreservingtorchutils',
      author='TODO',
      author_email='TODO',
      license='Apache-2.0',
      packages=['torchsparseutils'],
      install_requires=[
          'torch',
      ],
      zip_safe=False)