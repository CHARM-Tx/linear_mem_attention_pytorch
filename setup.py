from setuptools import setup, find_packages

setup(
  name = 'linear-mem-attention-pytorch',
  packages = find_packages(),
  version = '0.0.1',
  license='CreativeCommons4.0',
  description = 'Fast Attention Kernel with Linear Memory Footprint',
  author = 'Eric Alcaide',
  author_email = 'eric@charmtx.com',
  url = 'https://github.com/CHARM-Tx/linear_mem_attention_pytorch',
  keywords = [
    'artificial intelligence',
    'attention',
    'natural language processing',
    'deep learning'
  ],
  install_requires=[
    'torch>=1.6',
    'einops>=0.3',
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Programming Language :: Python :: 3.7',
  ],
)