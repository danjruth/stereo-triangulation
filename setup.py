from setuptools import setup, find_packages

setup(name='stereo-triangulation',
      version='0.1',
      packages=find_packages(),
      author='Daniel J. Ruth',
      url=r'https://github.com/DeikeLab/stereo-triangulation/',
      description='Find the 3-d positions of objects using images from two cameras.',
      long_description=open('README.md').read(),
      install_requires=[
          'numpy',
          'pandas',
          'scipy',
          'scikit-image',
          'opencv-python',
          'joblib',
          ],
      )