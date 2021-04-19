from setuptools import setup, find_packages

setup(
      name='tfts',
      version='0.1.0',
      description='tensorflow time prediction',
      url='https://github.com/LongxingTan/Time-series-prediction',
      author='Longxing Tan',
      author_email='tanlongxing888@163.com',
      install_requires=['tensorflow>=2.0.0'],
      packages=find_packages()
)
