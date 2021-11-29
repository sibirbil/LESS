from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(name='LESS',
      version='0.1.0',
      description='Learning with Subset Stacking',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='git@github.com:sibirbil/LESS.git',
      maintainer='Ilker Birbil',
      maintainer_email='s.i.birbil@uva.nl',
      license='MIT',
      packages=['less'],
      zip_safe=False,
      install_requires=[
        'scikit-learn>=1.0.1',
        'numpy>=1.21.4'
      ]),
