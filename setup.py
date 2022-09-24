from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='less-learn',
      version='0.3.0',
      description='Learning with Subset Stacking',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/sibirbil/LESS',
      maintainer='Ilker Birbil',
      maintainer_email='s.i.birbil@uva.nl',
      license='MIT',
      packages=['less'],
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=[
        'scikit-learn>=1.0.1',
        'numpy>=1.21.5'
      ])
