from setuptools import setup

setup(name='LESS',
      version='0.1.0',
      description='Learning with Subset Stacking',
      url='git@github.com:sibirbil/LESS.git',
      maintainer='Ilker Birbil',
      maintainer_email='s.i.birbil@uva.nl',
      license='MIT',
      packages=['less'],
      zip_safe=False,
      install_requires=[
        'scikit-learn>=1.0.1'
      ]),