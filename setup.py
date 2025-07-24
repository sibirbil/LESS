from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="less-learn",
    version="0.4.0",
    description="Learning with Subset Stacking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sibirbil/LESS",
    maintainer="Ilker Birbil, Samet Çopur",
    maintainer_email="s.i.birbil@uva.nl, sametcopur@yahoo.com",
    license="MIT",
    packages=["less"],
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=["scikit-learn>=1.5", "numpy>=2.1", "xgboost>=3.0"],
)
