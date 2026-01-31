from setuptools import find_packages, setup

setup(
    name="FACTMpy",
    version="0.2",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "scikit-learn"],
    author="Małgorzata Łazęcka",
    description="FACTM package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
