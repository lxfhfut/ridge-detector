import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ridge-detector",
    version="0.1.3",
    author="Gavin Lin",
    author_email="lxfhfut@gmail.com",
    description="A multi-scale ridge detector for identifying curvilinear structures in images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lxfhfut/ridge-detector.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
