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
    url="https://github.com/lxfhfut/ridge-detector",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "opencv-python",
        "imageio",
        "scikit-image",
        "scipy",
        "matplotlib",
        "numba",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
