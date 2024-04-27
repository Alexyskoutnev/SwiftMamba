from setuptools import setup, find_packages

setup(
    name="MambaVision",
    version="0.1",
    author="Alexy Skoutnev",
    author_email="alexyskoutnev@example.com",
    description="MambaVision is a computer vision library for object detection and image classification.",
    url="https://github.com/yourusername/my_project",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.21.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
