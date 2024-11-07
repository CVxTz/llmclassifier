from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="llmclassifier",
    version="0.1.0",  # Update the version as needed
    packages=find_packages(
        exclude=["tests.*", "tests"]
    ),  # Include all packages except test
    author="Youness",
    author_email="X@eY.Z",
    description="LLM wrapper for text classification",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Important for README rendering on PyPI
    url="https://github.com/CVxTz/llmclassifier",  # If you have a repo
    install_requires=open(
        "requirements.txt"
    ).readlines(),  # Reads dependencies from file
    extras_require={"dev": open("requirements_dev.txt").readlines()},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Or your required Python version
    test_suite="tests",  # for running tests via 'python setup.py test'
)
