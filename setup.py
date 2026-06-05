from setuptools import find_packages, setup

setup(
name="mcqgenerator",
version="0.0.1",
author="taufique",
author_email="taufiqueakbar@gmail.com",
install_requires=[
    "langchain",
    "streamlit",
    "python-dotenv",
    "scikit-learn",
    "pandas",
    "numpy",
    "matplotlib",
    "keras",
    "tensorflow",
    "uvicorn",
    "requests",
    "fastapi",
    "Flask",
    "nltk",
    "flasgger"
],
packages=find_packages()

)