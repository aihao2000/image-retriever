import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="ImageRetrievalEngine",
    version="0.1",
    description="Image Retrieval Engine based on OpenAI CLIP",
    author="AiHao",
    author_email='AiHao200000707@outlook.com',
    packages=find_packages(),
    install_requires=[
        "ftfy",
        "regex",
        "tqdm",
        "torch",
        "torchvision",
        "CLIP @ git+https://github.com/openai/CLIP.git",
        "glob",
        "prettytable"
    ],
    url='https://github.com/AisingioroHao0/ImageRetrieval'
)
