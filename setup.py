from setuptools import find_packages, setup


setup(
    name="image_retriever",
    version="0.1.0",
    description="High-performance image storage and retrieval engine package",
    keywords="clip image storage search retrieval",
    license="Apache",
    author="aihao",
    author_email="aihao2000@outlook.com",
    url="https://github.com/aihao2000/image-retriever",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.8.0",
    install_requires=["diffusers", "transformers", "numpy","tqdm","torch","datasets","pyarrow"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
