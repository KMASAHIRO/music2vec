import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="music2vec",
    version="0.0.2",
    install_requires=["tensorflow>=2.4.1","numpy>=1.19.5","librosa>=0.8.0"],
    author="KMASAHIRO",
    description="music genre classification model based on SoundNet",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KMASAHIRO/music2vec",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.10',
)