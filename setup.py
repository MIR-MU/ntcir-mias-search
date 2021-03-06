from os import path
from setuptools import setup, find_packages

AUTHOR = "Vit Novotny"
HERE = path.abspath(path.dirname(__file__))
SOURCE_URL = "https://github.com/MIR-MU/ntcir-mias-search"

# Get the long description from the README file
with open(path.join(HERE, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()


setup(
    author=AUTHOR,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Text Processing :: Markup :: XML",
    ],
    description="""
        The MIaS Search package implements the Math Information Retrival system
        that won the NTCIR-11 Math-2 main task (Růžička et al., 2014).
    """,
    entry_points={
        'console_scripts': [
            'ntcir-mias-search=ntcir_mias_search.__main__:main',
        ],
    },
    keywords="ntcir mias math_information_retrieval",
    install_requires=[
        "tqdm ~= 4.23.3",
        "lxml ~= 4.6.2",
        "matplotlib ~= 2.2.2",
        "numpy ~= 1.14.3",
        "requests ~= 2.20.0",
    ],
    license="MIT",
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    maintainer=AUTHOR,
    name="ntcir_mias_search",
    packages=find_packages(),
    python_requires="~= 3.5",
    project_urls={
        "Source": SOURCE_URL,
    },
    url=SOURCE_URL,
    version="0.2.2",
)
