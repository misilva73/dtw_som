import setuptools

with open("../README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dtw-som",
    version="1.0.9",
    author="Maria Ines Silva",
    author_email="misilva73@gmail.com",
    description="DTW-SOM: Self-organizing map for time-series data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GNU Public License",
    url="https://github.com/misilva73/dtw_som",
    py_modules=['dtwsom'],
    install_requires=['scipy', 'matplotlib', 'numpy', "pyclustering", "dtaidistance"],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    python_requires=">=3.6",
)
