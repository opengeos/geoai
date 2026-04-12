"""Setup for cli-anything-geoai -- CLI harness for GeoAI."""

from setuptools import setup, find_namespace_packages

setup(
    name="cli-anything-geoai",
    version="1.0.0",
    description="CLI harness for GeoAI -- AI-powered geospatial analysis",
    long_description=open("GEOAI.md").read(),
    long_description_content_type="text/markdown",
    author="Qiusheng Wu",
    author_email="giswqs@gmail.com",
    url="https://github.com/opengeos/geoai",
    packages=find_namespace_packages(include=("cli_anything.*",)),
    python_requires=">=3.10",
    install_requires=[
        "click>=8.1",
        "prompt-toolkit>=3.0",
        "geoai-py",
    ],
    entry_points={
        "console_scripts": [
            "cli-anything-geoai=cli_anything.geoai.geoai_cli:main",
        ],
    },
    package_data={
        "cli_anything.geoai": ["skills/*.md"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
