from setuptools import setup, find_packages

setup(
    name="playwright-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "playwright",
        "pandas",
        "numpy",
        "scikit-learn",
        "beautifulsoup4",
        "loguru"
    ]
)
