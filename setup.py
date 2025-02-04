from setuptools import setup, find_packages

setup(
    name="medium_transformer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'datasets>=2.12.0',
        'numpy>=1.24.0',
        'wandb>=0.15.0',
        'tqdm>=4.65.0',
    ],
)