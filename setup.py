from setuptools import setup, find_packages

setup(
    name="airstart3d",
    version="0.1.0",
    packages=find_packages(),  # Automatically finds packages
    install_requires=[
        "numpy",
        "matplotlib",
        "simplekml",      
        "scikit-learn",
        "vpython",
        "rasterio",
        "utm",
        "pandas",
        "tqdm",
        "pillow",
        "ephem"
    ],
)
