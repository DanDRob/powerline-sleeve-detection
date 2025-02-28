from setuptools import setup, find_packages

setup(
    name="powerline-sleeve-detection",
    version="0.1.0",
    description="Powerline sleeve detection system",
    author="Daniel Roberts",
    author_email="danielrob001@gmail.com",
    url="https://github.com/DanDRob/powerline-sleeve-detection",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24.3",
        "pandas>=1.5.3",
        "opencv-python>=4.5.0",
        "torch>=2.0.0",
        "ultralytics>=8.0.0",
        "pyyaml>=6.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "scikit-learn>=1.0.0",
        "pillow>=8.0.0",
        "torch>=1.7.0",
        "torchvision==0.15.2",
        "opencv-python>=4.5.0",
        "folium>=0.12.0",
        "dash>=2.0.0",
        "plotly>=5.0.0",
        "geopandas>=0.9.0",
        "shapely>=1.7.0",
        "requests>=2.25.0",
        "polyline>=1.4.0",
        "geopy>=2.2.0",
        "pyproj>=3.0.0",
        "pyyaml>=5.4.0",
        "aiohttp>=3.7.0",
        "asyncio>=3.4.3",
        "contextily==1.3.0",
        "ultralytics>=0.0.0",
        "effdet==0.4.1",
        "tqdm==4.65.0",
        "dash-bootstrap-components==1.4.1",
        "python-dotenv>=1.0.1",
        "googlemaps>=3.1.0",
        "albumentations>=1.3.0",
        "optuna>=3.0.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950"
        ],
    },
    entry_points={
        "console_scripts": [
            "powerline-detector=powerline_sleeve_detection.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11"
    ],
    python_requires=">=3.8"
)
