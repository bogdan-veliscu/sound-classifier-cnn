import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = ["efficientnet_pytorch",
                "torch",
                "torchvision",
                "librosa",
                "torchaudio",
                "torchsummary",
                "lmdb",
                "pandas",
                "tensorboardX"]

setuptools.setup(
    name="esc_cnn",
    version="0.1.0",
    description="A PyTorch model for sound classificationdi",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomasic/esc-cnn",
    package_data={
        "": ["*.txt", "*.rst", "*.pth", "*.npy", "*.log", "config/*.*", "checkpoints/*.*"],
    },
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
