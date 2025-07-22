from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="gnbg-gpu",
    version="0.1.0",
    rust_extensions=[
        RustExtension(
            "gnbg_gpu",
            binding=Binding.PyO3,
            features=["python"],
        )
    ],
    packages=["gnbg_gpu"],
    package_dir={"gnbg_gpu": "python"},
    install_requires=["numpy>=1.20.0"],
    zip_safe=False,
)