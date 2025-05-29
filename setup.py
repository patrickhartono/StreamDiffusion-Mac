import os
import re

from setuptools import find_packages, setup


_deps = [
    "torch",
    "diffusers==0.33.1",
    "transformers==4.35.2",
    "accelerate==1.7.0",
    "huggingface_hub==0.32.1",
    "fire",
    "omegaconf",
    "onnx==1.15.0",
    "onnxruntime==1.16.3",
    "protobuf==3.20.2",
    "colored",
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}
extras["torch"] = deps_list("torch", "accelerate")
extras["macos"] = deps_list("torch", "accelerate", "diffusers", "transformers", "huggingface_hub")  # macOS-specific extras

extras["dev"] = extras["torch"] + extras["macos"]

install_requires = [
    deps["fire"],
    deps["omegaconf"],
    deps["diffusers"],
    deps["transformers"],
    deps["accelerate"],
    deps["huggingface_hub"],
]

setup(
    name="streamdiffusion-mac",
    version="0.1.1",
    description="StreamDiffusion for macOS - real-time interactive image generation pipeline for Apple Silicon and Intel Macs",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning diffusion pytorch stable diffusion audioldm streamdiffusion real-time macos apple silicon",
    license="Apache 2.0 License",
    author="Patrick Hartono, based on original work by StreamDiffusion team (Aki, kizamimi, ddPn08, Verb, ramune, teftef6220, Tonimono, Chenfeng Xu, Ararat)",
    author_email="patrickhartono@example.com",
    url="https://github.com/patrickhartono/StreamDiffusion-Mac",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"streamdiffusion": ["py.typed"]},
    include_package_data=True,
    python_requires=">=3.10.0",
    install_requires=list(install_requires),
    extras_require=extras,
)
