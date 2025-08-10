from setuptools import find_packages, setup

def get_requirements(file_path:str) -> list :
    """
    This function reads a requirements file and returns a list of requirements.
    """
    requirements= []
    HYPEN_E_DOT = "-e ."
    with open(file_path, "r") as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]  # Remove newline characters
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

setup(
    name = "network Anamoly Detection",
    version = "0.0.1",
    author = "Octopus",
    author_email = "akhil96me@gmil.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)