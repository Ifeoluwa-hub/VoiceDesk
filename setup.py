# setup.py
from setuptools import setup, find_packages

setup(
    name="qa_agent", 
    version="0.1.0",
    packages=find_packages(), 
    install_requires=[
        # list dependencies here
        "fastapi",
        "uvicorn",
    ],
)
