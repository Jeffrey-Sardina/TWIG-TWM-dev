import setuptools 
  
with open("README.md", "r") as fh: 
    long_description = fh.read() 
  
setuptools.setup( 
    # Here is the module name. 
    name="twig_twm", 
  
    # version of the module 
    version="1.0.0", 
  
    # Name of Author 
    author="Jeffrey Seathrún Sardina", 
  
    # your Email address 
    author_email="jeffrey.sardina@gmail.com", 
  
    #Small Description about module 
    description="TWIG-TWM: KGE Simulation and hyperparameter optimisation from graph topology characteristics", 
  
    # Specifying that we are using markdown file for description 
    long_description=long_description, 
    long_description_content_type="text/markdown", 
  
    # Any link to reach this module, ***if*** you have any webpage or github profile 
    url="https://github.com/Jeffrey-Sardina", 
    packages=setuptools.find_packages(), 
    py_modules=["twig_twm"],
  
    # if module has dependencies i.e. if your package rely on other package at pypi.org 
    # then you must add there, in order to download every requirement of package 
    install_requires=[ 
        "numpy", 
        "torch", 
        "torchvision", 
        "torchaudio", 
        "torcheval", 
        "pykeen", 
        "frozendict", 
    ], 
  
    license="CC BY-NC-SA 4.0", 
  
    # classifiers like program is suitable for python3, just leave as it is. 
    classifiers=[ 
        "Programming Language :: Python :: 3", 
        "Operating System :: OS Independent", 
    ], 
) 
