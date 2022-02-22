from setuptools import setup, find_packages


setup(
    name="margay",
    version="0.1.0",
    description="Accelerator, radiation and x-ray optics simulation framework",
    install_requires=["matplotlib"],
    license="MIT",

    entry_points = {
        'console_scripts': [
            't20 = margay.t20:main',                  
        ],              
    },
    

)
