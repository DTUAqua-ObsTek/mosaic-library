from setuptools import setup

setup(
    name='mosaic-library',
    version='1.0',
    packages=['mosaicking'],
    url='',
    license='MIT',
    author='fft',
    author_email='fletho@aqua.dtu.dk',
    description='Mosaicking tools',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image==0.18.1',
        'matplotlib==3.4.1',
        'PyQt5',
        'keyboard',
        'pandas',
        'requests',
        'opencv-python',
        'opencv-contrib-python',
    ]
)
