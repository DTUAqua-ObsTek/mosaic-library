from setuptools import setup

setup(
    name='mosaic-library',
    version='0.0.1',
    packages=['mosaicking'],
    url='',
    license='MIT',
    author='fft',
    author_email='fletho@aqua.dtu.dk',
    description='Mosaicking tools',
    install_requires=[
        'wheel',
        'build',
        'numpy',
        'scipy',
        'scikit-image',
        'imutils',
        'matplotlib',
        'PyQt5',
        'keyboard',
        'pandas',
        'requests',
        'opencv-contrib-python',
    ]
)
