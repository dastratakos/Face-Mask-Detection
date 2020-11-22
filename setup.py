from setuptools import setup, find_packages

setup(
    name='FaceMaskDetection',
    description='Face Mask Detection',
    author='Charles Pan, Gilbert Rosal, Dean Stratakos',
    packages=find_packages(),
    py_modules=['config'],
    install_requires=[
        'matplotlib',
        'numpy',
        'opencv-python',
        'Pillow',
        'pip',
        'python',
        'scikit-image',
        'scipy',
        'sklearn',
        'tensorflow',
        'tqdm',
    ]
)
