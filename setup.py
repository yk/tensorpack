from setuptools import setup, find_packages

setup(name='tensorpack',
        version='0.0.1',
        description='blabla',
        url='http://github.com/yk/tensorpack',
        author='yk',
        license='MIT',
        packages=find_packages(),
        install_requires=[
                'numpy',
                'six',
                'termcolor',
                'tqdm>4.6.1',
                'msgpack-python',
                'msgpack-numpy'
            ],
        zip_safe=False)
