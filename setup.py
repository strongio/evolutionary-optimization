import setuptools

setuptools.setup(
    name='evolutionary-optimization',
    version='0.1',
    packages=setuptools.find_packages(include='evopt.*'),
    url='',
    license='',
    author='Strong Analytics',
    author_email='',
    description='Evolutionary Optimization',
    install_requires=[
        'matplotlib',
        'numpy'
    ]
)