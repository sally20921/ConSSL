from setuptools import setup, find_packages

setup(
    name = 'ConSSL',
    packages = find_packages(),
    version = '0.0.1',
    author = 'Seri Lee',
    author_email = 'sally20921@snu.ac.kr',
    license = 'MIT',
    description = 'SOTA self-supervised contrastive learning models including simclrv2, byol, swav, moco, simsiam etc.',
    url = 'https://github.com/sally20921/SimSiam',
    keywords = ['self-supervised learning', 'contrastive learning', 'SimCLR', 'BYOL', 'SwAV', 'MoCo', 'PIRL', 'SimSiam'],
    install_requires = [
        'torch',
    ],
    classifiers = [
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
