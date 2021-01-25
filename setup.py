from setuptools import setup, find_packages

setup(
    name = 'SOTA_SSL_Models',
    packages = find_packages(),
    version = '0.1.0',
    license = 'MIT',
    description = 'SOTA self-supervised contrastive learning models including simclr, byol, swav, moco, pirl, simsiam etc.',
    author = 'Seri Lee',
    author_email = 'sally20921@snu.ac.kr',
    url = 'https://github.com/sally20921/SimSiam',
    keywords = ['self-supervised learning', 'contrastive learning', 'SimCLR', 'BYOL', 'SwAV', 'MoCo', 'PIRL', 'SimSiam'],
    install_requires = [
        'torch',
    ],
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
