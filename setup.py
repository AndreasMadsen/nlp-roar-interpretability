from setuptools import setup

setup(name='comp550',
      version='0.1.0',
      description='Measuring if attention is explanation with ROAR',
      url='https://github.com/AndreasMadsen/python-comp550-interpretability',
      license='MIT',
      packages=['comp550'],
      install_requires=[
          'numpy>=1.19.4',
          'tqdm>=4.53.0',
          'torch>=1.7.0',
          'pytorch-lightning>=1.0.7',
          'datasets>=1.1.3',
          'spacy>=2.3.2',
          'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz',
          'torchtext>=0.8.0',
          'sklearn>=0.23.2',
          'nltk>=3.5'
      ],
      include_package_data=True,
      zip_safe=False)
