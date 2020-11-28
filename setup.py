from setuptools import setup

setup(name='comp550',
      version='0.1.0',
      description='Measuring if attention is explanation with ROAR',
      url='https://github.com/AndreasMadsen/python-comp550-interpretability',
      license='MIT',
      packages=['comp550'],
      install_requires=[
          'numpy',
          'tqdm',
          'torch',
          'pytorch-lightning',
          'datasets',
          'spacy',
          'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz',
          'torchtext',
          'sklearn',
          'nltk'
      ],
      include_package_data=True,
      zip_safe=False)
