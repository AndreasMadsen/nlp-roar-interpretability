from setuptools import setup

setup(name='comp550',
      version='0.1.0',
      description='Measuring if attention is explanation with ROAR',
      url='https://github.com/AndreasMadsen/python-comp550-interpretability',
      license='MIT',
      packages=['comp550'],
      install_requires=[
          'numpy>=1.19.5',
          'tqdm>=4.59.0',
          'torch>=1.7.0,<1.8.0',
          'pytorch-lightning>=1.2.1,<1.3.0',
          'spacy>=2.2.2,<2.3.0',
          'torchtext>=0.6.0<0.7.0',
          'scikit-learn>=0.24.1',
          'plotnine>=0.7.1',
          'pandas>=1.1.3',
          'scipy>=1.5.4',
          'nltk>=3.5',
          'gensim>=3.8.1'
      ],
      include_package_data=True,
      zip_safe=False)
