from setuptools import setup

setup(name='nlproar',
      version='0.1.0',
      description='Evaluating the Faithfulness of Importance Measures in NLP by Recursively Masking Allegedly Important Tokens and Retraining',
      url='https://github.com/AndreasMadsen/python-nlproar-interpretability',
      license='MIT',
      packages=['nlproar'],
      install_requires=[
          'numpy>=1.22.2',
          'tqdm>=4.63.1',
          'torch>=1.10.0,<1.11.0',
          'pytorch-lightning>=1.6.0,<1.7.0',
          'torchmetrics>=0.6.0,<0.8.0',
          'spacy>=3.1.0,<3.2.0',
          'torchtext>=0.11.0,<0.12.0',
          'scikit-learn>=1.0.2',
          'plotnine>=0.8.0',
          'pandas>=1.4.1',
          'scipy>=1.8.0',
          'numba>=0.53.1,<1.0.0',
          'nltk>=3.7,<3.8',
          'gensim>=4.0.1,<4.1.0',
          'transformers>=4.17.0,<4.18.0'
      ],
      include_package_data=True,
      zip_safe=False)
