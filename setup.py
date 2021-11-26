from setuptools import setup

setup(name='nlproar',
      version='0.1.0',
      description='Evaluating the Faithfulness of Importance Measures in NLP by Recursively Masking Allegedly Important Tokens and Retraining',
      license='MIT',
      packages=['nlproar'],
      install_requires=[
          'numpy>=1.21.0',
          'tqdm>=4.61.2',
          'torch>=1.9.0,<1.10.0',
          'pytorch-lightning>=1.2.6,<1.3.0',
          'spacy>=3.1.0,<3.2.0',
          'torchtext>=0.10.0,<0.11.0',
          'scikit-learn>=0.24.1',
          'plotnine>=0.8.0',
          'pandas>=1.3.0',
          'scipy>=1.7.0',
          'numba>=0.53.1,<1.0.0',
          'nltk>=3.5',
          'gensim>=4.0.1,<4.1.0'
      ],
      include_package_data=True,
      zip_safe=False)
