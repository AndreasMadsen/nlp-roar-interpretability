from setuptools import setup

setup(name='comp550',
      version='0.1.0',
      description='Measuring if attention is explanation with ROAR',
      url='https://github.com/AndreasMadsen/python-comp550-interpretability',
      license='MIT',
      packages=['comp550'],
      install_requires=[
          'numpy>=1.19.0',
          'tqdm>=4.53.0',
          'torch>=1.7.0',
          'pytorch-lightning>=1.0.0',
          'spacy>=2.2.0',
          'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/el_core_news_sm-2.2.0/el_core_news_sm-2.2.0.tar.gz',
          'torchtext>=0.6.0',
          'scikit-learn>=0.23.0',
          'nltk>=3.5',
          'plotnine>=0.7.0',
          'pandas>=1.1.0',
          'scipy>=1.5.0'
      ],
      include_package_data=True,
      zip_safe=False)
