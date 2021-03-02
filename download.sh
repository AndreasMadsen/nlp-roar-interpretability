
# Load modules
module load python/3.6
module load StdEnv/2020
module load scipy-stack/2020b

# Create enviorment
TMP_ENV=$(mktemp -d)
virtualenv --system-site-packages --no-download $TMP_ENV
source $TMP_ENV/bin/activate

# Download (and install) dependencies
rm -rf $HOME/python_wheels
mkdir -p $HOME/python_wheels
cd $HOME/python_wheels
pip3 download --no-deps 'pytorch-lightning<=1.1.0'
pip3 download --no-deps 'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz'

pip3 install --no-index --find-links $HOME/python_wheels \
    'numpy>=1.19.0' 'tqdm>=4.53.0' 'torch>=1.7.0' 'pytorch-lightning<=1.1.0' \
    'spacy>=2.2.0,<2.3.0' $HOME/python_wheels/en_core_web_sm-2.2.0.tar.gz 'torchtext>=0.6.0' \
    'scikit-learn>=0.23.0' 'nltk>=3.5' 'gensim>=3.8.0' 'pandas>=1.1.0'

# Fetch dataset
cd $HOME/workspace/comp550
pip3 install --no-index --no-deps -e .
python3 experiments/download_datasets.py --persistent-dir $SCRATCH/comp550
