
# Load modules
module load python/3.8.2

# Create environment
TMP_ENV=$(mktemp -d)
virtualenv --no-download $TMP_ENV
source $TMP_ENV/bin/activate

# Download dependencies
rm -rf $HOME/python_wheels
mkdir -p $HOME/python_wheels
cd $HOME/python_wheels
pip3 download --no-deps descartes mizani
pip3 download --no-deps 'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz'

# Install dependencies
pip3 install --no-index --no-deps $HOME/python_wheels/en_core_web_sm-2.2.0.tar.gz
pip3 install --no-index 'chardet<4.0,>=2.0'

cd $HOME/workspace/comp550
pip3 install --no-index --find-links $HOME/python_wheels -e .

# Fetch dataset
python3 experiments/download_datasets.py --persistent-dir $SCRATCH/comp550
