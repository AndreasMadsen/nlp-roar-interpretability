
# Load modules
module load python/3.8.10

# Create environment
TMP_ENV=$(mktemp -d)
virtualenv --no-download $TMP_ENV
source $TMP_ENV/bin/activate

# Download dependencies
rm -rf $HOME/python_wheels
mkdir -p $HOME/python_wheels
cd $HOME/python_wheels
python -m pip download --no-deps 'en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.1.0/en_core_web_sm-3.1.0-py3-none-any.whl'

# Install dependencies
python -m pip install --no-deps $HOME/python_wheels/en_core_web_sm-3.1.0-py3-none-any.whl
python -m pip install --no-index 'chardet<4.0,>=2.0' 'click<7.2.0,>=7.1.1'

cd $HOME/workspace/nlproar
python -m pip install --no-index -e .

# Fetch dataset
python experiments/download_datasets.py --persistent-dir $SCRATCH/nlproar
