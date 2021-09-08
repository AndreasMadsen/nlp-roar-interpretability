import os.path as path
import numpy as np
import os
import pickle
import requests
import tarfile

from sklearn.model_selection import train_test_split

from ._tokenizer import Tokenizer
from ._paired_sequence_dataset import PairedSequenceDataset

IDX_TO_TASK_MAP = {1: 'qa1_single-supporting-fact_',
                   2: 'qa2_two-supporting-facts_',
                   3: 'qa3_three-supporting-facts_'}

def _parse_babi_txtfile(file):
    data, story = [], []
    with open(file, 'r', encoding='utf-8') as fp:
        for line in fp:
            tid, text = line.rstrip('\n').split(' ', 1)
            if tid == '1':
                story = []
            # sentence
            if text.endswith('.'):
                story.append(text[:-1])
            # question
            else:
                query, answer, _ = (x.strip() for x in text.split('\t'))
                substory = " . ".join([x for x in story if x])
                data.append({"paragraph": substory, "question": query[:-1], "answer": answer})
    return data

class BabiTokenizer(Tokenizer):
    def tokenize(self, sentence):
        return sentence.split()


class BabiDataset(PairedSequenceDataset):
    def __init__(self, cachedir, task=1, batch_size=50, **kwargs):
        """Creates an bAbI dataset instance

        Args:
            cachedir (str): Directory to use for caching the compiled dataset.
            task (1, 2, or 3): Which bAbI task to use.
            seed (int): Seed used for shuffling the dataset.
            batch_size (int, optional): The batch size used in the data loader. Defaults to 32.
            num_workers (int, optional): The number of pytorch workers in the data loader. Defaults to 4.
        """
        if task not in [1, 2, 3]:
            raise ValueError('task must be either 1, 2, or 3')

        super().__init__(cachedir, f'babi-{task}', BabiTokenizer(), batch_size=batch_size, **kwargs)
        self._task = task
        self.label_names = ['put', 'picked', 'down', '.', 'travelled', 'was',
                            'football', 'got', 'garden', 'milk', 'discarded', 'is',
                            'moved', 'journeyed', 'apple', 'took', 'Sandra', 'before',
                            'Mary', 'Where', 'John', 'office', 'there', 'Daniel',
                            'bathroom', 'went', 'left', 'the', 'back', 'hallway',
                            'dropped', 'to', 'bedroom', 'kitchen', 'grabbed', 'up']

    def embedding(self):
        """Creates word embedding matrix.

        Returns:
            np.array: shape = (vocabulary, 300)
        """
        # Random embeddings of size 50
        # https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/model/modules/Encoder.py#L26
        # https://github.com/successar/AttentionExplanation/blob/425a89a49a8b3bffc3f5e8338287e2ecd0cf1fa2/Trainers/DatasetQA.py#L112
        embeddings = []
        for word in self.tokenizer.ids_to_token:
            if word == self.tokenizer.pad_token:
                embeddings.append(np.zeros(50))
            else:
                embeddings.append(np.random.randn(50))

        return np.vstack(embeddings)

    def prepare_data(self):
        """Download, compiles, and cache the dataset.
        """
        # Short-circuit the build logic if the minimum-required files exists
        if (path.exists(f'{self._cachedir}/encoded/{self.name}.pkl') and
            path.exists(f'{self._cachedir}/vocab/{self.name}.vocab')):
            self.tokenizer.from_file(f'{self._cachedir}/vocab/{self.name}.vocab')
            return

        # Download and extract dataset
        babi_url = 'http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz'
        os.makedirs(f'{self._cachedir}/text-datasets', exist_ok=True)
        if not path.exists(f'{self._cachedir}/text-datasets/tasks_1-20_v1-2'):
            r = requests.get(babi_url, stream=True)
            with open(f'{self._cachedir}/text-datasets/tasks_1-20_v1-2.tar.gz', 'wb') as f:
                f.write(r.raw.read())

            with open(f'{self._cachedir}/text-datasets/tasks_1-20_v1-2.tar.gz', 'rb') as f:
                thetarfile = tarfile.open(fileobj=f, mode="r|gz")
                thetarfile.extractall(path=f'{self._cachedir}/text-datasets')
                thetarfile.close()

        # Parse all of Babi
        tasks = IDX_TO_TASK_MAP.keys()
        data = {}
        for t in tasks:
            data[t] = {}
            for k in ['train', 'test']:
                data[t][k] = _parse_babi_txtfile(
                    f'{self._cachedir}/text-datasets/tasks_1-20_v1-2/en-10k/' + IDX_TO_TASK_MAP[t] + k + '.txt')

        # Split dataset
        trainidx, devidx = train_test_split(range(0, len(data[self._task]['train'])),
                                            train_size=0.85, random_state=self._np_rng)
        testidx = range(0, len(data[self._task]['test']))

        # Create dataset for this babi-task
        babi_data = {}
        for name, idxs, dataset in [
            ('train', trainidx, data[self._task]['train']),
            ('val', devidx, data[self._task]['train']),
            ('test', testidx, data[self._task]['test'])
        ]:
            babi_data[name] = [{
                'paragraph': dataset[idx]["paragraph"],
                'question': dataset[idx]["question"],
                'label': self.label_names.index(dataset[idx]["answer"])
            } for idx in idxs]

        # Build vocabulary
        if not path.exists(f'{self._cachedir}/vocab/{self.name}.vocab'):
            os.makedirs(f'{self._cachedir}/vocab', exist_ok=True)

            self.tokenizer.from_iterable(
                [instance["paragraph"] for instance in babi_data['train']] +
                [instance["question"] for instance in babi_data['train']])
            self.tokenizer.to_file(f'{self._cachedir}/vocab/{self.name}.vocab')
        else:
            self.tokenizer.from_file(f'{self._cachedir}/vocab/{self.name}.vocab')

        # Encoded dataset
        if not path.exists(f'{self._cachedir}/encoded/{self.name}.pkl'):
            os.makedirs(f'{self._cachedir}/encoded', exist_ok=True)

            data = {}
            for name in ['train', 'val', 'test']:

                dataset = babi_data[name]
                data[name] = [{
                    'sentence': self.tokenizer.encode(instance["paragraph"]),
                    'sentence_aux': self.tokenizer.encode(instance["question"]),
                    'label': instance['label']
                } for instance in dataset]

            with open(f'{self._cachedir}/encoded/{self.name}.pkl', 'wb') as fp:
                pickle.dump(data, fp)
