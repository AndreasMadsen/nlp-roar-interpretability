import argparse

from comp550.dataset import ExampleDataset
from comp550.model import ExampleModel

parser = argparse.ArgumentParser(description='Run example task')
parser.add_argument('--seed',
                    action='store',
                    default=0,
                    type=int,
                    help='Random seed')
parser.add_argument('--max-epochs',
                    action='store',
                    default=10000,
                    type=int,
                    help='The max number of epochs to use')
parser.add_argument('--use-gpu',
                    action='store',
                    default=torch.cuda.is_available(),
                    type=bool,
                    help=f'Should GPUs be used (detected automatically as {torch.cuda.is_available()})')
args = parser.parse_args()

dataset = ExampleDataset(seed=args.seed)
model = ExampleModel()
trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=int(args.use_gpu))
trainer.fit(model, dataset)
