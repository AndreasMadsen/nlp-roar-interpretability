
import torch

from ...dataset import SequenceBatch
from ._abstract import ImportanceMeasureModule

class MutualInformationImportanceMeasure(ImportanceMeasureModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutual_information = torch.zeros(
            len(self.dataset.vocabulary),
            device=self.device
        )

    def precompute(self, *args, **kwargs):
        # Prepare dataset
        was_setup = self.dataset.is_setup('fit')
        if not was_setup:
            self.dataset.setup('fit')

        # Note the following computation happens on the CPU, regardless of use_gpu Flags

        # Count number (word, label) pairs. Note that the same word appearing multiple times
        #   in one sentences, is just counted as one word.
        # Start counters, with "1" to indicate there is a fake document with all words for each class.
        #   This is to avoid divide-by-zero issues, which is a limitation of KL-divergence / Mutual Information.
        N_docs = torch.tensor(self.dataset.num_of_observations('train') + len(self.dataset.label_names), dtype=torch.int32)
        N_docs_label_1 = torch.ones(1, len(self.dataset.label_names), dtype=torch.int32)
        N_word_1_label_1 = torch.ones(len(self.dataset.vocabulary), len(self.dataset.label_names), dtype=torch.int32)

        for batch in self.dataset.dataloader('train', *args, **kwargs):
            for observation in self.dataset.uncollate(batch):
                words = torch.bincount(observation['sentence'], minlength=len(self.dataset.vocabulary)) > 0
                N_word_1_label_1[:, observation['label']] += words
                N_docs_label_1[0, observation['label']] += 1

        # Finalize dataset
        if not was_setup:
            self.dataset.clean('fit')

        # Setup count matrices for not-word, not-label, and not-word & not-label
        # The standard notation is count = P(U=u, C=c) * N
        N_word_1_label_0 = torch.sum(N_word_1_label_1, dim=1, keepdim=True) - N_word_1_label_1
        N_word_0_label_1 = N_docs_label_1 - N_word_1_label_1
        N_word_0_label_0 = torch.sum(N_word_0_label_1, dim=1, keepdim=True) - N_word_0_label_1

        N_label_1 = N_word_0_label_1 + N_word_1_label_1
        N_label_0 = N_word_0_label_0 + N_word_1_label_0
        N_word_1 = N_word_1_label_1 + N_word_1_label_0
        N_word_0 = N_word_0_label_1 + N_word_0_label_0

        # Compute the mutual information
        mutual_information = (
            (N_word_1_label_1 / N_docs) * (
                (torch.log2(N_docs) + torch.log2(N_word_1_label_1)) - (torch.log2(N_word_1) + torch.log2(N_label_1))
            ) +
            (N_word_1_label_0 / N_docs) * (
                (torch.log2(N_docs) + torch.log2(N_word_1_label_0)) - (torch.log2(N_word_1) + torch.log2(N_label_0))
            ) +
            (N_word_0_label_1 / N_docs) * (
                (torch.log2(N_docs) + torch.log2(N_word_0_label_1)) - (torch.log2(N_word_0) + torch.log2(N_label_1))
            ) +
            (N_word_0_label_0 / N_docs) * (
                (torch.log2(N_docs) + torch.log2(N_word_0_label_0)) - (torch.log2(N_word_0) + torch.log2(N_label_0))
            )
        )

        # Zero is the smallet value possible with MI. Hard-code [PAD], [CLR], and [EOS] to have zero
        # mutual information. Note, the ROAR masking already deals appropiately with these special tokens,
        # this is just to avoid potentional nan issues.
        mutual_information[self.dataset.tokenizer.pad_token_id, :] = 0
        mutual_information[self.dataset.tokenizer.start_token_id, :] = 0
        mutual_information[self.dataset.tokenizer.end_token_id, :] = 0

        # Aggregate by weighted mean, the mutual information for each label. If this is not done,
        # then removing all token_1 from label_1 will leak information about label_1. Because,
        # instead of the model now using the existence of token_1 to predict label_1, it uses
        # the absense of token_1 to predict label_1. It is therefore necessary to remove token_1
        # equally, independent of the document label.
        P_label_1 = N_docs_label_1 / N_docs
        mutual_information_agg = torch.sum(mutual_information * P_label_1, axis=1)

        # Assign mutual_information to the class parameter.
        self.mutual_information += mutual_information_agg.to(self.device)

    def forward(self, batch: SequenceBatch) -> torch.Tensor:
        return torch.index_select(
            self.mutual_information, 0, batch.sentence.reshape([-1])
        ).reshape_as(batch.sentence).detach()
