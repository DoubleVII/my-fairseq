from fairseq.tasks import register_task, FairseqTask
from fairseq.data import FairseqDataset
import numpy as np
import torch


@register_task("example_task")
class ExampleTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "--up-bound", default=50, type=int, help="max value of the data",
        )

    def __init__(self, args, up_bound):
        super().__init__(args)
        self.up_bound = up_bound

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        # load dictionaries
        up_bound = args.up_bound

        return cls(args, up_bound)

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        return None

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        self.datasets[split] = ExampleDataset(self.up_bound)

    def build_model(self, args):

        model = super().build_model(args)
        return model

    def build_criterion(self, args):
        """
        Build the :class:`~fairseq.criterions.FairseqCriterion` instance for
        this task.

        Args:
            args (argparse.Namespace): parsed command-line arguments

        Returns:
            a :class:`~fairseq.criterions.FairseqCriterion` instance
        """
        from fairseq import criterions

        return criterions.build_criterion(args, self)

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        return super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )


class ExampleDataset(FairseqDataset):
    def __init__(self, up_bound):
        self.up_bound = up_bound

    def __getitem__(self, index):
        rand_num = torch.FloatTensor([np.random.rand() * self.up_bound])
        target = torch.FloatTensor([rand_num + 1.5])
        return {"num": rand_num, "target": target}

    def __len__(self):
        return 100

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        nums = []
        targets = []
        for sample in samples:
            nums.append(sample["num"])
            targets.append(sample["target"])
        return {"nums": torch.stack(nums), "targets": torch.stack(targets)}

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return 1

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return 1

    def get_batch_shapes(self):
        """
        Return a list of valid batch shapes, for example::

            [(8, 512), (16, 256), (32, 128)]

        The first dimension of each tuple is the batch size and can be ``None``
        to automatically infer the max batch size based on ``--max-tokens``.
        The second dimension of each tuple is the max supported length as given
        by :func:`fairseq.data.FairseqDataset.num_tokens`.

        This will be used by :func:`fairseq.data.FairseqDataset.batch_by_size`
        to restrict batch shapes. This is useful on TPUs to avoid too many
        dynamic shapes (and recompilations).
        """
        return [(None, 1)]
