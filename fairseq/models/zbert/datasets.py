# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq.data import FairseqDataset
import numpy as np


class RandomConcatSentencesDataset(FairseqDataset):
    def __init__(self, random_rate, *datasets):
        super().__init__()
        self.datasets = datasets
        assert random_rate < 1.0
        self.random_rate = random_rate
        self.orig_sample_num = len(self.datasets[0])
        self.random_sample_num = int(
            (self.orig_sample_num / (1 - self.random_rate)) * self.random_rate
        )
        self.orig_sizes = sum(ds.sizes for ds in self.datasets)

        assert isinstance(self.orig_sizes, np.ndarray)
        assert len(self.orig_sizes.shape) == 1

        rest_size_num = self.random_sample_num
        self.random_sizes = np.array([])
        while rest_size_num > self.orig_sample_num:
            self.random_sizes = np.concatenate((self.random_sizes, self.orig_sizes))
            rest_size_num -= self.random_sample_num

        self.random_sizes = np.concatenate(
            (self.random_sizes, self.orig_sizes[:rest_size_num])
        )
        assert len(self.random_sizes) == self.random_sample_num

        assert all(
            len(ds) == len(datasets[0]) for ds in datasets
        ), "datasets must have the same length"

    def __getitem__(self, index):
        if index >= self.orig_sample_num:
            rand_index0 = np.random.randint(0, self.orig_sample_num)
            rand_index1 = np.random.randint(0, self.orig_sample_num)
            if rand_index0 == rand_index1:
                if rand_index0 + 1 < self.orig_sample_num:
                    rand_index1 = rand_index0 + 1
                else:
                    rand_index1 = rand_index0 - 1
            return torch.cat(
                [self.datasets[0][rand_index0]]
                + [ds[rand_index1] for ds in self.datasets[1:]]
            )
        else:
            return torch.cat([ds[index] for ds in self.datasets])

    def __len__(self):
        return self.orig_sample_num + self.random_sample_num

    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def sizes(self):
        return np.concatenate((self.orig_sizes, self.random_sizes), 0)

    def num_tokens(self, index):
        if index >= self.orig_sample_num:
            index = index % self.orig_sample_num
        return sum(ds.num_tokens(index) for ds in self.datasets)

    def size(self, index):
        if index >= self.orig_sample_num:
            index = index % self.orig_sample_num
        return self.orig_sizes[index]

    def ordered_indices(self):
        return super().ordered_indices()
        # return self.datasets[0].ordered_indices()

    @property
    def supports_prefetch(self):
        return any(getattr(ds, "supports_prefetch", False) for ds in self.datasets)

    def prefetch(self, indices):
        for ds in self.datasets:
            if getattr(ds, "supports_prefetch", False):
                new_indices = list(
                    filter(lambda idx: idx < self.orig_sample_num, indices)
                )
                new_indices = np.array(new_indices)
                ds.prefetch(new_indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)


class RandomLabelDataset(FairseqDataset):
    def __init__(self, randomConcatSentencesDataset):
        self.orig_sample_num = randomConcatSentencesDataset.orig_sample_num
        self.random_sample_num = randomConcatSentencesDataset.random_sample_num

    def __getitem__(self, index):
        if index >= self.orig_sample_num:
            return torch.LongTensor([1])
        else:
            return torch.LongTensor([0])

    def __len__(self):
        return self.orig_sample_num + self.random_sample_num

    def collater(self, samples):
        return torch.cat(samples)


class NoiseConcatSentencesDataset(FairseqDataset):
    def __init__(
        self,
        random_rate: float,
        drop_target_rate: float,
        drop_noise_rate: float,
        drop_noise_threshold: float,
        noise_sample_range: int,
        *datasets
    ):
        super().__init__()
        self.datasets = datasets
        assert random_rate < 1.0
        assert drop_target_rate < 1.0
        assert drop_noise_rate < 1.0
        assert drop_noise_threshold < 1.0
        self.random_rate = random_rate
        self.drop_target_rate = drop_target_rate
        self.drop_noise_rate = drop_noise_rate
        self.drop_noise_threshold = drop_noise_threshold
        self.noise_sample_range = noise_sample_range
        self.orig_sample_num = len(self.datasets[0])
        self.random_sample_num = int(
            (self.orig_sample_num / (1 - self.random_rate)) * self.random_rate
        )
        self.orig_sizes = sum(ds.sizes for ds in self.datasets)
        double_target_sizes = self.orig_sizes + sum(ds.sizes for ds in self.datasets[1:])

        assert isinstance(self.orig_sizes, np.ndarray)
        assert len(self.orig_sizes.shape) == 1

        rest_size_num = self.random_sample_num
        self.random_sizes = np.array([])
        while rest_size_num > self.orig_sample_num:
            self.random_sizes = np.concatenate((self.random_sizes, double_target_sizes))
            rest_size_num -= self.orig_sample_num

        self.random_sizes = np.concatenate(
            (self.random_sizes, double_target_sizes[:rest_size_num])
        )
        assert len(self.random_sizes) == self.random_sample_num

        assert all(
            len(ds) == len(datasets[0]) for ds in datasets
        ), "datasets must have the same length"

    def __getitem__(self, index):
        if index >= self.orig_sample_num:
            index %= self.orig_sample_num
            lowwer_bound_index = index - self.noise_sample_range
            lowwer_bound_index = 0 if lowwer_bound_index < 0 else lowwer_bound_index
            upper_bound_index = index + self.noise_sample_range
            upper_bound_index = (
                self.orig_sample_num
                if upper_bound_index > self.orig_sample_num
                else upper_bound_index
            )
            rand_index = np.random.randint(lowwer_bound_index, upper_bound_index)
            if rand_index == index:
                if index == upper_bound_index:
                    rand_index = np.random.randint(lowwer_bound_index, index)
                else:
                    rand_index = np.random.randint(index + 1, upper_bound_index)
                if rand_index > self.orig_sample_num:
                    rand_index = self.orig_sample_num

            if np.random.rand() < self.drop_target_rate:
                return torch.cat(
                    [self.datasets[0][index]]
                    + [ds[rand_index] for ds in self.datasets[1:]]
                )
            else:
                keep_target_ratio = np.random.rand()
                keep_target_length = int(len(self.datasets[1]) * keep_target_ratio)
                if (
                    keep_target_ratio < self.drop_noise_threshold
                    and np.random.rand() < self.drop_noise_rate
                ):
                    return torch.cat(
                        [self.datasets[0][index]]
                        + [ds[index][:keep_target_length] for ds in self.datasets[1:]]
                    )
                else:
                    return torch.cat(
                        [self.datasets[0][index]]
                        + [ds[index][:keep_target_length] for ds in self.datasets[1:]]
                        + [ds[rand_index] for ds in self.datasets[1:]]
                    )
        else:
            return torch.cat([ds[index] for ds in self.datasets])

    def __len__(self):
        return self.orig_sample_num + self.random_sample_num

    def collater(self, samples):
        return self.datasets[0].collater(samples)

    @property
    def sizes(self):
        return np.concatenate((self.orig_sizes, self.random_sizes), 0)

    def num_tokens(self, index):
        if index >= self.orig_sample_num:
            index = index % self.orig_sample_num
        return sum(ds.num_tokens(index) for ds in self.datasets)

    def size(self, index):
        if index >= self.orig_sample_num:
            index = index % self.orig_sample_num
        return self.orig_sizes[index]

    def ordered_indices(self):
        return super().ordered_indices()
        # return self.datasets[0].ordered_indices()

    @property
    def supports_prefetch(self):
        return any(getattr(ds, "supports_prefetch", False) for ds in self.datasets)

    def prefetch(self, indices):
        for ds in self.datasets:
            if getattr(ds, "supports_prefetch", False):
                new_indices = list(
                    filter(lambda idx: idx < self.orig_sample_num, indices)
                )
                new_indices = np.array(new_indices)
                ds.prefetch(new_indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        for ds in self.datasets:
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False
