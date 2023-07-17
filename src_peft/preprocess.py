from transformers import (
    GPT2TokenizerFast,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

from nltk import word_tokenize

import pandas as pd
import os
import glob
import unicodedata
from typing import List

import math
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, random_split


class MonolingualDataset(Dataset):
    def __init__(
        self,
        csv_file,
        text_column_name,
        tokenizer, max_length,
        stride_length,
        num_phrases_per_sample,
        query_length
    ):
        text_dataframe = pd.read_csv(csv_file, dtype=str)[["topic", "phrase_number", text_column_name]].dropna()
        text_dataframe = text_dataframe.sort_values(["phrase_number"]).groupby(["topic"])[text_column_name]
        text_dataframe = text_dataframe.apply(self.dynamic_split, num_phrases_per_sample=num_phrases_per_sample)
        text_dataframe = text_dataframe.apply(self.join_texts).reset_index().explode(text_column_name)
        text_dataframe = text_dataframe.loc[text_dataframe[text_column_name].str.contains("[a-zA-Z]")]

        self.query_length = query_length

        self.texts = [
            unicodedata.normalize("NFC", s)
            for s in text_dataframe[text_column_name].values.tolist()
            if len(word_tokenize(s)) >= self.query_length
        ]

        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            max_length=max_length,
            stride=stride_length,
            add_special_tokens=True
        )

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

        return item

    def __len__(self):
        return len(self.texts)

    @staticmethod
    def dynamic_split(topic_group, num_phrases_per_sample=16):
        num_sections = max(1, math.ceil(len(topic_group) / num_phrases_per_sample))
        return np.array_split(topic_group, indices_or_sections=num_sections)

    @staticmethod
    def join_texts(series_list):
        return list(map(lambda s: "\n".join(s.values), series_list))


class ConcatMonolingualDataset(ConcatDataset):
    def __init__(self, datasets: List[MonolingualDataset]):
        super().__init__(datasets)


def get_monolingual_dataset(
    tokenizer,
    max_length,
    stride_length=64,
    num_phrases_per_sample=16,
    text_column_name="phrase",
    query_length=8,
    input_path="../datasets/monolingual_Leichte_Sprache"
):
    dataset_path_list = glob.glob(f"{input_path}/*.csv")
    dataset_path_list.sort()

    dataset_list = [
        MonolingualDataset(
            csv_file=path,
            text_column_name=text_column_name,
            tokenizer=tokenizer,
            max_length=max_length,
            stride_length=stride_length,
            num_phrases_per_sample=num_phrases_per_sample,
            query_length=query_length
        )
        for path in dataset_path_list
    ]

    dataset_name_list = [os.path.basename(path) for path in dataset_path_list]

    return (
        ConcatMonolingualDataset(dataset_list),
        dataset_name_list
    )


class ParallelDataset(Dataset):
    def __init__(
        self,
        csv_file,
        tokenizer,
        max_length,
        stride_length,
        train_seq2seq,
    ):
        text_dataframe = pd.read_csv(csv_file, dtype=str)[["normal_phrase", "simple_phrase"]].dropna()
        text_dataframe = text_dataframe.loc[
            text_dataframe["normal_phrase"].str.contains("[a-zA-Z]")
            &
            text_dataframe["simple_phrase"].str.contains("[a-zA-Z]")
        ]
        src_texts = [unicodedata.normalize("NFC", s) for s in text_dataframe["normal_phrase"].values.tolist()]
        tgt_texts = [unicodedata.normalize("NFC", s) for s in text_dataframe["simple_phrase"].values.tolist()]

        self.train_seq2seq = train_seq2seq

        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.texts = [f"Source: {src}\nTarget: {tgt}" for src, tgt in zip(self.src_texts, self.tgt_texts)]

        self.src_encodings = tokenizer(
            self.src_texts,
            truncation=True,
            max_length=max_length,
            stride=stride_length,
            add_special_tokens=True
        )
        self.tgt_encodings = tokenizer(
            self.tgt_texts,
            truncation=True,
            max_length=max_length,
            stride=stride_length,
            add_special_tokens=True
        )

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.src_encodings.items()}

        if self.train_seq2seq:
            item["labels"] = torch.tensor(self.tgt_encodings["input_ids"][idx])

        return item

    def __len__(self):
        return len(self.texts)


class ConcatParallelDataset(ConcatDataset):
    def __init__(self, datasets: List[ParallelDataset]):
        super().__init__(datasets)


def get_parallel_dataset(
    tokenizer,
    max_length,
    stride_length=64,
    train_seq2seq=False,
    input_path="../datasets/aligned_German_simplification"
):
    dataset_path_list = glob.glob(f"{input_path}/*.csv")
    dataset_path_list.sort()

    dataset_list = [
        ParallelDataset(
            csv_file=path,
            tokenizer=tokenizer,
            max_length=max_length,
            stride_length=stride_length,
            train_seq2seq=train_seq2seq
        )
        for path in dataset_path_list
    ]

    dataset_name_list = [os.path.basename(path) for path in dataset_path_list]

    return (
        ConcatParallelDataset(dataset_list),
        dataset_name_list
    )


class TextComplexityDataset(Dataset):
    def __init__(
        self,
        csv_file,
        text_column_name,
        target_label,
        tokenizer,
        max_length,
        stride_length,
        return_features,
        feature_column_names=None,
        num_labels=1,
        label2id=None,
    ):
        label2id = {
            "A2": 0,
            "B1": 1,
            "B2": 2
        } if label2id is None else label2id

        text_dataframe = pd.read_csv(csv_file).dropna()
        text_dataframe = text_dataframe.loc[text_dataframe[text_column_name].str.contains("[a-zA-Z]")]
        texts = [unicodedata.normalize("NFC", s) for s in text_dataframe[text_column_name].values.tolist()]
        labels = text_dataframe[target_label].values.tolist()
        if num_labels > 1:
            assert num_labels == len(label2id)
            labels = list(map(lambda l: label2id[l], labels))

        if return_features and feature_column_names:
            self.feature_column_names = feature_column_names
            self.features = text_dataframe[feature_column_names].values.tolist()
        else:
            self.feature_column_names = None,
            self.features = None

        self.return_features = return_features

        self.texts = texts
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            stride=stride_length,
            add_special_tokens=True
        )
        self.target_label = target_label
        self.labels = labels

        self.max = max(labels)
        self.min = min(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        if self.features and self.return_features:
            item["features"] = torch.tensor(self.features[idx])
        return item

    def __len__(self):
        return len(self.texts)


class ConcatTextComplexityDataset(ConcatDataset):
    def __init__(
        self,
        datasets: List[TextComplexityDataset],
        do_rescaling=True,
        max_value=None,
        min_value=None,
        rescaling_factor=100
    ):
        self.target_label = datasets[0].target_label
        self.feature_column_names = datasets[0].feature_column_names
        self.max = max(dataset.max for dataset in datasets) if max_value is None else max_value
        self.min = min(dataset.min for dataset in datasets) if min_value is None else min_value
        self.rescaling_factor = rescaling_factor

        if do_rescaling:
            for dataset in datasets:
                dataset.labels = list(map(self.rescale_label, dataset.labels))
                dataset.max = max(dataset.labels)
                dataset.min = min(dataset.labels)

        super().__init__(datasets)

    def rescale_label(self, label):
        label = self.max if label > self.max else label
        label = self.min if label < self.min else label
        return (label - self.min) / (self.max - self.min) * self.rescaling_factor


def get_text_complexity_dataset(
    tokenizer,
    max_length,
    stride_length=64,
    return_features=False,
    feature_column_names=None,
    num_labels=1,
    label2id=None,
    text_column_name="phrase",
    target_label="Kincaid",
    do_rescaling=False,
    max_value=None,
    min_value=None,
    rescaling_factor=None,
    input_path="../datasets/TextComplexity/monolingual_dynamic_split"
):
    dataset_path_list = glob.glob(f"{input_path}/*.csv")
    dataset_path_list.sort()

    dataset_list = [
        TextComplexityDataset(
            csv_file=path,
            text_column_name=text_column_name,
            target_label=target_label,
            tokenizer=tokenizer,
            max_length=max_length,
            stride_length=stride_length,
            return_features=return_features,
            feature_column_names=feature_column_names,
            num_labels=num_labels,
            label2id=label2id
        )
        for path in dataset_path_list
    ]

    dataset_name_list = [os.path.basename(path) for path in dataset_path_list]

    return (
        ConcatTextComplexityDataset(dataset_list, do_rescaling, max_value, min_value, rescaling_factor),
        dataset_name_list
    )


def split_dataset(
    dataset,
    val_split=0.1,
    test_split=0.0,
    seed=40
):
    val_size = int(val_split * len(dataset))
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - val_size - test_size
    generator = torch.Generator().manual_seed(seed)

    if val_split > 0 and test_split > 0:
        train_set, val_set, test_set = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=generator
        )

        return train_set, val_set, test_set

    elif val_split > 0:
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=generator
        )

        return train_set, val_set

    elif test_split > 0:
        train_set, test_set = random_split(
            dataset,
            [train_size, test_size],
            generator=generator
        )

        return train_set, test_set


class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def adapter_summary(model, as_dict=False):
    adapter_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_params += num_params
        if param.requires_grad:
            adapter_params += num_params

    param = 100 * adapter_params / all_params

    if as_dict:
        return {
            "adapter params": adapter_params,
            "all params": all_params,
            "%param": param
        }
    else:
        return f"adapter params: {adapter_params} || all params: {all_params} || %param: {param}"
