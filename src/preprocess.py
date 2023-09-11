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
        tokenizer,
        max_length,
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


class TextComplexityDataset(Dataset):
    def __init__(
        self,
        csv_file,
        text_column_name,
        target_label,
        tokenizer,
        max_length,
        stride_length,
        return_features
    ):
        text_dataframe = pd.read_csv(csv_file).dropna()
        text_dataframe = text_dataframe.loc[text_dataframe[text_column_name].str.contains("[a-zA-Z]")]
        texts = [unicodedata.normalize("NFC", s) for s in text_dataframe[text_column_name].values.tolist()]
        labels = text_dataframe[target_label].values.tolist()

        if target_label == "MOS":
            feature_column_names = [
                "Kincaid",
                "ARI",
                "Coleman-Liau",
                "FleschReadingEase",
                "GunningFogIndex",
                "LIX",
                "SMOGIndex",
                "RIX",
                "DaleChallIndex",
                "WLF"
            ]
            features = text_dataframe[feature_column_names].values.tolist()
        else:
            features = None

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
        self.features = features
        self.return_features = return_features
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
            return_features=return_features
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


def specify_config(
    model_path,
    model_name,
    special_tokens_dict=None,
    head_type="causal",
    embd_pdrop=0.1,
    attn_pdrop=0.1,
    resid_pdrop=0.1,
    output_path="../adapters",
    save_config=False
):
    special_tokens_dict = {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>"
    } if special_tokens_dict is None else special_tokens_dict

    head_type_list = ["causal", "regression"]
    if head_type not in head_type_list:
        raise ValueError("Unknown head type.")

    bos = special_tokens_dict["bos_token"]
    eos = special_tokens_dict["eos_token"]

    tokenizer_orig = AutoTokenizer.from_pretrained(model_path)
    tokenizer_orig.add_special_tokens(special_tokens_dict)

    tokenizer = Tokenizer.from_pretrained(model_path)
    tokenizer.post_processor = TemplateProcessing(
        single=bos + " $A " + eos,
        special_tokens=[(eos, tokenizer_orig.eos_token_id), (bos, tokenizer_orig.bos_token_id)],
    )
    tokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens(special_tokens_dict)

    if head_type == "causal":
        model_config = AutoConfig.from_pretrained(
            model_path,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    elif head_type == "regression":
        model_config = AutoConfig.from_pretrained(
            model_path,
            num_labels=1,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        raise ValueError("Unknown head type.")

    model_config.embd_pdrop = embd_pdrop
    model_config.attn_pdrop = attn_pdrop
    model_config.resid_pdrop = resid_pdrop

    if head_type == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, config=model_config, device_map="auto"
        )
    elif head_type == "regression":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, config=model_config, device_map="auto"
        )
    else:
        raise ValueError("Unknown head type.")

    if tokenizer and model:
        model.resize_token_embeddings(len(tokenizer))

        if save_config:
            tokenizer.save_pretrained(
                os.path.join(output_path, f"{model_name}/Orig/{head_type}")
            )
            model.save_pretrained(
                os.path.join(output_path, f"{model_name}/Orig/{head_type}")
            )

    return model, tokenizer


def specify_bloom_config(
    model_path,
    model_name,
    special_tokens_dict=None,
    head_type="causal",
    output_path="../baseline_models",
    save_config=False
):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    special_tokens_dict = {
        "cls_token": tokenizer.eos_token,
        "mask_token": tokenizer.eos_token,
        "pad_token": tokenizer.eos_token,
        "sep_token": tokenizer.eos_token
    } if special_tokens_dict is None else special_tokens_dict

    tokenizer.add_special_tokens(special_tokens_dict)

    if head_type == "causal":
        model_config = AutoConfig.from_pretrained(
            model_path,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config, device_map="auto")

    elif head_type == "regression":
        model_config = AutoConfig.from_pretrained(
            model_path,
            num_labels=1,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = AutoModelForSequenceClassification.from_pretrained(model_path, config=model_config, device_map="auto")

    else:
        raise ValueError("Unknown head type.")

    if save_config:
        output_path = os.path.join(output_path, f"{model_name}/{head_type}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        tokenizer.save_pretrained(output_path)
        model.save_pretrained(output_path)

    return model, tokenizer
