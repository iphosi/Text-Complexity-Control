import pandas as pd
import numpy as np
import os
import glob

from tqdm import tqdm

import math
from simplicity import get_readability_grades
from nltk.tokenize import sent_tokenize


def dynamic_split(topic_group, num_phrases_per_sample=16):
    num_sections = max(1, math.ceil(topic_group.size / num_phrases_per_sample))
    return np.array_split(topic_group, indices_or_sections=num_sections)


def join_texts(sections):
    return list(map(lambda s: "\n".join(s.values), sections))


def add_labels(
    input_path,
    output_path=None,
    text_column_name="phrase",
    num_phrases_per_sample=16,
    do_split=True,
    add_readability_label=True,
    add_lexical_label=False,
):
    if text_column_name == "Sentence":
        output_path = f"../datasets/TextComplexity/human_feedback" if output_path is None else output_path

        text_dataframe = pd.read_csv(input_path)[["Sentence", "MOS"]].dropna()
        text_dataframe.rename(columns={"Sentence": "phrase"}, inplace=True)

    elif text_column_name == "phrase":
        output_path = f"../datasets/TextComplexity/monolingual_dynamic_split" if output_path is None else output_path

        text_dataframe = pd.read_csv(input_path)[["topic", "phrase_number", "phrase"]].dropna()
        print(f"Dataset size before splitting: {len(text_dataframe)}")

        if do_split:
            text_dataframe = text_dataframe.sort_values(["phrase_number"]).groupby(["topic"])["phrase"]
            text_dataframe = text_dataframe.apply(dynamic_split, num_phrases_per_sample=num_phrases_per_sample)
            text_dataframe = text_dataframe.apply(join_texts).reset_index().explode("phrase")
            print(f"Dataset size after splitting: {len(text_dataframe)}")

    else:
        raise ValueError("Invalid text column name.")

    text_column_name = "phrase"
    print("-" * 100)
    print(f"Dataset size before labeling: {len(text_dataframe)}")

    text_dataframe = text_dataframe.loc[text_dataframe[text_column_name].str.contains("[a-zA-Z]")]
    text_dataframe.drop_duplicates(inplace=True)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = os.path.join(output_path, os.path.basename(input_path))

    tqdm.pandas()

    if add_readability_label:
        text_list_series = text_dataframe[text_column_name].apply(sent_tokenize)
        readability_grades = text_list_series.progress_apply(get_readability_grades).apply(pd.Series)
        text_dataframe = pd.concat([text_dataframe, readability_grades], axis=1)

    if add_lexical_label:
        raise NotImplementedError

    print(f"Dataset size after labeling: {len(text_dataframe)}")
    text_dataframe.to_csv(output_path, index=False)


if __name__ == "__main__":
    data_path = "../datasets/monolingual_Leichte_Sprache"
    dataset_path_list = glob.glob(f"{data_path}/*.csv")

    human_feedback_dataset = "../datasets/TextComplexity/human_feedback_orig/text_complexity.csv"

    add_labels(
        input_path=human_feedback_dataset,
        text_column_name="Sentence",
        do_split=True,
        add_readability_label=True,
        add_lexical_label=False,
    )

    for dataset in dataset_path_list:
        print("=" * 100)
        print(os.path.basename(dataset))
        add_labels(
            input_path=dataset,
            text_column_name="phrase",
            do_split=True,
            add_readability_label=True,
            add_lexical_label=False,
        )

    print("End")
