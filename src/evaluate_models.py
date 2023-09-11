import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

import os
import glob
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

from preprocess import split_dataset, get_monolingual_dataset, get_text_complexity_dataset
from perplexity import get_mod_ppl
from similarity import get_rep_spaces, get_pearson_scores

import readability
from nltk import sent_tokenize
from statistics import mean, stdev
from math import sqrt


class Evaluate:
    def __init__(
        self,
        model_name,
        model_dict=None,
        max_length=1024,
        ppl_stride=512,
        task_type="CAUSAL_LM",
        bert_cls_model_path="krupper/text-complexity-classification",
        language="de"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.strategy = "stylistic_adaption"
        self.model_dict = {
            "gpt2-german-oscar": {
                "Baseline":
                    "../baseline_models/gpt2-german-oscar",
                "FT":
                    f"../adapters/gpt2-german-oscar/{self.strategy}/FT_Monolingual/model",
                "ADP_BN_S_r16":
                    f"../adapters/gpt2-german-oscar/{self.strategy}/ADP_BN_S_Monolingual/model_r16",
                "TCP_ADP_BN_S_r16_Kincaid":
                    f"../adapters/gpt2-german-oscar/{self.strategy}/ADP_BN_S_Kincaid/model_r16"
            },
            "distilbert-german": {
                "Baseline":
                    "../baseline_models/distilbert-german",
                "TCP_FT_MOS":
                    f"../adapters/distilbert-german/{self.strategy}/FT_MOS/model",
                "TCP_ADP_BN_S_r16_MOS":
                    f"../adapters/distilbert-german/{self.strategy}/ADP_BN_S_MOS/model_r16",
                "TCP_ADP_BN_S_r16_fusion_MOS": [
                    f"../adapters/distilbert-german/{self.strategy}/ADP_BN_S_MOS/model_r16",
                    f"../adapters/distilbert-german/{self.strategy}/ADP_BN_S_Kincaid/model_r16",
                    f"../adapters/distilbert-german/{self.strategy}/ADP_BN_S_Fusion_MOS/model"
                ],
                "TCP_ADP_BN_S_r16_Kincaid":
                    f"../adapters/distilbert-german/{self.strategy}/ADP_BN_S_Kincaid/model_r16",
            }
        } if model_dict is None else model_dict

        self.model_name = model_name
        assert self.model_name in self.model_dict.keys()

        self.task_type = task_type

        self.baseline_tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.model_dict[self.model_name]["Baseline"], self.task_type)
        )
        if self.task_type == "CAUSAL_LM":
            self.baseline_model = AutoModelForCausalLM.from_pretrained(
                os.path.join(self.model_dict[self.model_name]["Baseline"], self.task_type)
            )
        elif self.task_type == "SEQ_CLS":
            self.baseline_model = AutoModelForSequenceClassification.from_pretrained(
                os.path.join(self.model_dict[self.model_name]["Baseline"], self.task_type)
            )
        elif self.task_type == "SEQ_2_SEQ_LM":
            self.baseline_model = AutoModelForSeq2SeqLM.from_pretrained(
                os.path.join(self.model_dict[self.model_name]["Baseline"], self.task_type)
            )
        else:
            raise NotImplementedError
        self.max_length = max_length
        self.ppl_stride = ppl_stride

        self.bert_cls_model = AutoModelForSequenceClassification.from_pretrained(bert_cls_model_path)
        self.bert_cls_tokenizer = AutoTokenizer.from_pretrained(bert_cls_model_path)

        self.id2label = {
            0: "easy_language",
            1: "plain_language",
            2: "everyday_language",
            3: "special_language"
        }

        self.language = language
        self.pad_token_id = self.baseline_tokenizer.pad_token_id
        self.batch_size = 8

    def ppl_eval(
        self,
        leave_out=None,
        input_path="../datasets/aligned_German_simplification/evaluation/mdr_aligned_news.csv",
        custom_dir=None,
        output_path="../evaluation"
    ):
        print("=" * 100)
        print("Evaluating perplexity:")

        if leave_out:
            if custom_dir:
                output_path = f"{output_path}/{self.model_name}/leave_out/{custom_dir}"
            else:
                first = leave_out[0] + 1
                last = leave_out[-1] + 1
                output_path = f"{output_path}/{self.model_name}/leave_out/L{first}-{last}"
        else:
            output_path = f"{output_path}/{self.model_name}/leave_out/Full"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_path = os.path.join(output_path, "perplexity.csv")

        dataset_df = pd.read_csv(input_path)

        normal_texts = dataset_df.dropna(subset=["normal_phrase"])["normal_phrase"].values.tolist()
        simple_texts = dataset_df.dropna(subset=["simple_phrase"])["simple_phrase"].values.tolist()

        ppl_dict = defaultdict(list)

        for tuning_method, model_path in self.model_dict[self.model_name].items():
            print(f"{self.model_name} | {tuning_method}")

            if tuning_method.startswith("ADP"):
                model = deepcopy(self.baseline_model)
                model.load_adapter(model_path, leave_out=leave_out, set_active=True)
                param = sum(
                    summary["%param"] for summary in model.adapter_summary(as_dict=True)
                    if summary["name"] != "Full model"
                )
                param = round(param, 3)
            elif tuning_method == "Baseline":
                model = self.baseline_model
                param = "-"
            elif tuning_method == "FT":
                model = AutoModelForCausalLM.from_pretrained(model_path)
                param = 100
            else:
                print("Skip non-target model.")
                continue

            model.to(self.device)
            model.eval()

            with torch.no_grad():
                simple_ppl = get_mod_ppl(
                    model=model,
                    tokenizer=self.baseline_tokenizer,
                    texts=simple_texts,
                    max_length=self.max_length,
                    stride=self.ppl_stride,
                    device=self.device
                )
                normal_ppl = get_mod_ppl(
                    model=model,
                    tokenizer=self.baseline_tokenizer,
                    texts=normal_texts,
                    max_length=self.max_length,
                    stride=self.ppl_stride,
                    device=self.device
                )

            ppl_dict["Tuning Method"].append(tuning_method)
            ppl_dict["%Param"].append(param)
            ppl_dict["PPL Simple"].append(round(simple_ppl, 3))
            ppl_dict["PPL Normal"].append(round(normal_ppl, 3))

        ppl_df = pd.DataFrame(ppl_dict)
        ppl_df.to_csv(output_path, index=False)

    def rsa(
        self,
        leave_out=None,
        num_sample_texts=40,
        num_sample_tokens=800,
        seed=40,
        input_path="../datasets/aligned_German_simplification/evaluation/mdr_aligned_news.csv",
        custom_dir=None,
        output_path="../evaluation"
    ):
        print("=" * 50)
        print("Analyzing representational similarity:")

        if leave_out:
            if custom_dir:
                output_path = f"{output_path}/{self.model_name}/leave_out/{custom_dir}"
            else:
                first = leave_out[0] + 1
                last = leave_out[-1] + 1
                output_path = f"{output_path}/{self.model_name}/leave_out/L{first}-{last}"
        else:
            output_path = f"{output_path}/{self.model_name}/leave_out/Full"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_path = f"{output_path}/similarity.json"

        dataset_df = pd.read_csv(input_path)
        dataset_df = pd.concat(
            [dataset_df["normal_phrase"], dataset_df["simple_phrase"]],
            ignore_index=True
        ).dropna()

        sample_texts = dataset_df.sample(
            n=num_sample_texts,
            random_state=seed
        ).values.tolist()

        src_model = self.baseline_model
        src_model.to(self.device)
        src_model.eval()

        src_rep_spaces = get_rep_spaces(
            model=src_model,
            tokenizer=self.baseline_tokenizer,
            device=self.device,
            texts=sample_texts,
            num_sample_tokens=num_sample_tokens,
            seed=seed
        )

        src_model.cpu()

        score_dict = defaultdict(list)

        for tuning_method, model_path in self.model_dict[self.model_name].items():
            print(f"{self.model_name} | {tuning_method}")

            if tuning_method.startswith("ADP"):
                tgt_model = deepcopy(self.baseline_model)
                tgt_model.load_adapter(model_path, leave_out=leave_out, set_active=True)
            elif tuning_method == "Baseline":
                print("Skip baseline model.")
                continue
            elif tuning_method == "FT":
                tgt_model = AutoModelForCausalLM.from_pretrained(model_path)
            else:
                print("Skip non-target model.")
                continue

            tgt_model.to(self.device)
            tgt_model.eval()

            tgt_rep_spaces = get_rep_spaces(
                model=tgt_model,
                tokenizer=self.baseline_tokenizer,
                device=self.device,
                texts=sample_texts,
                num_sample_tokens=num_sample_tokens,
                seed=seed
            )

            tgt_model.cpu()

            with torch.no_grad():
                scores = get_pearson_scores(src_rep_spaces, tgt_rep_spaces, self.device)

            score_dict["Source | Target"].append(f"Baseline | {tuning_method}")
            score_dict["Layer Similarity"].append(scores)

        score_df = pd.DataFrame(score_dict)
        score_df.to_json(output_path)

    def generate_text(
        self,
        leave_out=None,
        query_length=8,
        target_label="Kincaid",
        input_path="../datasets/monolingual_Leichte_Sprache",
        custom_dir=None,
        output_path="../evaluation"
    ):
        print("-" * 50)
        print("Generating texts:")

        if leave_out:
            if custom_dir:
                output_path = f"{output_path}/{self.model_name}/leave_out/{custom_dir}"
            else:
                first = leave_out[0] + 1
                last = leave_out[-1] + 1
                output_path = f"{output_path}/{self.model_name}/leave_out/L{first}-{last}"
        else:
            output_path = f"{output_path}/{self.model_name}/leave_out/Full"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        text_output_path = f"{output_path}/generated_texts.json"
        statistic_output_path = f"{output_path}/statistic.json"

        if self.task_type == "CAUSAL_LM":
            dataset, _ = get_monolingual_dataset(
                tokenizer=self.baseline_tokenizer,
                max_length=self.max_length,
                query_length=query_length,
                input_path=input_path
            )
        else:
            raise NotImplementedError

        _, _, test_set = split_dataset(dataset=dataset, val_split=0.01, test_split=0.01)

        test_dataloader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            collate_fn=self.gen_collator,
            shuffle=False
        )

        text_dict = defaultdict(dict)
        text_df_columns = [
            "Tuning Method",
            "Generated Text"
        ]
        statistic_dict = defaultdict(dict)
        statistic_df_columns = [
            "Tuning Method",
            "Statistic"
        ]

        for tuning_method, model_path in self.model_dict[self.model_name].items():
            print(f"{self.model_name} | {tuning_method}")

            if tuning_method.startswith("ADP"):
                model = deepcopy(self.baseline_model)
                model.load_adapter(model_path, leave_out=leave_out, set_active=True)
            elif tuning_method == "Baseline":
                model = self.baseline_model
            elif tuning_method == "FT":
                model = AutoModelForCausalLM.from_pretrained(model_path)
            else:
                print("Skip non-target model.")
                continue

            model.to(self.device)
            model.eval()

            response_text_list = []
            cls_pred_list = []

            for step, batch in enumerate(tqdm(test_dataloader)):
                tensors = batch["input_ids"]
                mask = batch["attention_mask"]

                if self.task_type == "CAUSAL_LM":
                    query_tensors = tensors[:, :query_length].to(self.device)
                    query_mask = mask[:, :query_length].to(self.device)
                    max_new_tokens = 20
                else:
                    raise NotImplementedError

                with torch.no_grad():
                    response_tensors = model.generate(
                        input_ids=query_tensors,
                        attention_mask=query_mask,
                        top_p=0.2,
                        repetition_penalty=1.6,
                        do_sample=True,
                        max_new_tokens=max_new_tokens
                    )

                response_texts = self.baseline_tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                response_text_list.extend(response_texts)

                cls_preds = self.get_cls_simp_scores(response_texts)["preds"]
                cls_pred_list.extend(cls_preds)

            cls_text_dict = defaultdict(list)
            for i in range(len(response_text_list)):
                cls_id = cls_pred_list[i]
                text = response_text_list[i]
                cls_text_dict[self.id2label[cls_id]].append(text)

            cls_freq_dict = defaultdict(int)
            cls_simp_dict = defaultdict(dict)
            for cls, text_list in cls_text_dict.items():
                cls_freq_dict[cls] = len(text_list)
                simp_scores = self.get_reg_simp_scores(text_list, target_label=target_label)
                simp_dict = {
                    "Mean": mean(simp_scores) if len(simp_scores) >= 2 else -1,
                    "Std": stdev(simp_scores) if len(simp_scores) >= 2 else -1
                }
                cls_simp_dict[cls][target_label] = simp_dict

            summary_simp_scores = self.get_reg_simp_scores(response_text_list, target_label=target_label)
            summary_simp_dict = {
                "Mean": mean(summary_simp_scores) if len(summary_simp_scores) >= 2 else -1,
                "Std": stdev(summary_simp_scores) if len(summary_simp_scores) >= 2 else -1
            }
            cls_simp_dict["summary"][target_label] = summary_simp_dict

            text_dict[tuning_method] = cls_text_dict
            statistic_dict[tuning_method]["Frequency"] = cls_freq_dict
            statistic_dict[tuning_method]["Simplicity"] = cls_simp_dict

        text_df = pd.DataFrame(data=text_dict.items(), columns=text_df_columns)
        statistic_df = pd.DataFrame(data=statistic_dict.items(), columns=statistic_df_columns)

        text_df.to_json(text_output_path)
        statistic_df.to_json(statistic_output_path)

    def gen_collator(self, batch):
        encoding_list = list(map(lambda d: d["input_ids"], batch))
        encoding_batch = pad_sequence(encoding_list, batch_first=True, padding_value=self.pad_token_id)

        attention_mask_list = list(map(lambda d: d["attention_mask"], batch))
        attention_mask_batch = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        return {
            "input_ids": encoding_batch,
            "attention_mask": attention_mask_batch
        }

    def get_cls_simp_scores(self, texts):
        self.bert_cls_model.eval()

        encodings = self.bert_cls_tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            stride=64,
            add_special_tokens=True,
            return_tensors="pt"
        )

        softmax = torch.nn.Softmax(dim=1)

        with torch.no_grad():
            bert_output = self.bert_cls_model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"]
            )

        logits = bert_output.logits
        probs = softmax(logits)
        preds = probs.argmax(dim=1)

        return {
            "logits": logits.tolist(),
            "probs": probs.tolist(),
            "preds": preds.tolist()
        }

    def get_reg_simp_scores(self, texts, target_label="Kincaid"):
        if self.language == "de":
            language = "german"
        else:
            raise NotImplementedError

        grade_list = []
        for i in range(len(texts)):
            if not any(char.isalpha() for char in texts[i]):
                grade_list.append(-1)
                continue
            sents = sent_tokenize(texts[i], language=language)
            grade = readability.getmeasures(sents, lang=self.language)["readability grades"][target_label]
            grade_list.append(grade)
        return grade_list

    def tcp(
        self,
        text_column_name="phrase",
        target_label="Kincaid",
        do_rescaling=False,
        max_value=None,
        min_value=None,
        rescaling_factor=None,
        input_path="../datasets/TextComplexity/monolingual_dynamic_split",
        output_path="../evaluation"
    ):
        print("-" * 50)
        print("Predicting Text Complexity:")

        output_path = f"{output_path}/{self.model_name}"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_path = f"{output_path}/{target_label.lower()}_prediction.csv"

        dataset, _ = get_text_complexity_dataset(
            tokenizer=self.baseline_tokenizer,
            max_length=self.max_length,
            text_column_name=text_column_name,
            target_label=target_label,
            do_rescaling=do_rescaling,
            max_value=max_value,
            min_value=min_value,
            rescaling_factor=rescaling_factor,
            input_path=input_path
        )

        _, _, test_set = split_dataset(dataset=dataset, val_split=0.01, test_split=0.01)
        test_dataloader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            collate_fn=self.tcp_collator,
            shuffle=False
        )

        pred_dict = defaultdict(list)

        for tuning_method, model_path in self.model_dict[self.model_name].items():
            print(f"{self.model_name} | {tuning_method}")

            if tuning_method.startswith("TCP_ADP") and tuning_method.endswith(target_label):
                model = deepcopy(self.baseline_model)
                if "fusion" in tuning_method:
                    for i in range(len(model_path) - 1):
                        model.load_adapter(model_path[i], set_active=True)
                    model.load_adapter_fusion(model_path[-1], set_active=True)
                else:
                    model.load_adapter(model_path, set_active=True)
            elif tuning_method == f"TCP_FT_{target_label}":
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
            else:
                print("Skip non-target model.")
                continue

            model.to(self.device)
            model.eval()

            loss_list = []
            with torch.no_grad():
                for batch_data in tqdm(test_dataloader):
                    labels = batch_data["labels"].to(self.device)
                    output = model(
                        input_ids=batch_data["input_ids"].to(self.device),
                        attention_mask=batch_data["attention_mask"].to(self.device)
                    )
                    loss = (labels - output.logits.view(-1)) ** 2
                    loss_list.extend(loss.tolist())

            rmse = round(sqrt(mean(loss_list)), 3)

            pred_dict["Tuning Method"].append(tuning_method)
            pred_dict["RMSE"].append(rmse)

        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv(output_path, index=False)

    def tcp_collator(self, batch):
        encoding_list = list(map(lambda d: d["input_ids"], batch))
        encoding_batch = pad_sequence(encoding_list, batch_first=True, padding_value=self.pad_token_id)

        attention_mask_list = list(map(lambda d: d["attention_mask"], batch))
        attention_mask_batch = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        if "labels" in batch[0].keys():
            label_list = list(map(lambda d: d["labels"], batch))
            label_batch = torch.tensor(label_list, dtype=torch.float)
        else:
            label_batch = encoding_batch

        return {
            "input_ids": encoding_batch,
            "attention_mask": attention_mask_batch,
            "labels": label_batch
        }


if __name__ == "__main__":
    model_to_eval = "gpt2-german-oscar"
    # model_to_eval = "distilbert-german"

    evaluate = Evaluate(model_name=model_to_eval, task_type="CAUSAL_LM")
    evaluate.device = "cpu"

    # for leave_out_range in range(13):
    #     leave_out_layers = [layer for layer in range(leave_out_range)]
    #
    #     evaluate.ppl_eval(leave_out=leave_out_layers)
    #     evaluate.generate_text(leave_out=leave_out_layers)

    leave_out_layers = [4, 5, 6]
    # evaluate.ppl_eval(leave_out=leave_out_layers)
    # evaluate.generate_text(leave_out=leave_out_layers)

    evaluate.rsa(leave_out=leave_out_layers)

    # evaluate.tcp(
    #     target_label="Kincaid",
    #     do_rescaling=True,
    #     max_value=50,
    #     min_value=0,
    #     rescaling_factor=10,
    #     input_path="../datasets/TextComplexity/monolingual_dynamic_split"
    # )
