import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)
from peft import PeftModel

import os
import pandas as pd

from statistics import mean, stdev
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict

from nltk import sent_tokenize
import readability
from perplexity import get_mod_ppl

from preprocess import (
    get_monolingual_dataset,
    get_parallel_dataset,
    split_dataset,
    adapter_summary
)
from similarity import get_rep_spaces, get_pearson_scores


class Evaluate:
    def __init__(
        self,
        model_name,
        model_dict=None,
        prompt_template=None,
        max_length=1024,
        ppl_stride=512,
        task_type="CAUSAL_LM",
        bert_cls_model_path="krupper/text-complexity-classification",
        language="de"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.strategy = "stylistic_continuation"
        self.model_dict = {
            "gpt2-german-oscar": {
                "Baseline":
                    "../baseline_models/gpt2-german-oscar",
                # "ADP_LoRA_r16_reg_easy":
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_reg_easy",
                # "ADP_LoRA_r16_reg_plain":
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_reg_plain",
                # "ADP_LoRA_r16_reg_everyday":
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_reg_everyday",
                # "ADP_LoRA_r16_reg_special":
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_reg_special",
                "ADP_LoRA_r16_cls_plain_logits":
                    f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_cls_plain_logits",
                # "ADP_LoRA_r16_cls_everyday_logits":
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_cls_everyday_logits",
                # "ADP_LoRA_r16_cls_plain_probs":
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_cls_plain_probs",
                # "ADP_LoRA_r16_cls_everyday_probs":
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_cls_everyday_probs",
                # "ADP_LoRA_r16_reg_cls_plain":
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_reg_cls_plain",
                # "ADP_LoRA_r16_ensemble_reg_plain_everyday": [
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_reg_plain",
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_reg_everyday"
                # ],
                # "ADP_LoRA_r16_ensemble_cls_plain_everyday_probs": [
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_cls_plain_probs",
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_cls_everyday_probs"
                # ],
                # "ADP_LoRA_r16_ensemble_cls_plain_everyday_logits": [
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_cls_plain_logits",
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_cls_everyday_logits"
                # ],
                # "ADP_LoRA_r16_ensemble_reg_cls_plain": [
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_reg_plain",
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_cls_plain_probs"
                # ],
                # "ADP_LoRA_r16_fusion_reg_plain_everyday":
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_r16_fusion_reg_plain_everyday",
                # "ADP_LoRA_r16_diverge":
                #     f"../adapters/gpt2-german-oscar/{self.strategy}/LoRA_Monolingual/model_diverge"，
            },
            "T5-Base_GNAD": {
                "Baseline": "../baseline_models/T5-Base_GNAD"
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

        self.prompt_template = [
            "[Leichte Sprache]: ",
            "[Einfache Sprache]: ",
            "[Alltagssprache]: ",
            "[Fachsprache]: "
        ] if prompt_template is None else prompt_template

        self.ctrl_tokenizer = deepcopy(self.baseline_tokenizer)
        self.ctrl_tokenizer.padding_side = "left"
        self.ctrl_tokens = self.ctrl_tokenizer(
            self.prompt_template,
            padding=True,
            return_tensors="pt"
        )
        self.ctrl_dict = {
            self.prompt_template[i]: (
                self.ctrl_tokens["input_ids"][i].view(1, -1),
                self.ctrl_tokens["attention_mask"][i].view(1, -1)
            )
            for i in range(len(self.prompt_template))
        }

        self.bert_cls_model = AutoModelForSequenceClassification.from_pretrained(bert_cls_model_path)
        self.bert_cls_tokenizer = AutoTokenizer.from_pretrained(bert_cls_model_path)

        self.id2label = {
            0: "easy_language",
            1: "plain_language",
            2: "everyday_language",
            3: "special_language"
        }
        self.ctrl2id = {
            "[Leichte Sprache]: ": 0,
            "[Einfache Sprache]: ": 1,
            "[Alltagssprache]: ": 2,
            "[Fachsprache]: ": 3
        }

        self.language = language
        self.pad_token_id = self.baseline_tokenizer.pad_token_id
        self.batch_size = 8

    def ppl_eval(
        self,
        ctrl_string=None,
        weights=None,
        input_path="../datasets/aligned_German_simplification/evaluation/mdr_aligned_news.csv",
        output_path="../evaluation_peft"
    ):
        print("=" * 100)
        print("Evaluating perplexity:")

        if ctrl_string:
            ctrl_dir = self.id2label[self.ctrl2id[ctrl_string]]
        else:
            ctrl_dir = "no_control"

        output_path = f"{output_path}/{self.model_name}/{ctrl_dir}"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_path = f"{output_path}/perplexity.csv"

        dataset_df = pd.read_csv(input_path)

        normal_texts = dataset_df.dropna(subset=["normal_phrase"])["normal_phrase"].values.tolist()
        simple_texts = dataset_df.dropna(subset=["simple_phrase"])["simple_phrase"].values.tolist()

        if ctrl_string:
            normal_texts = list(map(lambda t: ctrl_string + t, normal_texts))
            simple_texts = list(map(lambda t: ctrl_string + t, simple_texts))

        ppl_dict = defaultdict(list)

        for tuning_method, model_path in self.model_dict[self.model_name].items():
            print(f"{self.model_name} | {tuning_method}")

            model = deepcopy(self.baseline_model)

            if tuning_method.startswith("ADP"):
                if "ensemble" in tuning_method:
                    adapter_names = []
                    ensemble_adapter_name = "ensemble"

                    for i in range(len(model_path)):
                        path = model_path[i]
                        if "cls" in path:
                            name = "cls_" + path.split("_")[-2]
                        elif "reg" in path:
                            name = "reg_" + path.split("_")[-1]
                        else:
                            raise ValueError("Invalid path name.")

                        adapter_names.append(name)
                        ensemble_adapter_name += f"_{name}"

                        if i == 0:
                            model = PeftModel.from_pretrained(model, path, adapter_name=name, is_trainable=True)
                        else:
                            model.load_adapter(path, adapter_name=name, is_trainable=True)

                    weights = [1] * (len(adapter_names)) if weights is None else weights
                    assert len(weights) == len(adapter_names)

                    model.add_weighted_adapter(
                        adapters=adapter_names,
                        weights=weights,
                        adapter_name=ensemble_adapter_name
                    )

                    model.set_adapter(ensemble_adapter_name)
                elif "fusion" in tuning_method:
                    model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
                else:
                    model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
                param = adapter_summary(model, as_dict=True)["%param"]
                param = round(param, 3)

            elif tuning_method == "Baseline":
                param = "-"

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

            model.cpu()

            ppl_dict["Tuning Method"].append(tuning_method)
            ppl_dict["%Param"].append(param)
            ppl_dict["PPL Simple"].append(round(simple_ppl, 3))
            ppl_dict["PPL Normal"].append(round(normal_ppl, 3))

        ppl_df = pd.DataFrame(ppl_dict)
        ppl_df.to_csv(output_path, index=False)

    def generate_text(
        self,
        ctrl_string=None,
        query_length=8,
        target_label="Kincaid",
        weights=None,
        input_path="../datasets/monolingual_Leichte_Sprache",
        output_path="../evaluation_peft"
    ):
        print("=" * 100)
        print("Generating texts:")

        if ctrl_string:
            ctrl_dir = self.id2label[self.ctrl2id[ctrl_string]]
        else:
            ctrl_dir = "no_control"

        output_path = f"{output_path}/{self.model_name}/{ctrl_dir}"

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
        elif self.task_type == "SEQ_2_SEQ_LM":
            dataset, _ = get_parallel_dataset(
                tokenizer=self.baseline_tokenizer,
                max_length=self.max_length,
                input_path=input_path
            )
        else:
            raise NotImplementedError

        _, test_set = split_dataset(dataset=dataset, val_split=0.0, test_split=0.01)

        test_dataloader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            collate_fn=self.gen_collator,
            shuffle=False
        )

        assert ctrl_string in self.prompt_template or ctrl_string is None

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

            model = deepcopy(self.baseline_model)

            if tuning_method.startswith("ADP"):
                if "ensemble" in tuning_method:
                    adapter_names = []
                    ensemble_adapter_name = "ensemble"

                    for i in range(len(model_path)):
                        path = model_path[i]
                        if "cls" in path:
                            name = "cls_" + path.split("_")[-2]
                        elif "reg" in path:
                            name = "reg_" + path.split("_")[-1]
                        else:
                            raise ValueError("Invalid path name.")
                        adapter_names.append(name)
                        ensemble_adapter_name += f"_{name}"

                        if i == 0:
                            model = PeftModel.from_pretrained(model, path, adapter_name=name)
                        else:
                            model.load_adapter(path, adapter_name=name, is_trainable=True)

                    weights = [1] * (len(adapter_names)) if weights is None else weights
                    assert len(weights) == len(adapter_names)

                    model.add_weighted_adapter(
                        adapters=adapter_names,
                        weights=weights,
                        adapter_name=ensemble_adapter_name
                    )

                    model.set_adapter(ensemble_adapter_name)
                elif "fusion" in tuning_method:
                    model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
                else:
                    model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
            elif tuning_method == "Baseline":
                pass
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
                batch_size = tensors.size(0)

                if self.task_type == "CAUSAL_LM":
                    query_tensors = tensors[:, :query_length].to(self.device)
                    query_mask = mask[:, :query_length].to(self.device)
                    max_new_tokens = 20
                elif self.task_type == "SEQ_2_SEQ_LM":
                    query_tensors = tensors.to(self.device)
                    query_mask = mask.to(self.device)
                    max_new_tokens = 60
                else:
                    raise NotImplementedError

                if ctrl_string:
                    ctrl_tokens = [self.ctrl_dict[ctrl_string]] * batch_size
                    ctrl_input_ids, ctrl_mask = list(zip(*ctrl_tokens))

                    ctrl_input_ids = torch.cat(ctrl_input_ids, dim=0).to(self.device)
                    ctrl_mask = torch.cat(ctrl_mask, dim=0).to(self.device)

                    query_tensors = torch.cat([ctrl_input_ids, query_tensors], dim=1)
                    query_mask = torch.cat([ctrl_mask, query_mask], dim=1)

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

                if ctrl_string:
                    response_texts = [
                        response_texts[i].replace(ctrl_string, "")
                        for i in range(batch_size)
                    ]

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

    def rsa(
        self,
        ctrl_string=None,
        weights=None,
        num_sample_texts=40,
        num_sample_tokens=800,
        input_path="../datasets/aligned_German_simplification/evaluation/mdr_aligned_news.csv",
        output_path="../evaluation_peft"
    ):
        print("=" * 100)
        print("Analyzing representational similarity:")

        if ctrl_string:
            ctrl_dir = self.id2label[self.ctrl2id[ctrl_string]]
        else:
            ctrl_dir = "no_control"

        output_path = f"{output_path}/{self.model_name}/{ctrl_dir}"

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
            random_state=40
        ).values.tolist()

        src_model = self.baseline_model
        src_model.to(self.device)
        src_model.eval()

        src_rep_spaces = get_rep_spaces(
            model=src_model,
            tokenizer=self.baseline_tokenizer,
            device=self.device,
            texts=sample_texts,
            ctrl_string=ctrl_string,
            num_sample_tokens=num_sample_tokens
        )

        src_model.cpu()

        score_dict = defaultdict(list)

        for tuning_method, model_path in self.model_dict[self.model_name].items():
            print(f"{self.model_name} | {tuning_method}")

            tgt_model = deepcopy(self.baseline_model)

            if tuning_method.startswith("ADP"):
                if "ensemble" in tuning_method:
                    adapter_names = []
                    ensemble_adapter_name = "ensemble"

                    for i in range(len(model_path)):
                        path = model_path[i]
                        if "cls" in path:
                            name = "cls_" + path.split("_")[-2]
                        elif "reg" in path:
                            name = "reg_" + path.split("_")[-1]
                        else:
                            raise ValueError("Invalid path name.")
                        adapter_names.append(name)
                        ensemble_adapter_name += f"_{name}"

                        if i == 0:
                            tgt_model = PeftModel.from_pretrained(tgt_model, path, adapter_name=name)
                        else:
                            tgt_model.load_adapter(path, adapter_name=name, is_trainable=True)

                    weights = [1] * (len(adapter_names)) if weights is None else weights
                    assert len(weights) == len(adapter_names)

                    tgt_model.add_weighted_adapter(
                        adapters=adapter_names,
                        weights=weights,
                        adapter_name=ensemble_adapter_name
                    )

                    tgt_model.set_adapter(ensemble_adapter_name)
                elif "fusion" in tuning_method:
                    tgt_model = PeftModel.from_pretrained(tgt_model, model_path, is_trainable=True)
                else:
                    tgt_model = PeftModel.from_pretrained(tgt_model, model_path, is_trainable=True)

            elif tuning_method == "Baseline":
                print("Skip baseline model.")
                continue
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
                ctrl_string=ctrl_string,
                num_sample_tokens=num_sample_tokens
            )

            tgt_model.cpu()

            with torch.no_grad():
                scores = get_pearson_scores(src_rep_spaces, tgt_rep_spaces, self.device)

            score_dict["Source | Target"].append(f"Baseline | {tuning_method}")
            score_dict["Layer Similarity"].append(scores)

        score_df = pd.DataFrame(score_dict)
        score_df.to_json(output_path)


if __name__ == "__main__":
    model_to_eval = "gpt2-german-oscar"

    prompt_choices = [
        "[Leichte Sprache]: ",
        "[Einfache Sprache]: ",
        "[Alltagssprache]: ",
        "[Fachsprache]: ",
        None
    ]

    evaluate = Evaluate(model_name=model_to_eval, task_type="CAUSAL_LM")

    ensemble_weights = []

    for i in [1, -1]:
        ctrl_choice = prompt_choices[i]
        # evaluate.generate_text(ctrl_string=ctrl_choice, weights=ensemble_weights)
        evaluate.ppl_eval(ctrl_string=ctrl_choice, weights=ensemble_weights)
        evaluate.rsa(ctrl_string=ctrl_choice, weights=ensemble_weights)

