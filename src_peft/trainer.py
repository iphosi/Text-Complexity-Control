import torch
import os
from random import choices
from copy import deepcopy

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer
from trl import PPOTrainer
from simctg.lossfunction import SimCTGLoss

from joblib import load
import readability
from nltk.tokenize import sent_tokenize

from tqdm import tqdm


class ContrastiveTrainer(Trainer):
    def __init__(
        self,
        vocab_size,
        pad_token_id,
        margin=0.5,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.margin = margin
        super().__init__(**kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False
    ):
        input_ids = inputs.get("input_ids")
        bsz, seqlen = input_ids.size()

        labels = inputs.get("labels")
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.get("logits")
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])

        if self.label_smoother is not None:
            mle_loss = self.label_smoother(outputs, labels, shift_labels=True)
        else:
            mle_loss = outputs.get("loss")

        # Contrastive Loss
        last_hidden_states = outputs.hidden_states[-1]
        assert last_hidden_states.size() == torch.Size([bsz, seqlen, model.config.hidden_size])

        norm_rep = last_hidden_states / last_hidden_states.norm(dim=2, keepdim=True)
        cosine_scores = torch.matmul(norm_rep, norm_rep.transpose(1, 2))
        assert cosine_scores.size() == torch.Size([bsz, seqlen, seqlen])

        simctgloss = SimCTGLoss(
            margin=self.margin,
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id
        )

        cl_loss = simctgloss.contrastive_loss(cosine_scores, input_ids)

        loss = mle_loss + cl_loss

        return (loss, outputs) if return_outputs else loss


class EasyLangPPOTrainer(PPOTrainer):
    def __init__(
        self,
        save_steps,
        save_path,
        train_epochs,
        device,
        query_length=8,
        max_length=1024,
        task_type="CAUSAL_LM",
        reward_type="cls",
        reward_scaling_factor=1,
        use_cls_logits=False,
        baselines=None,
        dynamic_baseline=False,
        use_bert_reg=False,
        bert_features=None,
        features=None,
        prompt_template=None,
        generation_kwargs=None,
        bert_cls_model_path="krupper/text-complexity-classification",
        bert_reg_model_path="../reward_models/reg/distilbert_ft/model",
        bert_reg_tokenizer_path="../baseline_models/distilbert-german/SEQ_CLS_K1",
        ensemble_reg_model_path="../reward_models/reg/ensemble/ensemble.joblib",
        reg_model_path_list=None,
        language="de",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.device = device

        self.train_epochs = train_epochs
        self.save_steps = save_steps
        self.save_path = save_path

        self.task_type = task_type
        self.query_length = query_length
        self.max_length = max_length
        self.prompt_template = [
            "[Leichte Sprache]: ",
            "[Einfache Sprache]: ",
            "[Alltagssprache]: ",
            "[Fachsprache]: "
        ] if prompt_template is None else prompt_template

        self.id2label = {
            0: "easy_language",
            1: "plain_language",
            2: "everyday_language",
            3: "special_language"
        }
        self.label2id = {
            "easy_language": 0,
            "plain_language": 1,
            "everyday_language": 2,
            "special_language": 3
        }
        self.ctrl2id = {
            "[Leichte Sprache]: ": 0,
            "[Einfache Sprache]: ": 1,
            "[Alltagssprache]: ": 2,
            "[Fachsprache]: ": 3
        }

        if self.task_type == "CAUSAL_LM":
            self.generation_kwargs = {
                "top_p": 0.2,
                "repetition_penalty": 1.6,
                "do_sample": True,
                "max_new_tokens": 20
            } if generation_kwargs is None else generation_kwargs
        elif self.task_type == "SEQ_2_SEQ_LM":
            self.generation_kwargs = {
                "top_p": 0.4,
                "repetition_penalty": 1.6,
                "do_sample": True,
                "min_new_tokens": 5,
                "max_new_tokens": 60
            } if generation_kwargs is None else generation_kwargs
        else:
            raise NotImplementedError

        self.ctrl_tokenizer = deepcopy(self.tokenizer)
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

        self.use_bert_reg = use_bert_reg

        if self.use_bert_reg:
            reg_model_path_list = [
                "../reward_models/reg/ensemble/lr_ridge.joblib",
                "../reward_models/reg/ensemble/gbr.joblib"
            ] if reg_model_path_list is None else reg_model_path_list

            self.bert_reg_model = AutoModelForSequenceClassification.from_pretrained(bert_reg_model_path)
            self.bert_reg_tokenizer = AutoTokenizer.from_pretrained(bert_reg_tokenizer_path)
            self.ensemble_reg_model = load(ensemble_reg_model_path)
            self.reg_models = [load(path) for path in reg_model_path_list]
            self.bert_features = ["Coleman-Liau", "LIX", "SMOGIndex"] if bert_features is None else bert_features

        self.reward_type = reward_type
        self.reward_scaling_factor = reward_scaling_factor
        self.use_cls_logits = use_cls_logits

        if self.reward_type == "cls":
            if self.use_cls_logits:
                baselines = [0, 0, 0, 0] if baselines is None else baselines
            else:
                baselines = [0.5, 0.5, 0.5, 0.5] if baselines is None else baselines
        elif self.reward_type == "reg":
            baselines = [6.5, 8, 8, 9.5] if baselines is None else baselines

        self.baselines = {self.ctrl2id[prompt]: baseline for prompt, baseline in zip(prompt_template, baselines)}
        self.dynamic_baseline = dynamic_baseline
        self.features = ["Kincaid"] if features is None else features

        self.language = language

    def train(self):
        print("=" * 100)
        print(f"Reward type: {self.reward_type}")
        print(f"Baselines: {self.baselines}")
        for epoch in range(self.train_epochs):
            print("=" * 100)
            print(f"Epoch {epoch}")
            for step, batch in enumerate(tqdm(self.dataloader)):
                tensors = batch["input_ids"]
                mask = batch["attention_mask"]
                batch_size = tensors.size(0)

                ctrl_strings = choices(self.prompt_template, k=batch_size)
                ctrl_tokens = [self.ctrl_dict[string] for string in ctrl_strings]
                ctrl_input_ids, ctrl_mask = list(zip(*ctrl_tokens))

                ctrl_input_ids = torch.cat(ctrl_input_ids, dim=0).to(self.device)
                ctrl_mask = torch.cat(ctrl_mask, dim=0).to(self.device)

                tgt_ids = [self.ctrl2id[string] for string in ctrl_strings]

                if self.task_type == "CAUSAL_LM":
                    query_tensors = tensors[:, :self.query_length]
                    query_mask = mask[:, :self.query_length]
                elif self.task_type == "SEQ_2_SEQ_LM":
                    query_tensors = tensors
                    query_mask = mask
                else:
                    raise NotImplementedError

                query_tensors = torch.cat([ctrl_input_ids, query_tensors], dim=1)[:, :self.max_length]
                query_mask = torch.cat([ctrl_mask, query_mask], dim=1)[:, :self.max_length]

                with torch.no_grad():
                    response_tensors = self.accelerator.unwrap_model(self.model).generate(
                        input_ids=query_tensors, attention_mask=query_mask, **self.generation_kwargs
                    )

                texts = self.tokenizer.batch_decode(tensors, skip_special_tokens=True)
                query_texts = self.tokenizer.batch_decode(query_tensors, skip_special_tokens=True)
                response_texts = self.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
                response_texts = [
                    response_texts[i].replace(ctrl_strings[i], "")
                    for i in range(batch_size)
                ]

                if self.reward_type == "cls":
                    simp_scores = self.get_cls_simp_scores(response_texts)

                    if self.task_type == "CAUSAL_LM":
                        if self.use_cls_logits:
                            rewards = [
                                simp_scores["logits"][i][tgt_ids[i]] - self.baselines[tgt_ids[i]]
                                for i in range(batch_size)
                            ]
                        else:
                            rewards = [
                                simp_scores["probs"][i][tgt_ids[i]] - self.baselines[tgt_ids[i]]
                                for i in range(batch_size)
                            ]
                    elif self.task_type == "SEQ_2_SEQ_LM":
                        raise NotImplementedError
                    else:
                        raise NotImplementedError

                elif self.reward_type == "reg":
                    simp_scores = self.get_reg_simp_scores(response_texts)

                    if self.task_type == "CAUSAL_LM":
                        rewards = [
                            (simp_scores["preds"][i] - self.baselines[tgt_ids[i]]) * ((tgt_ids[i] >= 2) - 0.5) * 2
                            if simp_scores["preds"][i] != -1 else -1
                            for i in range(batch_size)
                        ]
                    elif self.task_type == "SEQ_2_SEQ_LM":
                        if self.dynamic_baseline:
                            simp_baselines = self.get_reg_simp_scores(texts)
                            rewards = [
                                (simp_scores["preds"][i] - simp_baselines["preds"][i]) * ((tgt_ids[i] >= 2) - 0.5) * 2
                                if simp_scores["preds"][i] != -1 else -1
                                for i in range(batch_size)
                            ]
                        else:
                            rewards = [
                                (simp_scores["preds"][i] - self.baselines[tgt_ids[i]]) * ((tgt_ids[i] >= 2) - 0.5) * 2
                                if simp_scores["preds"][i] != -1 else -1
                                for i in range(batch_size)
                            ]
                    else:
                        raise NotImplementedError

                else:
                    raise ValueError("Invalid reward type.")

                batch["query"] = query_texts
                batch["response"] = response_texts

                query_tensors = list(query_tensors)
                response_tensors = list(response_tensors)
                rewards = list(torch.tensor(rewards) * self.reward_scaling_factor)

                stats = self.step(query_tensors, response_tensors, rewards)

                for s in self.prompt_template:
                    ctrl_id = self.ctrl2id[s]
                    key = f"env/reward_mean_{self.id2label[ctrl_id]}"
                    ctrl_reward = [rewards[i] for i in range(batch_size) if tgt_ids[i] == ctrl_id]
                    if ctrl_reward:
                        ctrl_reward = torch.tensor(ctrl_reward).mean().item()
                    else:
                        ctrl_reward = 0
                    stats[key] = ctrl_reward

                self.log_stats(stats, batch, rewards)

                if (step + 1) % self.save_steps == 0:
                    print("=" * 100)
                    print(f"Saving checkpoint-{step + 1}-{epoch}")
                    self.model.pretrained_model.save_pretrained(
                        os.path.join(self.save_path, f"checkpoint-{step + 1}-{epoch}")
                    )

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

    def get_reg_simp_scores(self, texts):
        if self.language == "de":
            language = "german"
        else:
            raise NotImplementedError

        pred_list = []
        feature_list = []

        for i in range(len(texts)):
            if not any(char.isalpha() for char in texts[i]):
                pred_list.append(-1)
                feature_list.append([-1])
                continue
            sents = sent_tokenize(texts[i], language=language)

            if self.use_bert_reg:
                features = self.get_readability_grades(sents, self.bert_features)
                feature_list.append(features)

                encodings = self.bert_reg_tokenizer(
                    sents,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    stride=64,
                    add_special_tokens=True,
                    return_tensors="pt"
                )

                self.bert_reg_model.eval()

                preds = []

                with torch.no_grad():
                    bert_pred = self.bert_reg_model(
                        input_ids=encodings["input_ids"],
                        attention_mask=encodings["attention_mask"]
                    ).logits.view(-1).mean().item()
                    preds.append(bert_pred)

                for reg in self.reg_models:
                    reg_pred = reg.predict(torch.tensor(features).view(1, -1)).item()
                    preds.append(reg_pred)

                ensemble_pred = self.ensemble_reg_model.predict(torch.tensor(preds).view(1, -1)).item()

                pred_list.append(ensemble_pred)

            else:
                features = self.get_readability_grades(sents, self.features)
                feature_list.append(features)
                pred_list.append(features[0])

        return {
            "preds": pred_list,
            "features": feature_list
        }

    def get_readability_grades(self, text, features):
        grade_dict = readability.getmeasures(text, lang=self.language)["readability grades"]
        return list(map(grade_dict.get, features))
