from sklearn import linear_model, svm, ensemble
import pandas as pd
import torch
from torch.utils.data import Subset

import math
from statistics import mean

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from preprocess import get_text_complexity_dataset, split_dataset

from joblib import dump

from collections import defaultdict
from itertools import product, compress

import os
import shutil
from tqdm import tqdm


def train_reg(
    train_dataset,
    feature_mask,
    reg_model=None,
    save_model=False,
    output_path="../reward_models/reg/ensemble/lr.joblib"
):
    src_tgt = [
        (list(compress(data["features"].tolist(), feature_mask)), data["labels"].item())
        for data in iter(train_dataset)
    ]
    batch_features, batch_labels = list(zip(*src_tgt))

    reg_model = linear_model.Ridge(alpha=0.5) if reg_model is None else reg_model
    reg_model.fit(batch_features, batch_labels)

    if save_model:
        if not os.path.exists(os.path.split(output_path)[0]):
            os.makedirs(os.path.split(output_path)[0])
        dump(reg_model, filename=output_path)

    return reg_model


def train_ensemble_reg(
    train_dataset,
    feature_mask,
    bert_model_dict,
    reg_model_dict,
    ensemble_reg_model=None,
    save_model=False,
    output_path="../reward_models/reg/ensemble"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_preds = []
    batch_labels = []

    for data in tqdm(train_dataset):
        input_ids = data["input_ids"]
        features = data["features"]
        features = torch.tensor(list(compress(features.tolist(), feature_mask)))
        label = data["labels"]

        preds = []

        for _, bert_model in bert_model_dict.items():
            bert_model.to(device)
            bert_model.eval()
            with torch.no_grad():
                bert_pred = bert_model(input_ids=input_ids.view(1, -1).to(device)).logits.item()
                preds.append(bert_pred)

        for _, reg_model in reg_model_dict.items():
            reg_pred = reg_model.predict(features.view(1, -1)).item()
            preds.append(reg_pred)

        batch_preds.append(preds)
        batch_labels.append(label)

    ensemble_reg_model = linear_model.Ridge(alpha=0.5) if ensemble_reg_model is None else ensemble_reg_model
    ensemble_reg_model.fit(batch_preds, batch_labels)

    if save_model:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        dump(ensemble_reg_model, filename=os.path.join(output_path, "ensemble.joblib"))

    return ensemble_reg_model


def train_concat_reg(
    train_dataset,
    feature_mask,
    bert_model_dict,
    scaling_factor=1,
    concat_reg_model=None,
    save_model=False,
    output_path="../reward_models/reg/concat"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer_idx = -1
    batch_idx = 0
    token_idx = 0

    batch_cls_token_hidden_states = []
    batch_labels = []

    for data in tqdm(train_dataset):
        input_ids = data["input_ids"]
        features = data["features"]
        features = torch.tensor(list(compress(features.tolist(), feature_mask)))
        label = data["labels"]

        accumulated_hidden_state = []

        for name, bert_model in bert_model_dict.items():
            bert_model.to(device)
            bert_model.eval()
            with torch.no_grad():
                output = bert_model(
                    input_ids=input_ids.view(1, -1).to(device),
                    output_hidden_states=True
                )

            cls_token_hidden_state = output.hidden_states[layer_idx][batch_idx][token_idx]
            accumulated_hidden_state.append(cls_token_hidden_state.view(-1, 1))

        features = scaling_factor * features.to(device)
        accumulated_hidden_state = torch.cat(accumulated_hidden_state, dim=1).sum(dim=1)
        accumulated_hidden_state = torch.cat([accumulated_hidden_state, features], dim=0)

        batch_cls_token_hidden_states.append(accumulated_hidden_state.tolist())
        batch_labels.append(label.item())

    concat_reg_model = linear_model.Ridge(alpha=0.5) if concat_reg_model is None else concat_reg_model
    concat_reg_model.fit(batch_cls_token_hidden_states, batch_labels)

    if save_model:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        dump(concat_reg_model, filename=os.path.join(output_path, "concat.joblib"))

    return concat_reg_model


def eval_ensemble_reg(
    test_dataset,
    feature_mask,
    feature_names,
    bert_model_dict,
    reg_model_dict,
    ensemble_reg_model=None,
    save_result=False,
    baseline_model_name="DistilBERT_FT",
    output_path="../reward_models/reg/ensemble"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_dict = defaultdict(list)

    for test_data in tqdm(test_dataset):
        input_ids = test_data["input_ids"]
        features = test_data["features"]
        features = torch.tensor(list(compress(features.tolist(), feature_mask)))
        label = test_data["labels"]

        preds = []

        for name, bert_model in bert_model_dict.items():
            bert_model.to(device)
            bert_model.eval()
            with torch.no_grad():
                bert_pred = bert_model(input_ids=input_ids.view(1, -1).to(device)).logits
                preds.append(bert_pred.item())

            loss_dict[name].append((bert_pred.item() - label.item()) ** 2)

        for name, reg_model in reg_model_dict.items():
            reg_pred = reg_model.predict(features.view(1, -1))
            preds.append(reg_pred.item())

            loss_dict[name].append((reg_pred.item() - label.item()) ** 2)

        if ensemble_reg_model:
            ensemble_pred = ensemble_reg_model.predict(torch.tensor(preds).view(1, -1))
            loss_dict["Ensemble"].append((ensemble_pred.item() - label.item()) ** 2)

    loss_dict = {k: round(math.sqrt(mean(v)), 3) for k, v in loss_dict.items()}
    rmse_df = pd.DataFrame(loss_dict.items(), columns=["Model", "MOS RMSE"])

    global best_ensemble_reg_rmse

    if loss_dict[baseline_model_name] < best_ensemble_reg_rmse:
        best_ensemble_reg_rmse = loss_dict[baseline_model_name]

    if save_result:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if loss_dict["Ensemble"] < best_ensemble_reg_rmse:
            best_ensemble_reg_rmse = loss_dict["Ensemble"]

            model_name_list = []
            model_name_list.extend(list(bert_model_dict.keys()))
            model_name_list.extend(list(reg_model_dict.keys()))

            with open(os.path.join(output_path, "ensemble.txt"), "w") as f:
                for i in range(len(model_name_list)):
                    f.write(f"{model_name_list[i]}: {ensemble_reg_model.coef_[i]:.3f}\n")
                feature_names = " ".join(feature_names)
                f.write(f"Features: {feature_names}")

            rmse_df.to_csv(os.path.join(output_path, "rmse.csv"), index=False)
        else:
            shutil.rmtree(output_path)

    return loss_dict


def eval_concat_reg(
    test_dataset,
    feature_mask,
    feature_names,
    bert_model_dict,
    scaling_factor=1,
    concat_reg_model=None,
    save_result=False,
    baseline_model_name="DistilBERT_FT",
    output_path="../reward_models/reg/concat"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    layer_idx = -1
    batch_idx = 0
    token_idx = 0

    loss_dict = defaultdict(list)

    for data in tqdm(test_dataset):
        input_ids = data["input_ids"]
        features = data["features"]
        features = torch.tensor(list(compress(features.tolist(), feature_mask)))
        label = data["labels"]

        accumulated_hidden_state = []

        for name, bert_model in bert_model_dict.items():
            bert_model.to(device)
            bert_model.eval()
            with torch.no_grad():
                output = bert_model(
                    input_ids=input_ids.view(1, -1).to(device),
                    output_hidden_states=True
                )

            bert_pred = output.logits

            loss_dict[name].append((bert_pred.item() - label.item()) ** 2)

            cls_token_hidden_state = output.hidden_states[layer_idx][batch_idx][token_idx]
            accumulated_hidden_state.append(cls_token_hidden_state.view(-1, 1))

        features = scaling_factor * features.to(device)
        accumulated_hidden_state = torch.cat(accumulated_hidden_state, dim=1).sum(dim=1)
        accumulated_hidden_state = torch.cat([accumulated_hidden_state, features], dim=0)

        concat_pred = concat_reg_model.predict(accumulated_hidden_state.view(1, -1).cpu())
        loss_dict["Concatenate"].append((concat_pred.item() - label.item()) ** 2)

    loss_dict = {k: round(math.sqrt(mean(v)), 3) for k, v in loss_dict.items()}
    rmse_df = pd.DataFrame(loss_dict.items(), columns=["Model", "MOS RMSE"])

    global best_concat_reg_rmse

    if loss_dict[baseline_model_name] < best_concat_reg_rmse:
        best_concat_reg_rmse = loss_dict[baseline_model_name]

    if save_result:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if loss_dict["Concatenate"] < best_concat_reg_rmse:
            best_concat_reg_rmse = loss_dict["Concatenate"]
            with open(os.path.join(output_path, "concat.txt"), "w") as f:
                for name in bert_model_dict.keys():
                    f.write(f"{name}\n")
                feature_names = " ".join(feature_names)
                f.write(f"Features: {feature_names}")

            rmse_df.to_csv(os.path.join(output_path, "rmse.csv"), index=False)
        else:
            shutil.rmtree(output_path)

    return loss_dict


def reg_loop():
    # Features
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
    ]

    print("=" * 100)
    print(f"Features: {feature_column_names}")

    feature_masks = list(product((True, False), repeat=len(feature_column_names)))[:-1][::-1]

    # Model
    basline_model_path = "../baseline_models/distilbert-german/SEQ_CLS"
    tokenizer = AutoTokenizer.from_pretrained(basline_model_path)

    ft_model_path = "../reward_models/reg/distilbert_ft/model"
    distilbert_ft = AutoModelForSequenceClassification.from_pretrained(ft_model_path)

    # Dataset
    data_path = "../datasets/TextComplexity/human_feedback"

    max_length = 512

    dataset, _ = get_text_complexity_dataset(
        tokenizer=tokenizer,
        max_length=max_length,
        return_features=True,
        feature_column_names=feature_column_names,
        target_label="MOS",
        input_path=data_path
    )
    train_set, val_set, test_set = split_dataset(dataset=dataset, val_split=0.05, test_split=0.05)

    reg_train_set_indices = []
    reg_train_set_indices.extend(train_set.indices)
    reg_train_set_indices.extend(val_set.indices)
    reg_train_set = Subset(dataset=dataset, indices=reg_train_set_indices)

    # Ensemble and Concatenate
    global best_ensemble_reg_rmse, best_concat_reg_rmse

    for j in range(len(feature_masks)):
        print("=" * 100)
        feature_combination = list(compress(feature_column_names, feature_masks[j]))
        print(f"Selected features: {feature_combination}")

        lr_ridge = linear_model.Ridge(alpha=0.2)
        lr_ridge_path = f"../reward_models/ensemble_{j}/lr_ridge.joblib"
        gbr = ensemble.GradientBoostingRegressor()
        gbr_path = f"../reward_models/ensemble_{j}/gbr.joblib"

        lr_ridge = train_reg(
            train_dataset=reg_train_set,
            feature_mask=feature_masks[j],
            reg_model=lr_ridge,
            save_model=True,
            output_path=lr_ridge_path
        )
        gbr = train_reg(
            train_dataset=reg_train_set,
            feature_mask=feature_masks[j],
            reg_model=gbr,
            save_model=True,
            output_path=gbr_path
        )

        bert_models = {
            "DistilBERT_FT": distilbert_ft
        }

        reg_models = {
            "LR_Ridge": lr_ridge,
            "BGR": gbr,
        }

        # Ensemble
        ensemble_model = linear_model.Ridge(alpha=0.2)
        ensemble_model_path = f"../reward_models/ensemble_{j}"
        ensemble_model = train_ensemble_reg(
            train_dataset=reg_train_set,
            feature_mask=feature_masks[j],
            bert_model_dict=bert_models,
            reg_model_dict=reg_models,
            ensemble_reg_model=ensemble_model,
            save_model=True,
            output_path=ensemble_model_path
        )

        eval_ensemble_reg(
            test_dataset=test_set,
            feature_mask=feature_masks[j],
            feature_names=feature_combination,
            bert_model_dict=bert_models,
            reg_model_dict=reg_models,
            ensemble_reg_model=ensemble_model,
            save_result=True,
            output_path=ensemble_model_path
        )

        # Concatenate
        concat_model = svm.SVR()
        concat_model_path = f"../reward_models/concat_{j}"
        concat_model = train_concat_reg(
            train_dataset=reg_train_set,
            feature_mask=feature_masks[j],
            scaling_factor=1,
            bert_model_dict=bert_models,
            concat_reg_model=concat_model,
            save_model=True,
            output_path=concat_model_path
        )

        eval_concat_reg(
            test_dataset=test_set,
            feature_mask=feature_masks[j],
            feature_names=feature_combination,
            scaling_factor=1,
            bert_model_dict=bert_models,
            concat_reg_model=concat_model,
            save_result=True,
            output_path=concat_model_path
        )


if __name__ == "__main__":
    best_ensemble_reg_rmse = 10
    best_concat_reg_rmse = 10

    reg_loop()

    print("End")
