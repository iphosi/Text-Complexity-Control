import pandas as pd
import os
import matplotlib.pyplot as plt
from natsort import natsorted
from statistics import mean

import torch
from torchmetrics import PearsonCorrCoef


def vis_ppl(
    model_name="gpt2-german-oscar",
    pause=3
):
    input_path = f"../evaluation/{model_name}/leave_out"
    output_path = f"../evaluation/{model_name}"

    dirs = natsorted(os.listdir(input_path))

    plt.figure(figsize=(12, 5))

    tuning_method_list = []
    simple_ppl_list = []
    normal_ppl_list = []

    for dir_name in dirs:
        file_path = os.path.join(input_path, dir_name, "perplexity.csv")
        ppl_df = pd.read_csv(file_path)

        tuning_method_list = ppl_df["Tuning Method"].values.tolist()

        simple_ppl_list.append(
            ppl_df["PPL Simple"].values.tolist()
        )
        normal_ppl_list.append(
            ppl_df["PPL Normal"].values.tolist()
        )

    simple_ppl_list = list(zip(*simple_ppl_list))
    normal_ppl_list = list(zip(*normal_ppl_list))

    for i, tuning_method in enumerate(tuning_method_list):
        simple_ppl = simple_ppl_list[i]
        normal_ppl = normal_ppl_list[i]

        plt.subplot(1, 2, 1)
        plt.plot(
            simple_ppl,
            label=tuning_method,
            linestyle="dashed" if tuning_method in ["Baseline", "FT"] else "solid"
        )
        plt.legend(loc="upper left", prop={"size": 8})
        plt.xlabel("Leave Out Range")
        plt.ylabel("Perplexity")
        plt.title("Simple Text Perplexity")

        plt.subplot(1, 2, 2)
        plt.plot(
            normal_ppl,
            label=tuning_method,
            linestyle="dashed" if tuning_method in ["Baseline", "FT"] else "solid"
        )
        plt.legend(loc="upper right", prop={"size": 8})
        plt.xlabel("Leave Out Range")
        plt.ylabel("Perplexity")
        plt.title("Normal Text Perplexity")

    plt.suptitle(model_name)
    plt.savefig(os.path.join(output_path, "perplexity.png"))
    plt.show(block=False)
    plt.pause(pause)
    plt.close()


def vis_simp(
    model_name="gpt2-german-oscar",
    target_label="Kincaid",
    num_baselines=2,
    pause=3
):
    input_path = f"../evaluation/{model_name}/leave_out"
    output_path = f"../evaluation/{model_name}"
    dirs = natsorted(os.listdir(input_path))

    plt.figure(figsize=(6, 5))

    tuning_method_list = []
    score_list = []

    for dir_name in dirs:
        file_path = os.path.join(input_path, dir_name, "statistic.json")
        simp_df = pd.read_json(file_path)

        tuning_method_list = simp_df["Tuning Method"].values.tolist()

        scores = []

        for statistic_dict in simp_df["Statistic"].values.tolist():
            scores.append(statistic_dict["Simplicity"]["summary"][target_label]["Mean"])

        score_list.append(scores)

    score_list = list(zip(*score_list))

    for i in range(num_baselines):
        score_list[i] = (mean(score_list[i]),) * len(score_list[i])

    for i, tuning_method in enumerate(tuning_method_list):
        scores = score_list[i]

        plt.plot(
            scores,
            label=tuning_method,
            linestyle="dashed" if tuning_method in ["Baseline", "FT"] else "solid"
        )
        plt.legend(loc="upper left", prop={"size": 8})
        plt.xlabel("Leave Out Range")
        plt.ylabel(f"{target_label}")
        plt.title("Readability")

    plt.suptitle(model_name)
    plt.savefig(os.path.join(output_path, "simplicity.png"))
    plt.show(block=False)
    plt.pause(pause)
    plt.close()


def vis_freq(
    model_name="gpt2-german-oscar",
    target_style="easy",
    num_baselines=2,
    pause=3
):
    style2key = {
        "easy": "easy_language",
        "plain": "plain_language",
        "everyday": "everyday_language",
        "special": "special_language"
    }
    style2title = {
        "easy": "Easy Language",
        "plain": "Plain Language",
        "everyday": "Everyday Language",
        "special": "Special Language"
    }
    input_path = f"../evaluation/{model_name}/leave_out"
    output_path = f"../evaluation/{model_name}"
    dirs = natsorted(os.listdir(input_path))

    plt.figure(figsize=(6, 5))

    tuning_method_list = []
    freq_list = []

    for dir_name in dirs:
        file_path = os.path.join(input_path, dir_name, "statistic.json")
        simp_df = pd.read_json(file_path)

        tuning_method_list = simp_df["Tuning Method"].values.tolist()

        freqs = []

        for statistic_dict in simp_df["Statistic"].values.tolist():
            if style2key[target_style] not in statistic_dict["Frequency"].keys():
                freqs.append(0)
            else:
                freqs.append(statistic_dict["Frequency"][style2key[target_style]])

        freq_list.append(freqs)

    freq_list = list(zip(*freq_list))

    for i in range(num_baselines):
        freq_list[i] = (mean(freq_list[i]),) * len(freq_list[i])

    for i, tuning_method in enumerate(tuning_method_list):
        freqs = freq_list[i]

        plt.plot(
            freqs,
            label=tuning_method,
            linestyle="dashed" if tuning_method in ["Baseline", "FT"] else "solid"
        )
        plt.legend(loc="upper left", prop={"size": 8})
        plt.xlabel("Leave Out Range")
        plt.ylabel("Frequency")
        plt.title(style2title[target_style])

    plt.suptitle(model_name)
    plt.savefig(os.path.join(output_path, f"frequency_{target_style}.png"))
    plt.show(block=False)
    plt.pause(pause)
    plt.close()


def vis_sim(
    model_name="gpt2-german-oscar",
    drop_last=1,
    pause=3,
    figsize=(12, 5),
    comment_list=None,
    input_path_list=None
):
    input_path_list = [f"../evaluation/{model_name}/similarity.json"] if input_path_list is None else input_path_list
    output_path = f"../evaluation/{model_name}"

    sim_df = pd.concat([pd.read_json(input_path) for input_path in input_path_list], ignore_index=True)

    plt.figure(figsize=figsize)

    for i, row in sim_df.iterrows():
        if comment_list:
            comment = comment_list[i]
        else:
            comment = ""
        src_tgt = row["Source | Target"] + comment
        sim = row["Layer Similarity"]

        plt.subplot(1, 2, 1)
        plt.plot(sim, label=src_tgt)
        plt.xlabel("Layer")
        plt.ylabel("PCC")
        plt.legend(loc="lower left", prop={"size": 8})
        plt.title("Representational Similarity")

        plt.subplot(1, 2, 2)
        plt.plot(sim[:-drop_last], label=src_tgt)
        plt.xlabel("Layer")
        plt.ylabel("PCC")
        plt.legend(loc="lower left", prop={"size": 8})
        plt.title(f"Representational Similarity Drop Last (N = {drop_last})")

    plt.suptitle(model_name)
    plt.savefig(os.path.join(output_path, "similarity.png"))
    plt.show(block=False)
    plt.pause(pause)
    plt.close()


def vis_corr(
    src_label="MOS",
    tgt_label="Kincaid",
    src=None,
    tgt=None,
    device=None,
    pause=3,
    input_path="../datasets/TextComplexity/human_feedback/text_complexity.csv",
    output_path="../evaluation/correlation"
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(input_path).dropna()
    src = torch.tensor(df[src_label].values) if src is None else torch.tensor(src)
    tgt = torch.tensor(df[tgt_label].values) if tgt is None else torch.tensor(tgt)

    pearson = PearsonCorrCoef().to(device)
    score = pearson(src.to(device), tgt.to(device)).item()

    plt.figure(figsize=(6, 6))
    plt.scatter(src, tgt, label=f"PCC: {round(score, 3)}")
    plt.xlabel(src_label)
    plt.ylabel(tgt_label)
    plt.legend(loc="lower right", prop={"size": 8})
    plt.title(f"Correlation {src_label} | {tgt_label}")

    plt.savefig(os.path.join(output_path, f"pearson_{src_label.lower()}_{tgt_label.lower()}.png"))
    plt.show(block=False)
    plt.pause(pause)
    plt.close()


if __name__ == "__main__":
    vis_sim(
        comment_list=[
            " (CTRL: None)",
            " (CTRL: Plain)",
            " (CTRL: None)",
            " (CTRL: Plain)",
            " (CTRL: None)",
            " (CTRL: Plain)"
        ],
        input_path_list=[
            "../evaluation_peft/gpt2-german-oscar/reg_plain/no_control/similarity.json",
            "../evaluation_peft/gpt2-german-oscar/reg_plain/plain_language/similarity.json",
            "../evaluation_peft/gpt2-german-oscar/cls_plain_probs/no_control/similarity.json",
            "../evaluation_peft/gpt2-german-oscar/cls_plain_probs/plain_language/similarity.json",
            "../evaluation_peft/gpt2-german-oscar/cls_plain_logits/no_control/similarity.json",
            "../evaluation_peft/gpt2-german-oscar/cls_plain_logits/plain_language/similarity.json",
        ],
        figsize=(14, 5),
        drop_last=1
    )
    # vis_ppl()
    # vis_simp()
    # vis_freq(target_style="easy")
    # vis_corr(tgt_label="Kincaid")
    print("End")
