from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    Seq2SeqTrainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq
)
from peft import PeftModel, LoraConfig, get_peft_model
import torch
import numpy as np
import evaluate

from preprocess import (
    split_dataset,
    get_monolingual_dataset,
    get_parallel_dataset,
    get_text_complexity_dataset
)

from argparse import ArgumentParser
import os

import wandb


def train_adapter(
    # Wandb Args
    project_name="huggingface",
    # Model Args
    model_path=None,
    model_name=None,
    task_type="CAUSAL_LM",
    max_length=1024,
    num_labels=1,
    id2label=None,
    label2id=None,
    from_local=False,
    # Dataset Args
    num_phrases_per_sample=16,
    text_column_name="phrase",
    dataset_type="Monolingual",
    prompt_template=None,
    target_label="Kincaid",
    do_rescaling=False,
    max_value=None,
    min_value=None,
    rescaling_factor=None,
    data_path="../datasets/monolingual_Leichte_Sprache",
    # Adapter Args
    adapter_type="LoRA",
    adapter_name=None,
    target_modules=None,
    checkpoint_path="",
    # Training Args
    batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    learning_rate=4e-4,
    warmup_steps=0,
    num_train_epochs=1,
    num_steps=1500,
    early_stopping_patience=4,
    label_smoothing_factor=0.0,
    disable_tqdm=False
):
    # Wandb
    wandb.login()
    wandb.init(project=project_name)

    # Task type
    task_type_set = {
        "CAUSAL_LM",
        "SEQ_CLS",
        "SEQ_2_SEQ_LM",
        "TOKEN_CLS"
    }
    assert task_type in task_type_set

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    print("=" * 100)
    print(f"Baseline model: {model_name}")

    if from_local:
        print("Load the model from local.")
        if task_type == "CAUSAL_LM":
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto"
            )
        elif task_type == "SEQ_CLS":
            if num_labels > 1:
                id2label = {
                    0: "A2",
                    1: "B1",
                    2: "B2"
                } if id2label is None else id2label

                label2id = {
                    "A2": 0,
                    "B1": 1,
                    "B2": 2
                } if label2id is None else label2id

                assert num_labels == len(id2label) == len(label2id)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_mismatched_sizes=True
                )
            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=num_labels
                )

            model.to(device)

        elif task_type == "SEQ_2_SEQ_LM":
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                device_map="auto"
            )
        else:
            raise NotImplementedError

        tokenizer = AutoTokenizer.from_pretrained(model_path)

    else:
        print("Load the model from remote.")
        raise NotImplementedError

    for param in model.parameters():
        param.requires_grad = False

    model.enable_input_require_grads()
    model.config.use_cache = False

    # Dataset
    prompt_template = [
        "Textvereinfachung", "Normale Sprache", "Leichte Sprache"
    ] if prompt_template is None else prompt_template

    if task_type == "CAUSAL_LM":
        adapter_name = f"{adapter_type}_{dataset_type}" if adapter_name is None else adapter_name

        if dataset_type == "Monolingual":
            dataset, dataset_names = get_monolingual_dataset(
                tokenizer=tokenizer,
                max_length=max_length,
                num_phrases_per_sample=num_phrases_per_sample,
                text_column_name=text_column_name,
                input_path=data_path
            )
        elif dataset_type == "Parallel":
            dataset, dataset_names = get_parallel_dataset(
                tokenizer=tokenizer,
                max_length=max_length,
                prompt_template=prompt_template,
                input_path=data_path
            )
        else:
            raise ValueError("Unknown dataset type.")

        sample_text = dataset.datasets[0].texts[0]
        print("=" * 100)
        print(f"Dataset: {dataset_type}")
        for name in dataset_names:
            print(name)
        print("-" * 100)
        print(f"Text of the sample:\n{sample_text}")

    elif task_type == "SEQ_CLS":
        adapter_name = f"{adapter_type}_{target_label}" if adapter_name is None else adapter_name

        dataset, dataset_names = get_text_complexity_dataset(
            tokenizer=tokenizer,
            max_length=max_length,
            num_labels=num_labels,
            label2id=label2id,
            text_column_name=text_column_name,
            target_label=target_label,
            do_rescaling=do_rescaling,
            max_value=max_value,
            min_value=min_value,
            rescaling_factor=rescaling_factor,
            input_path=data_path
        )

        sample_text = dataset.datasets[0].texts[0]
        sample_label = dataset[0]["labels"]
        print("=" * 100)
        print(f"Dataset: {target_label}")
        for name in dataset_names:
            print(name)
        print("-" * 100)
        print(f"Text of the sample:\n{sample_text}")
        print("-" * 100)
        print(f"Label of the sample: {sample_label:.2f}")
        print(f"Maximum of labels: {dataset.max:.2f}")
        print(f"Minimum of labels: {dataset.min:.2f}")

    elif task_type == "SEQ_2_SEQ_LM":
        assert dataset_type == "Parallel"
        adapter_name = f"{adapter_type}_{dataset_type}" if adapter_name is None else adapter_name

        dataset, dataset_names = get_parallel_dataset(
            tokenizer=tokenizer,
            max_length=max_length,
            prompt_template=prompt_template,
            train_seq2seq=True,
            input_path=data_path
        )

        sample_text = dataset.datasets[0].texts[0]
        sample_label = dataset.datasets[0].tgt_texts[0]
        print("=" * 100)
        print(f"Dataset: {dataset_type}")
        for name in dataset_names:
            print(name)
        print("-" * 100)
        print(f"Text of the sample:\n{sample_text}")
        print("-" * 100)
        print(f"Label of the sample:\n{sample_label}")

    else:
        raise NotImplementedError

    if dataset_type == "Monolingual" and task_type == "CAUSAL_LM":
        ft_dataset, _ = split_dataset(dataset, val_split=0.5)

        train_set, val_set = split_dataset(dataset=ft_dataset, val_split=0.01, test_split=0)

        dataset_size = len(ft_dataset)
        train_set_size = len(train_set)
        val_set_size = len(val_set)
        test_set_size = 0

    elif dataset_type == "Monolingual" and task_type == "SEQ_CLS":
        train_set, val_set, test_set = split_dataset(dataset=dataset, val_split=0.1, test_split=0.1)
        dataset_size = len(dataset)
        train_set_size = len(train_set)
        val_set_size = len(val_set)
        test_set_size = len(test_set)

    elif dataset_type == "Parallel" and task_type == "SEQ_2_SEQ_LM":
        ft_dataset, _ = split_dataset(dataset, val_split=0.5)

        train_set, val_set, test_set = split_dataset(dataset=ft_dataset, val_split=0.1, test_split=0.1)
        dataset_size = len(ft_dataset)
        train_set_size = len(train_set)
        val_set_size = len(val_set)
        test_set_size = len(test_set)

    else:
        raise NotImplementedError

    print("-" * 100)
    print(f"Size: {dataset_size} || Train: {train_set_size} || Val: {val_set_size} || Test: {test_set_size}")

    # Adapter
    target_modules = ["query_key_value"] if target_modules is None else target_modules

    adapter_dict = {
        "LoRA": LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=task_type
        )
    }

    adapter_config = adapter_dict[adapter_type]

    print("=" * 100)
    print(f"Adapter name: {adapter_name}")
    print("Target modules: " + " ".join(target_modules))

    if os.path.exists(checkpoint_path):
        print("Resume from checkpoint.")
        model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
    else:
        print("Train the adapter from scratch.")
        model = get_peft_model(model, adapter_config)

    model.print_trainable_parameters()

    # Training
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    if task_type == "SEQ_2_SEQ_LM":
        training_args = Seq2SeqTrainingArguments(
            report_to=["wandb"],
            output_dir=f"../adapters/{model_name}/{adapter_name}/checkpoints",
            remove_unused_columns=False,
            lr_scheduler_type="linear",
            optim="adafactor",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            label_smoothing_factor=label_smoothing_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # gradient_checkpointing=gradient_checkpointing,
            fp16=True,
            predict_with_generate=True,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            save_strategy="steps",
            evaluation_strategy="steps",
            save_steps=num_steps,
            eval_steps=num_steps,
            logging_steps=num_steps,
            save_total_limit=1,
            overwrite_output_dir=True,
            load_best_model_at_end=True,
            disable_tqdm=disable_tqdm
        )
    else:
        training_args = TrainingArguments(
            report_to=["wandb"],
            output_dir=f"../adapters/{model_name}/{adapter_name}/checkpoints",
            remove_unused_columns=False,
            lr_scheduler_type="linear",
            optim="adafactor",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            label_smoothing_factor=label_smoothing_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # gradient_checkpointing=gradient_checkpointing,
            fp16=True,
            warmup_steps=warmup_steps,
            num_train_epochs=num_train_epochs,
            save_strategy="steps",
            evaluation_strategy="steps",
            save_steps=num_steps,
            eval_steps=num_steps,
            logging_steps=num_steps,
            save_total_limit=1,
            overwrite_output_dir=True,
            load_best_model_at_end=True,
            disable_tqdm=disable_tqdm
        )

    if task_type == "CAUSAL_LM":
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=val_set,
            data_collator=data_collator
        )

    elif task_type == "SEQ_CLS":
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        if num_labels > 1:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_set,
                eval_dataset=val_set,
                data_collator=data_collator,
                compute_metrics=compute_metrics
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_set,
                eval_dataset=val_set,
                data_collator=data_collator
            )

    elif task_type == "SEQ_2_SEQ_LM":
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_set,
            eval_dataset=val_set,
            data_collator=data_collator
        )

    else:
        raise NotImplementedError

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    print("=" * 100)
    print("Start training.")
    trainer.train()

    model.save_pretrained(save_directory=f"../adapters/{model_name}/{adapter_name}/model")

    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    # Wandb Args
    parser.add_argument("--project_name", type=str, default="huggingface")

    # Model Args
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--task_type", type=str, default="CAUSAL_LM")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--from_local", action="store_true")

    # Dataset Args
    parser.add_argument("--num_phrases_per_sample", type=int, default=16)
    parser.add_argument("--text_column_name", type=str, default="phrase")
    parser.add_argument("--dataset_type", type=str, default="Monolingual")

    parser.add_argument("--prompt_template", type=str, nargs="+", default=None)
    parser.add_argument("--target_label", type=str, default="Kincaid")
    parser.add_argument("--do_rescaling", action="store_true")
    parser.add_argument("--max_value", type=float, default=None)
    parser.add_argument("--min_value", type=float, default=None)
    parser.add_argument("--rescaling_factor", type=float, default=None)
    parser.add_argument("--data_path", type=str, default="../datasets/monolingual_Leichte_Sprache")

    # Adapter Args
    parser.add_argument("--adapter_type", type=str, default="LoRA")
    parser.add_argument("--adapter_name", type=str, default=None)
    parser.add_argument("--target_modules", type=str, nargs="+")
    parser.add_argument("--checkpoint_path", type=str, default="")

    # Training Args
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1500)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--label_smoothing_factor", type=float, default=0.0)
    parser.add_argument("--disable_tqdm", action="store_true")

    args = parser.parse_args()

    # Debugging
    # train_adapter(
    #     model_path="../baseline_models/distilbert-german/SEQ_CLS",
    #     model_name="distilbert-german",
    #     num_labels=3,
    #     max_length=512,
    #     task_type="SEQ_CLS",
    #     from_local=True,
    #     dataset_type="Monolingual",
    #     target_label="Level",
    #     data_path="../datasets/TextComplexity/apa",
    #     adapter_type="LoRA",
    #     target_modules=["q_lin", "v_lin"],
    #     checkpoint_path="",
    #     batch_size=2,
    #     gradient_accumulation_steps=32,
    #     learning_rate=4e-4,
    #     warmup_steps=0,
    #     num_train_epochs=4,
    #     num_steps=100,
    #     early_stopping_patience=3,
    #     label_smoothing_factor=0.1,
    #     disable_tqdm=False
    # )

    train_adapter(
        # Wandb Args
        project_name=args.project_name,
        # Model Args
        model_path=args.model_path,
        model_name=args.model_name,
        task_type=args.task_type,
        max_length=args.max_length,
        num_labels=args.num_labels,
        from_local=args.from_local,
        # Dataset Args
        num_phrases_per_sample=args.num_phrases_per_sample,
        text_column_name=args.text_column_name,
        dataset_type=args.dataset_type,
        prompt_template=args.prompt_template,
        target_label=args.target_label,
        do_rescaling=args.do_rescaling,
        max_value=args.max_value,
        min_value=args.min_value,
        rescaling_factor=args.rescaling_factor,
        data_path=args.data_path,
        # Adapter Args
        adapter_type=args.adapter_type,
        adapter_name=args.adapter_name,
        target_modules=args.target_modules,
        checkpoint_path=args.checkpoint_path,
        # Training Args
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        num_steps=args.num_steps,
        early_stopping_patience=args.early_stopping_patience,
        label_smoothing_factor=args.label_smoothing_factor,
        disable_tqdm=args.disable_tqdm
    )

