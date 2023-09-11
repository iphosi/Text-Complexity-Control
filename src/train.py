import torch.cuda
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding
)
from trainer import ContrastiveTrainer

from transformers.adapters import (
    AdapterConfig,
    CompacterPlusPlusConfig,
    AdapterTrainer
)

from preprocess import (
    split_dataset,
    get_monolingual_dataset,
    get_text_complexity_dataset
)

from argparse import ArgumentParser
import os

import wandb


def train_adapter(
    # Wandb Args
    log_with=None,
    project_name="huggingface",
    # Model Args
    model_path=None,
    model_name=None,
    task_type="CAUSAL_LM",
    max_length=1024,
    from_local=False,
    # Dataset Args
    num_phrases_per_sample=16,
    text_column_name="phrase",
    dataset_type="Monolingual",
    target_label="Kincaid",
    do_rescaling=False,
    max_value=None,
    min_value=None,
    rescaling_factor=None,
    data_path="../datasets/monolingual_Leichte_Sprache",
    # Adapter Args
    use_adapter=False,
    adapter_type="ADP_BN_S",
    adapter_name=None,
    checkpoint_path="",
    leave_out=None,
    # Training Args
    batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=4e-4,
    warmup_steps=0,
    num_train_epochs=1,
    num_steps=1500,
    early_stopping_patience=4,
    label_smoothing_factor=0.0,
    use_contrastive=False,
    disable_tqdm=False
):
    # Wandb
    if log_with:
        wandb.login()
        wandb.init(project=project_name)
        print("=" * 100)
        print(f"Project: {project_name}")
    else:
        os.environ["WANDB_DISABLED"] = "true"

    # Model
    print("=" * 100)
    print(f"Baseline model: {model_name}")

    if from_local:
        print("Load the model from local.")
        if task_type == "CAUSAL_LM":
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
        elif task_type == "SEQ_CLS":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            raise NotImplementedError

    else:
        print("Load the model from remote.")
        raise NotImplementedError

    model.config.use_cache = False

    # Dataset
    adapter_type = adapter_type if use_adapter else "FT"
    if task_type == "CAUSAL_LM":
        adapter_name = f"{adapter_type}_{dataset_type}" if adapter_name is None else adapter_name

        if dataset_type == "Monolingual":
            dataset, dataset_names = get_monolingual_dataset(
                tokenizer=tokenizer,
                max_length=max_length,
                num_phrases_per_sample=num_phrases_per_sample,
                text_column_name=text_column_name,
                query_length=0,
                input_path=data_path
            )
        else:
            raise NotImplementedError

        val_split = 0.01
        test_split = 0.01

        sample_text = dataset.datasets[0].texts[0]
        print("=" * 100)
        print(f"Dataset: {dataset_type}")
        for name in dataset_names:
            print(name)
        print("-" * 100)
        print(f"Text of the first sample:\n{sample_text}")
        print("-" * 100)

    elif task_type == "SEQ_CLS":
        adapter_name = f"{adapter_type}_{target_label}" if adapter_name is None else adapter_name

        dataset, dataset_names = get_text_complexity_dataset(
            tokenizer=tokenizer,
            max_length=max_length,
            text_column_name=text_column_name,
            target_label=target_label,
            do_rescaling=do_rescaling,
            max_value=max_value,
            min_value=min_value,
            rescaling_factor=rescaling_factor,
            input_path=data_path
        )

        val_split = 0.05
        test_split = 0.05

        sample_text = dataset.datasets[0].texts[0]
        sample_label = dataset[0]["labels"]
        print("=" * 100)
        print(f"Dataset: {target_label}")
        for name in dataset_names:
            print(name)
        print("-" * 100)
        print(f"Text of the first sample:\n{sample_text}")
        print("-" * 100)
        print(f"Label of the first sample: {sample_label:.2f}")
        print(f"Maximum of labels: {dataset.max:.2f}")
        print(f"Minimum of labels: {dataset.min:.2f}")
        print(f"Rescaling factor: {rescaling_factor}")

    else:
        raise NotImplementedError

    dataset_size = len(dataset)
    train_set, val_set, test_set = split_dataset(dataset=dataset, val_split=val_split, test_split=test_split)

    print(f"Size: {dataset_size} || Train: {len(train_set)} || Val: {len(val_set)} || Test: {len(test_set)}")

    # Adapter
    adapter_dict = {
        "ADP_BN_S": AdapterConfig(
            mh_adapter=False,
            output_adapter=True,
            reduction_factor=16,
            non_linearity="gelu"
        ),
        "Comp++": CompacterPlusPlusConfig(
            reduction_factor=16,
            phm_dim=4,
            phm_rank=14
        )
    }

    if use_adapter:
        adapter_config = adapter_dict[adapter_type]
        print("=" * 100)
        print(f"Adapter name: {adapter_name}")

        if os.path.exists(checkpoint_path):
            print("Resume from checkpoint.")
            model.load_adapter(adapter_name_or_path=checkpoint_path, leave_out=leave_out)
        else:
            print("Train the adapter from scratch.")
            model.add_adapter(adapter_name=adapter_name, config=adapter_config)

        model.train_adapter(adapter_setup=adapter_name)
    else:
        print("=" * 100)
        print(f"Full fine-tuning.")
    print(model.adapter_summary())

    # Training
    training_args = TrainingArguments(
        report_to=log_with,
        output_dir=f"../adapters/{model_name}/stylistic_adaption/{adapter_name}/{project_name}/checkpoints",
        remove_unused_columns=False,
        lr_scheduler_type="linear",
        optim="adafactor",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        label_smoothing_factor=label_smoothing_factor,
        gradient_accumulation_steps=gradient_accumulation_steps,
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

        if use_adapter:
            if use_contrastive:
                vocab_size = len(tokenizer)
                model.resize_token_embeddings(vocab_size)
                trainer = ContrastiveTrainer(
                    vocab_size=vocab_size,
                    pad_token_id=tokenizer.pad_token_id,
                    margin=0.5,
                    model=model,
                    args=training_args,
                    tokenizer=tokenizer,
                    train_dataset=train_set,
                    eval_dataset=val_set,
                    data_collator=data_collator,
                )
            else:
                trainer = AdapterTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=train_set,
                    eval_dataset=val_set,
                    data_collator=data_collator
                )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_set,
                eval_dataset=val_set,
                data_collator=data_collator
            )

    elif task_type == "SEQ_CLS":
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        if use_adapter:
            trainer = AdapterTrainer(
                model=model,
                args=training_args,
                train_dataset=train_set,
                eval_dataset=val_set,
                data_collator=data_collator
            )
        else:
            trainer = Trainer(
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

    if use_adapter:
        model.save_adapter(
            save_directory=f"../adapters/{model_name}/stylistic_adaption/{adapter_name}/{project_name}/model",
            adapter_name=adapter_name
        )
    else:
        model.save_pretrained(
            save_directory=f"../adapters/{model_name}/stylistic_adaption/{adapter_name}/{project_name}/model",
        )

    if log_with:
        wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()

    # Wandb Args
    parser.add_argument("--log_with", type=str, default=None)
    parser.add_argument("--project_name", type=str, default="huggingface")

    # Model Args
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--task_type", type=str, default="CAUSAL_LM")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--from_local", action="store_true")

    # Dataset Args
    parser.add_argument("--num_phrases_per_sample", type=int, default=16)
    parser.add_argument("--text_column_name", type=str, default="phrase")
    parser.add_argument("--dataset_type", type=str, default="Monolingual")

    parser.add_argument("--target_label", type=str, default="Kincaid")
    parser.add_argument("--do_rescaling", action="store_true")
    parser.add_argument("--max_value", type=float, default=None)
    parser.add_argument("--min_value", type=float, default=None)
    parser.add_argument("--rescaling_factor", type=float, default=None)
    parser.add_argument("--data_path", type=str, default="../datasets/monolingual_Leichte_Sprache")

    # Adapter Args
    parser.add_argument("--use_adapter", action="store_true")
    parser.add_argument("--adapter_type", type=str, default="ADP_BN_S")
    parser.add_argument("--adapter_name", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--leave_out", type=int, nargs='+', default=[])

    # Training Args
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1500)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--label_smoothing_factor", type=float, default=0.0)
    parser.add_argument("--use_contrastive", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true")

    args = parser.parse_args()

    # Debugging
    # train_adapter(
    #     log_with=None,
    #     project_name="test",
    #     model_path="../baseline_models/gpt2-german-oscar/CAUSAL_LM",
    #     model_name="gpt2-german-oscar",
    #     task_type="CAUSAL_LM",
    #     from_local=True,
    #     dataset_type="Monolingual",
    #     data_path="../datasets/monolingual_Leichte_Sprache/test",
    #     use_adapter=False,
    #     adapter_type="ADP_BN_S",
    #     checkpoint_path="",
    #     batch_size=4,
    #     gradient_accumulation_steps=4,
    #     learning_rate=4e-4,
    #     warmup_steps=0,
    #     num_train_epochs=8,
    #     num_steps=100,
    #     early_stopping_patience=3,
    #     label_smoothing_factor=0.1,
    #     disable_tqdm=True
    # )

    train_adapter(
        # Wandb Args
        log_with=args.log_with,
        project_name=args.project_name,
        # Model Args
        model_path=args.model_path,
        model_name=args.model_name,
        task_type=args.task_type,
        max_length=args.max_length,
        from_local=args.from_local,
        # Dataset Args
        num_phrases_per_sample=args.num_phrases_per_sample,
        text_column_name=args.text_column_name,
        dataset_type=args.dataset_type,
        target_label=args.target_label,
        do_rescaling=args.do_rescaling,
        max_value=args.max_value,
        min_value=args.min_value,
        rescaling_factor=args.rescaling_factor,
        data_path=args.data_path,
        # Adapter Args
        use_adapter=args.use_adapter,
        adapter_type=args.adapter_type,
        adapter_name=args.adapter_name,
        checkpoint_path=args.checkpoint_path,
        leave_out=args.leave_out,
        # Training Args
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        num_steps=args.num_steps,
        early_stopping_patience=args.early_stopping_patience,
        label_smoothing_factor=args.label_smoothing_factor,
        use_contrastive=args.use_contrastive,
        disable_tqdm=args.disable_tqdm
    )

