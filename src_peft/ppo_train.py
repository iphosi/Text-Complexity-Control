from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers import Adafactor
from torch.optim.lr_scheduler import LinearLR
import torch
from torch.nn.utils.rnn import pad_sequence
from trainer import EasyLangPPOTrainer
from trl import (
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
    create_reference_model
)
from peft import PeftModel, LoraConfig, get_peft_model

from preprocess import (
    split_dataset,
    get_monolingual_dataset,
    get_parallel_dataset,
    adapter_summary
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
    use_ref=False,
    # Dataset Args
    dataset_type="Monolingual",
    target_styles=None,
    query_length=8,
    data_path="../datasets/monolingual_Leichte_Sprache",
    # Adapter Args
    adapter_type="LoRA",
    adapter_name=None,
    target_modules=None,
    checkpoint_path="",
    fuse_subadapters=False,
    weights=None,
    # Training Args
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    use_lr_scheduler=False,
    num_train_epochs=1,
    num_ppo_epochs=1,
    num_steps=100,
    target_kl=0.1,
    init_kl_coef=0.4,
    reward_type="cls",
    reward_scaling_factor=1,
    use_cls_logits=False,
    baselines=None,
    dynamic_baseline=False,
    use_bert_reg=False
):
    # Wandb
    if log_with:
        wandb.login()
        wandb.init(project=project_name)
        print("=" * 100)
        print(f"Project: {project_name}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    print("=" * 100)
    print(f"Baseline model: {model_name}")

    if from_local:
        print("Load the model from local.")
        assert os.path.exists(model_path)

        if task_type == "CAUSAL_LM":
            model = AutoModelForCausalLM.from_pretrained(model_path)
        elif task_type == "SEQ_2_SEQ_LM":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            raise NotImplementedError

        tokenizer = AutoTokenizer.from_pretrained(model_path)

    else:
        print("Load the model from remote.")
        raise NotImplementedError

    if use_ref:
        ref_model = create_reference_model(model)
    else:
        ref_model = None

    for param in model.parameters():
        param.requires_grad = False

    model.enable_input_require_grads()
    model.config.use_cache = False

    # Dataset
    target_styles = ["easy", "plain", "everyday", "special"] if target_styles is None else target_styles
    style2prompt = {
        "easy": "[Leichte Sprache]: ",
        "plain": "[Einfache Sprache]: ",
        "everyday": "[Alltagssprache]: ",
        "special": "[Fachsprache]: "
    }
    prompt_template = list(map(style2prompt.get, target_styles))

    if dataset_type == "Monolingual":
        dataset, dataset_names = get_monolingual_dataset(
            tokenizer=tokenizer,
            max_length=max_length,
            query_length=query_length,
            input_path=data_path
        )

    elif dataset_type == "Parallel":
        dataset, dataset_names = get_parallel_dataset(
            tokenizer=tokenizer,
            max_length=max_length,
            input_path=data_path
        )

    else:
        raise NotImplementedError

    train_set, test_set = split_dataset(dataset=dataset, val_split=0.0, test_split=0.01)

    sample_text = dataset.datasets[0].texts[0]

    print("=" * 100)
    print(f"Prompt template: {prompt_template}")
    print(f"Dataset: {dataset_type}")
    for name in dataset_names:
        print(name)
    print("-" * 100)
    print(f"Text of the sample:\n{sample_text}")
    print("-" * 100)
    print(f"Size: {len(dataset)} || Train: {len(train_set)} || Test: {len(test_set)}")

    # Adapter
    adapter_name = f"{adapter_type}_{dataset_type}" if adapter_name is None else adapter_name

    print("=" * 100)
    print(f"Adapter name: {adapter_name}")

    target_modules = ["c_attn"] if target_modules is None else target_modules

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

    if fuse_subadapters:
        print("Fuse LoRA adapters.")
        subadapter_dict = {
            "easy": f"../adapters/ppo/{model_name}/{adapter_name}/reg/easy/checkpoint-300-0",
            "plain": f"../adapters/ppo/{model_name}/{adapter_name}/reg/plain/checkpoint-300-0",
            "everyday": f"../adapters/ppo/{model_name}/{adapter_name}/reg/everyday/checkpoint-300-0",
            "special": f"../adapters/ppo/{model_name}/{adapter_name}/reg/special/checkpoint-300-0"
        }

        model = PeftModel(model, adapter_config, adapter_name="fusion")

        print("Load subadapters.")
        for i, name in enumerate(target_styles):
            path = subadapter_dict[name]
            model.load_adapter(path, adapter_name=name)

        adapter_names = ["fusion"] + target_styles
        fusion_adapter_name = "_".join(adapter_names)

        weights = [1] * (len(adapter_names)) if weights is None else weights
        assert len(weights) == len(adapter_names)

        model.add_weighted_adapter(adapters=adapter_names, weights=weights, adapter_name=fusion_adapter_name)
        model.set_adapter(fusion_adapter_name)

        print("Select the first target module as fusion head.")
        for name, param in model.named_parameters():
            if fusion_adapter_name in name and target_modules[0] in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif os.path.exists(checkpoint_path):
        print("Resume from checkpoint.")
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            adapter_name="_".join(target_styles),
            is_trainable=True
        )

    else:
        print("Train the adapter from scratch.")
        model = PeftModel(
            model,
            adapter_config,
            adapter_name="_".join(target_styles)
        )

    print(f"Active adapters: {model.active_adapter}")
    print(adapter_summary(model))

    if task_type == "CAUSAL_LM":
        model = AutoModelForCausalLMWithValueHead(
            pretrained_model=model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if ref_model:
            ref_model = AutoModelForCausalLMWithValueHead(
                pretrained_model=ref_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )

    elif task_type == "SEQ_2_SEQ_LM":
        model = AutoModelForSeq2SeqLMWithValueHead(
            pretrained_model=model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if ref_model:
            ref_model = AutoModelForSeq2SeqLMWithValueHead(
                pretrained_model=ref_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )

    else:
        raise NotImplementedError

    # Training
    pad_token_id = tokenizer.pad_token_id

    def ppo_collator(batch):
        encoding_list = list(map(lambda d: d["input_ids"], batch))
        encoding_batch = pad_sequence(encoding_list, batch_first=True, padding_value=pad_token_id)

        attention_mask_list = list(map(lambda d: d["attention_mask"], batch))
        attention_mask_batch = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        return {
            "input_ids": encoding_batch,
            "attention_mask": attention_mask_batch,
        }

    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )

    num_iters = num_train_epochs * len(train_set) // batch_size
    if use_lr_scheduler:
        scheduler = LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_iters)
    else:
        scheduler = None

    output_path = f"../adapters/ppo/{model_name}/{adapter_name}/{project_name}/checkpoints"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ppo_config = PPOConfig(
        log_with=log_with,
        model_name=model_name,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mini_batch_size=mini_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optimize_cuda_cache=True,
        target=target_kl,
        ppo_epochs=num_ppo_epochs,
        init_kl_coef=init_kl_coef,
        cliprange_value=0.2,
        cliprange=0.2,
        gamma=0.99,
        seed=40,
    )

    trainer = EasyLangPPOTrainer(
        save_steps=num_steps,
        save_path=output_path,
        train_epochs=num_train_epochs,
        device=device,
        query_length=query_length,
        max_length=max_length,
        task_type=task_type,
        reward_type=reward_type,
        reward_scaling_factor=reward_scaling_factor,
        use_cls_logits=use_cls_logits,
        baselines=baselines,
        dynamic_baseline=dynamic_baseline,
        use_bert_reg=use_bert_reg,
        prompt_template=prompt_template,
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=train_set,
        data_collator=ppo_collator,
        optimizer=optimizer,
        lr_scheduler=scheduler
    )

    trainer.train()

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
    parser.add_argument("--use_ref", action="store_true")

    # Dataset Args
    parser.add_argument("--dataset_type", type=str, default="Monolingual")
    parser.add_argument("--target_styles", type=str, nargs="+", default=None)
    parser.add_argument("--query_length", type=int, default=8)
    parser.add_argument("--data_path", type=str, default="../datasets/monolingual_Leichte_Sprache")

    # Adapter Args
    parser.add_argument("--adapter_type", type=str, default="LoRA")
    parser.add_argument("--adapter_name", type=str, default=None)
    parser.add_argument("--target_modules", type=str, nargs="+")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--fuse_subadapters", action="store_true")
    parser.add_argument("--weights", type=float, nargs="+", default=None)

    # Training Args
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--use_lr_scheduler", action="store_true")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_ppo_epochs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1500)
    parser.add_argument("--target_kl", type=float, default=0.1)
    parser.add_argument("--init_kl_coef", type=float, default=0.4)

    # Reward Args
    parser.add_argument("--reward_type", type=str, default="cls")
    parser.add_argument("--reward_scaling_factor", type=float, default=1)
    parser.add_argument("--use_cls_logits", action="store_true")
    parser.add_argument("--baselines", type=float, nargs="+", default=None)
    parser.add_argument("--dynamic_baseline", action="store_true")
    parser.add_argument("--use_bert_reg", action="store_true")

    args = parser.parse_args()

    # Debugging
    train_adapter(
        log_with=None,
        model_path="../baseline_models/gpt2-german-oscar/CAUSAL_LM",
        model_name="gpt2-german-oscar",
        use_ref=True,
        from_local=True,
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_fc"],
        target_styles=["plain", "everyday"],
        max_length=1024,
        dataset_type="Monolingual",
        data_path="../datasets/monolingual_Leichte_Sprache/test",
        fuse_subadapters=False,
        batch_size=4,
        reward_type="reg",
        use_bert_reg=False,
        baselines=[8, 8],
        query_length=8
    )

    # train_adapter(
    #     log_with=None,
    #     model_path="../baseline_models/T5-Base_GNAD/SEQ_2_SEQ_LM",
    #     model_name="bart-german",
    #     use_ref=True,
    #     from_local=True,
    #     task_type="SEQ_2_SEQ_LM",
    #     target_modules=["q", "v"],
    #     prompt_template=["[Einfache Sprache]: "],
    #     max_length=1024,
    #     dataset_type="Parallel",
    #     data_path="../datasets/aligned_German_simplification",
    #     fuse_subadapters=False,
    #     add_fusion_adapter=False,
    #     trainable_subadapters=False,
    #     batch_size=1,
    #     reward_type="reg",
    #     use_bert_reg=False,
    #     baselines=[8, 8]
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
        use_ref=args.use_ref,
        # Dataset Args
        dataset_type=args.dataset_type,
        target_styles=args.target_styles,
        query_length=args.query_length,
        data_path=args.data_path,
        # Adapter Args
        adapter_type=args.adapter_type,
        adapter_name=args.adapter_name,
        target_modules=args.target_modules,
        checkpoint_path=args.checkpoint_path,
        fuse_subadapters=args.fuse_subadapters,
        weights=args.weights,
        # Training Args
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        use_lr_scheduler=args.use_lr_scheduler,
        num_train_epochs=args.num_train_epochs,
        num_ppo_epochs=args.num_ppo_epochs,
        num_steps=args.num_steps,
        target_kl=args.target_kl,
        init_kl_coef=args.init_kl_coef,
        # Reward Args
        reward_type=args.reward_type,
        reward_scaling_factor=args.reward_scaling_factor,
        use_cls_logits=args.use_cls_logits,
        baselines=args.baselines,
        dynamic_baseline=args.dynamic_baseline,
        use_bert_reg=args.use_bert_reg
    )

