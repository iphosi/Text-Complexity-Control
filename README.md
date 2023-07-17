# Text Complexity Control
## Overview
This is the repo for the text complexity control project,
which aims to develop a controllable decoder model that generates German text at different complexity levels.
The repo contains:

- The adapters for multi-complexity-level text generation.
- The source code to train and evaluate the models.
- The evaluation results.

Note: The training data is from a private dataset and thus not provided in this repo.
Actually, any text dataset can be used as an alternative as the training process only requires some short query texts.

## Baseline Model
`ml6team/gpt2-small-german-finetune-oscar` is used as the baseline model.
Please download the model and save it under `./baseline_models/gpt2-german-oscar/CAUSAL_LM`

## Training Strategy
Basically, the baseline model parameters are frozen and the inserted `LoRA` adapter modules are trained on the stylistic
continuation task via `PPO`.
### Stylistic Continuation
The first n tokens of a text sample will be fed into the model together with one of the following control tokens.
The model should extend the query text at a certain complexity level accordingly.

| Control Token      | Target Class      | Target Class ID |
|--------------------|-------------------|-----------------|
| [Leichte Sprache]  | Easy Language     | 0               |
| [Einfache Sprache] | Plain Language    | 1               |
| [Alltagssprache]   | Everyday Language | 2               |
| [Fachsprache]      | Special Language  | 3               |

Prompt template: `<ctrl token>: <query text>`

## Reward Modeling

### Regression Manner
$$\cdot$$

### Classification Manner