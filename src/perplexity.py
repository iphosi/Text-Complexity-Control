import torch
from tqdm import tqdm


def get_ppl(
    model,
    tokenizer,
    texts,
    device
):
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt", add_special_tokens=False)

    if hasattr(model.config, "n_positions"):
        max_length = model.config.n_positions
    elif hasattr(model.config, "seq_length"):
        max_length = model.config.seq_length
    else:
        max_length = 1024
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            # Loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return torch.exp(torch.stack(nlls).sum() / end_loc).item()


def get_mod_ppl(
    model,
    tokenizer,
    texts,
    max_length,
    stride,
    device
):
    encodings = [tokenizer(text, return_tensors="pt", add_special_tokens=False) for text in texts]

    if hasattr(model.config, "n_positions"):
        max_length = model.config.n_positions if max_length is None else max_length
    elif hasattr(model.config, "seq_length"):
        max_length = model.config.seq_length if max_length is None else max_length
    else:
        max_length = 1024 if max_length is None else max_length

    stride = max_length if stride is None else stride

    nlls = []
    for sample in tqdm(encodings):
        sample_len = sample['input_ids'].size(1)
        sample_nlls = []
        for begin_loc in range(0, sample_len, stride):
            end_loc = min(begin_loc + stride, sample_len)
            # Do not create samples with len 1
            if (sample_len - (begin_loc + stride)) == 1:
                end_loc += 1
            input_ids = sample['input_ids'][:, begin_loc:end_loc].to(device)
            if len(input_ids[0]) == 1:
                print('producing nan')
            target_ids = input_ids.clone()

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids).loss
            sample_nlls.append(outputs)

            if end_loc == sample_len:
                break

        nlls.append(sum(sample_nlls) / len(sample_nlls))
    return torch.exp(torch.stack(nlls).sum() / len(nlls)).item()
