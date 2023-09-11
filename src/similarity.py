import torch
from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics import PearsonCorrCoef


def get_pearson_scores(
    src_rep_spaces,
    tgt_rep_spaces,
    device
):
    pearson = PearsonCorrCoef().to(device)
    return list(map(lambda src, tgt: pearson(src, tgt).item(), src_rep_spaces, tgt_rep_spaces))


def get_rep_spaces(
    model,
    tokenizer,
    device,
    texts,
    num_sample_tokens=1000,
    seed=40
):
    """
    Returns the flattened upper triangular of the similarity matrix in each layer.
    """

    batch_input_ids = (
        tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        for text in texts
    )
    # hidden_states: Tuple of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size).
    # One for the output of the embeddings + One for the output of each layer
    with torch.no_grad():
        batch_hidden_states = (
            model(
                input_ids=input_ids,
                output_hidden_states=True
            ).hidden_states
            for input_ids in batch_input_ids
        )

    rep_spaces = []
    generator = torch.Generator().manual_seed(seed)

    for layer_hidden_states in zip(*batch_hidden_states):
        layer_hidden_states = torch.cat(
            layer_hidden_states,
            dim=1
        )[0]
        num_tokens = layer_hidden_states.size(0)

        if num_tokens >= num_sample_tokens:
            sample_idx = torch.randint(
                low=0,
                high=num_tokens,
                size=(num_sample_tokens,),
                generator=generator
            )
            sim_matrix = pairwise_cosine_similarity(layer_hidden_states[sample_idx, :])

        else:
            sim_matrix = pairwise_cosine_similarity(layer_hidden_states)

        rep_space = torch.triu(sim_matrix).view(-1)
        rep_spaces.append(rep_space)

    return rep_spaces
