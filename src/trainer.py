import torch

from transformers.adapters import AdapterTrainer
from simctg.lossfunction import SimCTGLoss


class ContrastiveTrainer(AdapterTrainer):
    def __init__(
        self,
        vocab_size,
        pad_token_id,
        margin=0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.margin = margin

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
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
