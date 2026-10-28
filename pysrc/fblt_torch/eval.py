import torch.nn.functional as F


def causal_bpb(logits, targets):
    # logits: (N,V); targets: (N,) next bytes
    ce = F.cross_entropy(logits, targets.long(), reduction="mean")
    return ce.item() / 0.69314718056  # nats -> bits


def masked_acc(logits, targets, mask):
    preds = logits.argmax(dim=-1)
    correct = (preds == targets) & mask
    if mask.sum() == 0:
        return 0.0
    return (correct.sum().float() / mask.sum().float()).item()
