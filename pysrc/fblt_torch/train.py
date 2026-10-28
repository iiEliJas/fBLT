import torch


def training_step(model, optimizer, x, y):
    optimizer.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    return loss.item()
