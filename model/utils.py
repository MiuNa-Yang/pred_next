from torch.nn import Module


def freeze(net: Module):
    for p in net.parameters():
        p.requires_grad = False
