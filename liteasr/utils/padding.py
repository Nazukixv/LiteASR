"""Padding a tensor list."""


def pad(xs, pad_value):
    batch = len(xs)
    l_max = max(x.size(0) for x in xs)
    padded_xs = xs[0].new(batch, l_max, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(batch):
        padded_xs[i, :xs[i].size(0)] = xs[i]
    return padded_xs
