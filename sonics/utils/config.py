from types import SimpleNamespace


def dict2cfg(d):
    """
    Converts a dictionary into a SimpleNamespace
    """
    for k, v in d.items():
        if type(v) == dict:
            d[k] = SimpleNamespace(**v)
    c = SimpleNamespace(**d)
    c.audio.max_len = int(c.audio.max_time * c.audio.sample_rate)
    return c

def cfg2dict(cfg):
    """
    Converts a SimpleNamespace into a dictionary without modifying the original cfg.
    """
    d = vars(cfg).copy()
    for k, v in d.items():
        if isinstance(v, SimpleNamespace):
            d[k] = cfg2dict(v)
    return d
