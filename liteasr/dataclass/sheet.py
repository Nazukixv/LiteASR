from liteasr.dataclass.vocab import Vocab
from liteasr.utils.kaldiio import load_mat


class AudioSheet(object):

    def __init__(self, scp, segments=None):
        self.scp = scp
        self.segments = segments

    def __iter__(self):
        # 默认没有 segments 是 feats.scp
        if self.segments is None:
            with open(self.scp, 'r') as fscp:
                for line in fscp.readlines():
                    entry = line.strip().split(None, 1)
                    if len(entry) != 2:
                        raise ValueError(f"Invalid line is found:\n>   {line}")
                    wavid, wavfd = entry
                    shape = list(load_mat(wavfd).shape)
                    yield wavid, wavfd, -1, -1, shape

        # 存在 segments 则为 wav.scp
        else:
            with open(self.segments, 'r') as fseg, open(self.scp, 'r') as fscp:
                fds = {}
                for line in fscp.readlines():
                    entry = line.strip().split(None, 1)
                    if len(entry) != 2:
                        raise ValueError(f"Invalid line is found:\n>   {line}")
                    wavid, wavfd = entry
                    fds[wavid] = wavfd

                for line in fseg.readlines():
                    entry = line.strip().split()
                    if len(entry) != 4:
                        raise ValueError(f"Invalid line is found:\n>   {line}")
                    uttid, wavid, start, end = entry
                    start, end = float(start), float(end)
                    shape = [int(end * 16000) - int(start * 16000) - 1]
                    yield uttid, fds[wavid], start, end, shape


class TextSheet(object):

    def __init__(self, text, vocab: Vocab):
        self.text = text
        self.vocab = vocab

    def __iter__(self):
        with open(self.text, 'r') as ftxt:
            for line in ftxt.readlines():
                entry = line.strip().split()
                uttid, *tokens = entry
                yield uttid, self.vocab.lookup(tokens)
