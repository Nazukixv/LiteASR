import os

from liteasr.dataclass.vocab import Vocab


class AudioSheet(object):

    def __init__(self, data_cfg):
        all_files = os.listdir(data_cfg)

        # find feats.scp & utt2num_frames in priority
        if "feats.scp" in all_files:
            self.scp = f"{data_cfg}/feats.scp"
            assert "utt2num_frames" in all_files
            self.shape = f"{data_cfg}/utt2num_frames"
            self.segments = None
        elif "wav.scp" in all_files:
            self.scp = f"{data_cfg}/wav.scp"
            self.segments = f"{data_cfg}/segments" \
                if "segments" in all_files else None
        else:
            raise FileNotFoundError(f"wav.scp not found in {data_cfg}")

    def __iter__(self):
        if self.scp.endswith("feats.scp"):
            with open(self.scp, "r") as fscp, open(self.shape, "r") as fshp:
                for scp_line, shp_line in zip(
                    fscp.readlines(), fshp.readlines()
                ):
                    scp_entry = scp_line.strip().split(None, 1)
                    shp_entry = shp_line.strip().split(None, 1)
                    if len(scp_entry) != 2 or len(shp_entry) != 2:
                        raise ValueError(
                            "Invalid line found:\n"
                            f">\t{scp_line}\n"
                            f">\t{shp_line}"
                        )
                    wavid, wavfd = scp_entry
                    wavid_, frames = shp_entry
                    assert wavid == wavid_
                    yield wavid, wavfd, -1, -1, [int(frames), 83]
        elif self.segments is not None:
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
        else:
            raise NotImplementedError


class TextSheet(object):

    def __init__(self, data_cfg, vocab: Vocab):
        self.text = f"{data_cfg}/text.split"
        self.vocab = vocab

    def __iter__(self):
        with open(self.text, 'r') as ftxt:
            for line in ftxt.readlines():
                entry = line.strip().split()
                uttid, *tokens = entry
                text = "".join(tokens)  # naive impl
                yield uttid, self.vocab.lookup(tokens), text
