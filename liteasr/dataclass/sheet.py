import os
from typing import Optional

import soundfile as sf

from liteasr.dataclass.vocab import Vocab


def _get_lineno(file_name: Optional[str]) -> int:
    if file_name is None:
        return 0

    with open(file_name, "r") as f:
        for i, _ in enumerate(f):
            pass
        return i + 1


class AudioSheet(object):

    def __init__(self, data_cfg):
        all_files = os.listdir(data_cfg)

        # find feats.scp & utt2num_frames in priority
        if "feats.scp" in all_files:
            self.scp = f"{data_cfg}/feats.scp"
            assert "utt2num_frames" in all_files
            self.shape = f"{data_cfg}/utt2num_frames"
            self.segments = None
            self.lines = _get_lineno(self.scp)
        elif "wav.scp" in all_files:
            self.scp = f"{data_cfg}/wav.scp"
            self.segments = f"{data_cfg}/segments" \
                if "segments" in all_files else None
            self.lines = max(_get_lineno(self.scp), _get_lineno(self.segments))
        else:
            raise FileNotFoundError(f"wav.scp not found in {data_cfg}")

    def __iter__(self):
        if self.scp.endswith("feats.scp"):
            with open(self.scp, "r") as fscp, open(self.shape, "r") as fshp:
                while True:
                    scp_line, shp_line = fscp.readline(), fshp.readline()
                    if not scp_line or not shp_line:
                        break
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
                    yield wavid, wavfd, None, int(frames)
        elif self.segments is not None:
            with open(self.segments, 'r') as fseg, open(self.scp, 'r') as fscp:
                fds = {}
                while True:
                    line = fscp.readline()
                    if not line:
                        break
                    entry = line.strip().split(None, 1)
                    if len(entry) != 2:
                        raise ValueError(f"Invalid line is found:\n>   {line}")
                    wavid, wavfd = entry
                    fds[wavid] = wavfd

                while True:
                    line = fseg.readline()
                    if not line:
                        break
                    entry = line.strip().split()
                    if len(entry) != 4:
                        raise ValueError(f"Invalid line is found:\n>   {line}")
                    uttid, wavid, start, end = entry
                    start = round(float(start) * 16000)
                    end = round(float(end) * 16000)
                    shape = end - start - 1
                    yield uttid, fds[wavid], start, shape
        else:
            with open(self.scp, "r") as fscp:
                while True:
                    line = fscp.readline()
                    if not line:
                        break
                    entry = line.strip().split(None, 1)
                    if len(entry) != 2:
                        raise ValueError(f"Invalid line is found:\n>   {line}")
                    wavid, wavfd = entry
                    samples, _ = sf.read(wavfd)
                    yield wavid, wavfd, 0, len(samples)

    def __len__(self):
        return self.lines


class TextSheet(object):

    def __init__(self, data_cfg, vocab: Vocab):
        self.text = f"{data_cfg}/text"
        self.vocab = vocab
        self.lines = _get_lineno(self.text)

    def __iter__(self):
        with open(self.text, 'r') as ftxt:
            while True:
                line = ftxt.readline()
                if not line:
                    break
                entry = line.strip().split()
                uttid, *tokens = entry
                text = "".join(tokens)  # naive impl
                yield uttid, self.vocab.lookup(text), text

    def __len__(self):
        return self.lines
