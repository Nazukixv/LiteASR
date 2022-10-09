from typing import Any, Iterable


class Vocab(object):
    """Vocabulary

    Example:
        >>> vocab = Vocab('path/to/vocab.txt')
        >>> vocab['的']
        2553
        >>> vocab[2553]
        '的'
        >>> vocab.lookup('中国智造惠及全球')
        [30, 712, 1718, 3738, 1327, 482, 272, 2425]
        >>> vocab.lookup([30, 712, 1718, 3738, 1327, 482, 272, 2425])
        ['中', '国', '智', '造', '惠', '及', '全', '球']

    """

    def __init__(self, vocab_path: str) -> None:
        """Construct vocabulary object

        vocab file format:
            <unk>     1
            ...       ...
            <token>   <tokenid>
            ...       ...
            <sos/eos> ...

        """

        self.token2id = {'<blank>': 0}
        self.id2token = ['<blank>']
        with open(vocab_path, 'r') as vocab:
            for line in vocab.readlines():
                entry = line.strip().split()
                if len(entry) != 2:
                    raise ValueError(f'Invalid line is found:\n>    {line}')
                token, tokenid = entry
                tokenid = int(tokenid)
                if tokenid != len(self.id2token):
                    raise ValueError(f'Missing token id: {len(self.id2token)}')
                self.token2id[token] = tokenid
                self.id2token.append(token)
        self.token2id['<sos/eos>'] = len(self.id2token)
        self.id2token.append('<sos/eos>')

    @property
    def valid(self) -> bool:
        return all(
            [self.id2token[self.token2id[t]] == t for t in self.token2id]
        )

    def __getitem__(self, index):
        if isinstance(index, str):
            if index in self.token2id:
                return self.token2id[index]
            else:
                return self.token2id['<unk>']
        elif isinstance(index, int):
            if index < len(self.id2token):
                return self.id2token[index]
            else:
                raise IndexError('Index out of range of vocabulary')
        else:
            raise KeyError(f'Key {index} is not valid')

    def convert(self, index):
        assert isinstance(index, int)
        if self.id2token[index] in ["<blank>", "<sos/eos>"]:
            return ""
        elif self.id2token[index] in ["<space>"]:
            return " "
        else:
            return self.id2token[index]

    def __len__(self) -> int:
        return len(self.id2token)

    def lookupi(self, seq: Iterable[Any], convert=False):
        if not convert:
            return map(lambda t: self[t], seq)
        else:
            return map(lambda t: self.convert(t), seq)

    def lookup(self, seq: Iterable[Any], convert=False):
        return tuple(self.lookupi(seq, convert=convert))
