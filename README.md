# LiteASR
A lite framework focusing on ASR.

```shell
cd liteasr
python train.py \
  +task.name=asr \
  +task.scp=/path/to/feats.scp \
  +task.segments=null \
  +task.vocab=/path/to/vocab \
  +task.text=/path/to/text
```
