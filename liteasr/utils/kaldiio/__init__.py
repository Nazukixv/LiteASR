# flake8: noqa F401
import pkg_resources

from liteasr.utils.kaldiio.matio import load_ark
from liteasr.utils.kaldiio.matio import load_mat
from liteasr.utils.kaldiio.matio import load_scp
from liteasr.utils.kaldiio.matio import load_scp_sequential
from liteasr.utils.kaldiio.matio import load_wav_scp
from liteasr.utils.kaldiio.matio import save_ark
from liteasr.utils.kaldiio.matio import save_mat
from liteasr.utils.kaldiio.highlevel import ReadHelper
from liteasr.utils.kaldiio.highlevel import WriteHelper
from liteasr.utils.kaldiio.utils import open_like_kaldi
from liteasr.utils.kaldiio.utils import parse_specifier

try:
    __version__ = pkg_resources.get_distribution("kaldiio").version
except Exception:
    __version__ = None
del pkg_resources
