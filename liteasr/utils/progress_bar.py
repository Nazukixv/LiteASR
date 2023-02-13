"""Progress bar."""

import sys
import time

BLOCK = ["", "╸", "━"]


class ProgressBar(object):
    def __init__(
        self,
        total: int,
        length: int = 50,
        title: str = "",
    ) -> None:
        self.total = total
        self.length = length
        self.title = title
        self.start = time.process_time()
        self.now = self.start

    def update(self, num):
        new_now = time.process_time()
        if (new_now - self.now) >= 0.01 or num == self.total:
            self.now = new_now
            self._update(num)

    def _update(self, num):
        time_consumption = self.now - self.start
        time_left = time_consumption * (self.total - num) / num

        bar = "{0:>5.0%} [{1}] ({2}/{3}) {4} [{5} eta {6}]".format(
            num / self.total,
            ProgressBar.format_progress(num, self.total, self.length),
            num,
            self.total,
            self.title,
            ProgressBar.format_interval(int(time_consumption)),
            ProgressBar.format_interval(int(time_left)),
        )

        sys.stdout.write("\r" + bar)
        if num == self.total:
            sys.stdout.write("\n")

    @staticmethod
    def format_progress(num, total, length):
        unit = total / length
        full, part = divmod(num, unit)
        full = int(full)
        part = int(part / unit * 2)
        return "{0}{1}{2}".format(
            full * BLOCK[2],
            BLOCK[part],
            (length - full - int(part != 0)) * " ",
        )

    @staticmethod
    def format_interval(t):
        """
        Formats a number of seconds as a clock time, [H:]MM:SS

        Parameters
        ----------
        t  : int
            Number of seconds.

        Returns
        -------
        out  : str
            [H:]MM:SS
        """
        mins, s = divmod(int(t), 60)
        h, m = divmod(mins, 60)
        return "{0:02d}:{1:02d}:{2:02d}".format(h, m, s)
