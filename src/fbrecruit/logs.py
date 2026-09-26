import sys


class Tee:
    """Everything to the full log; lines printed through key() reach the summary log as well."""

    def __init__(self, full, brief):
        self.full = full
        self.brief = brief
        self.echo = False

    def write(self, text):
        self.full.write(text)
        if self.echo:
            self.brief.write(text)
        self.flush()

    def flush(self):
        self.full.flush()
        self.brief.flush()


def key(*args, **kwargs):
    """Print to the full log and, when one is open, to the summary log as well."""
    tee = sys.stdout if isinstance(sys.stdout, Tee) else None
    if tee is not None:
        tee.echo = True
    print(*args, **kwargs)
    if tee is not None:
        tee.echo = False


def show(frame, decimals=6):
    return frame.to_string(float_format=lambda v: f"{v:.{decimals}f}")
