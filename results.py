from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .model import *


def _timestr(f):
    f = float(f)
    return str(int(f)) if f == int(f) else ("%g" % f)


def _trunc(label):
    return label if len(label) <= 12 else label[:10] + "-" + label[-1]
