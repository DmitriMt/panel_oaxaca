from __future__ import annotations

import re
import warnings
from typing import Any, List

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects

__all__ = ["PanelRegressionModel", "_fit"]


##################################
#                                #
#       OLS RESULTS WRAPPER      #
#                                #
##################################


def _fmt(v, w=10, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return f"{'':>{w}}"
    return f"{v:>{w}.{nd}f}"


# class RegressionSummary:
#     def __init__(self, model):
#         self.model = model


##########################
#                        #
#       OLS WRAPPER      #
#                        #
##########################


class PanelRegressionModel:
    """
    Linearmodels wrapper that adds addtional features not yet covered by linearmodels
    """

    def __init__(
        self,
        res,
    ):
        self.res = res

    @classmethod
    def from_result(cls, res, df):
        flat = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
        idx_names = []
        exog = getattr(res.model, "exog", None)

        if exog is not None and hasattr(exog, "index"):
            idx_names = list(exog.index.names or [])

        method = "ols"
        if isinstance(res.model, RandomEffects):
            method = "re"
        elif isinstance(res.model, PanelOLS):
            method = "fe" if getattr(res.model, "entity_effects", False) else "ols"

        patsy_formula = str(getattr(res.model, "formula", ""))
        lhs, rhs = [s.strip() for s in patsy_formula.split("~", 1)]
