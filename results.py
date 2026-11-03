from __future__ import annotations

import warnings
from typing import Any, List, Union

import numpy as np
import pandas as pd

from .decomp import (
    change_interventionist,
    change_interventionist_twofold,
    change_kim,
    change_mpjd,
    change_smithwelch,
    change_ssm,
    change_wellington,
    levels_2fold,
    levels_3fold,
)
from .utils import (
    PanelOaxacaResults,
    _refmat,
    _to_list,
    calc_coefficients,
    calc_means,
    detail_names,
    get_model_method,
    spec_from_res,
)

CHANGE_ORDER = [
    "interventionist",
    "ssm",
    "kim",
    "smithwelch",
    "mpjd",
    "wellington",
    "interventionist_twofold",
]


def panel_oaxaca(
    varlist: List[str],
    df: pd.DataFrame,
    model: Any,
    groupvar: str,
    groupcat: str,
    timevar: str,
    times: List[int],
    basemodel: Any | None = None,
    timeref: int | None = None,
    timebandwidth: float = 0.1,
    change: str = "kim",
    twofold: str | None = None,
    tfweight: float = 0.5,
    nolevels: bool = False,
    nochange: bool = False,
    detail: bool = False,
    normalize: Union[str, List[str]] | None = None,
    weights: pd.Series | None = None,
    force_sample: bool = False,
    seed: int | None = None,
    rep: int = 100,
    factor_vars: List[str] | None = None,
) -> None:
    _id_var = _time_var = None
    if isinstance(df.index, pd.MultiIndex):
        _id_var = df.index.names[0]
        _time_var = df.index.names[1]
        df = df.reset_index()

    if timeref is None:
        timeref = times[0]
    if timeref not in times:
        raise ValueError(f"timeref({timeref!r}) not in times {times}")

    spec = spec_from_res(model, df)

    r = times.index(timeref)

    _nm, meansA, meansB, sdA, sdB = calc_means(
        df=df,
        model=model,
        groupvar=groupvar,
        groupcat=groupcat,
        timevar=timevar,
        times=times,
        timebandwidth=timebandwidth,
        weights=weights,
    )

    print(f"DEBUG meansA, meansB, sdA, sdB {(_nm, meansA, meansB, sdA, sdB)}")

    atvars = True
    coefsA, coefsB = calc_coefficients(
        df=df,
        model=model,
        basemodel=basemodel,
        varlist=varlist,
        groupvar=groupvar,
        groupcat=groupcat,
        timevar=timevar,
        times=times,
        atvars=atvars,
        meansA=meansA,
        meansB=meansB,
        timebandwidth=timebandwidth,
    )
    coefsA, coefsB = np.asarray(coefsA, float), np.asarray(coefsB, float)

    names = detail_names(varlist, spec)
    n = len(names)

    _nlist = list(_to_list(normalize) or [])
    _is_kim = "kim" in [c.lower().strip() for c in (_to_list(change) or [])]
    _nwarn = False
    if _is_kim:
        if not _nlist:
            _nwarn = True
        _nlist = [v for v in varlist if v in spec.factor_levels]
    for nv in _nlist:
        if nv not in varlist:
            raise ValueError(
                f"error ##: {nv!r} is missing from the decomposition "
                "variables varlist. All variables specified in the "
                "normalize option need to be part of the varlist of "
                "decomposition variables."
            )
        if nv not in spec.factor_levels:
            raise ValueError(
                f"error ##: {nv!r} is not a categorical factor variable. "
                "Only categorical factor variables may be specified in "
                "normalize()."
            )
    if _nlist:
        if _nwarn:
            warnings.warn(
                "The decomposition method based on Kim expects all "
                "categorical variables to be normalized. Normalization "
                "will be conducted on all categorical decomposition "
                f"variables: {_nlist}.",
                stacklevel=2,
            )
        _cstart = {}
        c = 0
        for v in varlist:
            _cstart[v] = c
            c += len(spec.factor_levels[v]) if v in spec.factor_levels else 1
        for nv in _nlist:
            i0, i1 = _cstart[nv], _cstart[nv] + len(spec.factor_levels[nv])
            for M in (coefsA, coefsB):
                mu = M[:, i0:i1].mean(axis=1, keepdims=True)  # (T, 1)
                M[:, i0:i1] = M[:, i0:i1] - mu
                M[:, n] = M[:, n] + mu[:, 0]
    T = len(times)
    refmat = _refmat(
        df, _id_var, _time_var, model, basemodel, groupcat, times, timebandwidth
    )
    re = get_model_method(model) in ("re", "mixed")
    refrow = refmat[1] if re else np.zeros(T)

    use_twofold = twofold in ("twofold", "weight", "pooled")

    level_comps = {}
    if not nolevels:
        if use_twofold:
            if twofold == "pooled":
                level_comps = levels_2fold(
                    meansA, meansB, coefsA, coefsB, sdA, sdB, float(tfweight), refrow
                )
            else:
                level_comps = levels_2fold(
                    meansA, meansB, coefsA, coefsB, None, None, float(tfweight), refrow
                )
        else:
            level_comps = levels_3fold(meansA, meansB, coefsA, coefsB, refrow)
        n = meansA.shape[0] - 1
        if re:
            level_comps["RE"] = refrow.reshape(1, -1)
            level_comps["pRE"] = np.full_like(refrow, np.nan).reshape(1, -1)
        print(f"DEBUG level_comps: {level_comps}")

        parts = (
            ["endowment", "coefficient", "coefficient_endowment_interaction"]
            if "coefficient_endowment_interaction" in level_comps
            else ["endowment", "coefficient"]
        )
        tot = sum(level_comps[p][n + 1] for p in parts)
        tot = tot + refrow
        level_comps["Total"] = tot.reshape(1, -1)
        level_comps["pTotal"] = np.full_like(tot, np.nan).reshape(1, -1)

    changes = {}
    if not nochange:
        change_list = _to_list(change) or (
            ["interventionist"] if not nochange else None
        )
        change_list = []
        if change is None and not nochange:
            change_list = ["interventionist"]
        elif nochange:
            change_list = []
        for cname in change_list:
            cname = cname.lower().strip()
            if cname not in CHANGE_ORDER:
                raise ValueError(f"unknown change() type {cname!r}")
            drefrow = refrow - refrow[r] if re else np.zeros(T)
            if cname == "interventionist":
                changes[cname] = change_interventionist(
                    meansA, meansB, coefsA, coefsB, r, refrow
                )
                # Consume the RE change the function already computed (the
                # faithful image of row 2 of Stata's drefmat) instead of
                # duplicating `_dref` here.
                changes[cname]["RE"] = changes[cname].pop("drefmat")
            elif cname == "ssm":
                changes[cname] = change_ssm(meansA, meansB, coefsA, coefsB, r, refrow)
                changes[cname]["RE"] = drefrow.reshape(1, -1)
            elif cname == "kim":
                changes[cname] = change_kim(
                    meansA, meansB, coefsA, coefsB, r, n, refrow
                )
                changes[cname]["RE"] = drefrow.reshape(1, -1)
            elif cname == "smithwelch":
                changes[cname] = change_smithwelch(
                    meansA, meansB, coefsA, coefsB, r, n, refrow
                )
                changes[cname]["RE"] = drefrow.reshape(1, -1)
            elif cname == "mpjd":
                changes[cname] = change_mpjd(
                    meansA, meansB, coefsA, coefsB, r, n, refrow
                )
                changes[cname]["RE"] = drefrow.reshape(1, -1)
            elif cname == "wellington":
                changes[cname] = change_wellington(
                    meansA, meansB, coefsA, coefsB, r, n, refrow
                )
                changes[cname]["RE"] = drefrow.reshape(1, -1)
            elif cname == "interventionist_twofold":
                changes[cname] = change_interventionist_twofold(
                    meansA,
                    meansB,
                    coefsA,
                    coefsB,
                    r,
                    sdA if twofold == "pooled" else None,
                    sdB if twofold == "pooled" else None,
                    float(tfweight),
                    refrow,
                )
                changes[cname]["RE"] = drefrow.reshape(1, -1)
            for k, M in changes[cname].items():
                if k.startswith("perc_"):
                    M = np.asarray(M, float)
                    if M.ndim == 2:
                        M = M.copy()
                        M[:, r] = np.nan
                        changes[cname][k] = M
    res = PanelOaxacaResults(
        varlist=varlist,
        names=names,
        times=times,
        timeref=timeref,
        groupvar=groupvar,
        groupcat=groupcat,
        timevar=timevar,
        level=level_comps if not nolevels else None,
        change=changes,
        meansA=meansA,
        meansB=meansB,
        coefsA=coefsA,
        coefsB=coefsB,
        model=model,
        basemodel=basemodel,
        df=df,
        re=re,
        spec=spec,
        twofold=bool(use_twofold),
        weights=weights,
        groupcat_=groupcat,
        timebandwidth=timebandwidth,
        refe="fe" if model.method == "fe" else ("re" if re else ""),
    )
    res._bs_args = dict(
        varlist=varlist,
        groupvar=groupvar,
        groupcat=groupcat,
        timevar=timevar,
        times=times,
        timeref=timeref,
        timebandwidth=timebandwidth,
        change=change,
        twofold=twofold,
        tfweight=tfweight,
        nolevels=nolevels,
        nochange=nochange,
        normalize=_nlist,
        weights=weights,
        factor_vars=factor_vars,
        rep=rep,
    )
    #     if seed is not None:
    #         from .bootstrap import bootstrap
    #
    #         bootstrap(res, seed=seed, rep=rep)
    return res
