from __future__ import annotations

import re
from types import SimpleNamespace
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import patsy
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects

__all__ = [
    "calc_means",
    "calc_coefficients",
    "detail_names",
    "spec_from_res",
    "get_model_method",
    "_to_list",
    "_refmat",
]

_TERM_RE = re.compile(r"C\(([^,)]+)(?:,[^)]*)?\)\[T\.([^\]]+)\]")
_CALL_RE = re.compile(r"C\(([^,)]+)(?:,[^)]*)?\)")


CITATIONS = {
    "interventionist": (
        "Kr\xf6ger, H., & Hartmann, J. (2020). xtoaxaca - Extending the "
        "Kitagawa-Oaxaca-Blinder Decomposition Approach to Panel Data. "
        "https://doi.org/10.31235/osf.io/egj79"
    ),
    "hartmann": (
        "Kr\xf6ger, H., & Hartmann, J. (2020). xtoaxaca - Extending the "
        "Kitagawa-Oaxaca-Blinder Decomposition Approach to Panel Data. "
        "https://doi.org/10.31235/osf.io/egj79"
    ),
    "interventionist_twofold": (
        "Kr\xf6ger, H., & Hartmann, J. (2020). xtoaxaca - Extending "
        "the Kitagawa-Oaxaca-Blinder Decomposition Approach to "
        "Panel Data. https://doi.org/10.31235/osf.io/egj79"
    ),
    "ssm": (
        "Kim C. Decomposing the Change in the Wage Gap Between White and Black Men "
        "Over Time, 1980-2005: An Extension of the Blinder-Oaxaca Decomposition "
        "Method. Sociol Methods Res 2010; 38: 619\u201351."
    ),
    "kim": (
        "Kim C. Decomposing the Change in the Wage Gap Between White and Black Men "
        "Over Time, 1980-2005: An Extension of the Blinder-Oaxaca Decomposition "
        "Method. Sociol Methods Res 2010; 38: 619\u201351."
    ),
    "smithwelch": (
        "Smith JP, Welch FR. Black economic progress after Myrdal. "
        "J Econ Lit 1989; 27: 519\u201364."
    ),
    "mpjd": (
        "Makepeace G, Paci P, Joshi H, Dolton P. How Unequally Has Equal Pay "
        "Progressed since the 1970s? A Study of Two British Cohorts. "
        "J Hum Resour 1999; 34: 534."
    ),
    "wellington": (
        "Wellington, A. J. (1993). Changes in the Male/Female Wage Gap, "
        "1976-85. The Journal of Human Resources, 28(2), 383."
    ),
}


##################################
#                                #
# Forumla scpecification handler #
#                                #
##################################


def spec_from_res(model, df: pd.DataFrame) -> SimpleNamespace:
    factors, numerics = [], []
    seen_f, seen_n = set(), set()

    for p in map(str, model.params.index):
        p = p.strip()
        if p in ("Intercept", "const"):
            continue
        for term in p.split(":"):
            term = term.strip()
            m = _TERM_RE.fullmatch(term)
            if m:
                v = m.group(1)
                if v not in seen_f:
                    seen_f.add(v)
                    factors.append(v)
            elif term not in seen_n:
                seen_n.add(term)
                numerics.append(term)

    if not factors and not numerics:
        return SimpleNamespace(factor_levels={}, references={}, numeric=[])

    di = patsy.dmatrix(
        model.model.formula.split("~")[1], df, return_type="dataframe"
    ).design_info

    factor_levels, references = {}, {}
    for factor, info in di.factor_infos.items():
        if info.type != "categorical":
            continue
        m = _CALL_RE.match(factor.code)
        var = m.group(1) if m else factor.code
        levels = list(info.categories)
        ref = levels[0]
        references[var] = ref
        factor_levels[var] = [ref] + [l for l in levels if l != ref]

    return SimpleNamespace(
        factor_levels=factor_levels,
        references=references,
        numeric=numerics,
    )


def detail_names(varlist: list, spec) -> List[str]:
    names = []
    for v in varlist:
        if v in spec.factor_levels:
            lv = spec.factor_levels[v]
            ref = spec.references[v]
            ordered = [ref] + [l for l in lv if l != ref]
            for l in ordered:
                names.append(f"{l}.{v}")
        else:
            names.append(v)
    return names


def detail_spec(varlist: List[str], spec) -> List[Tuple[str, str | None]]:
    out = []
    for v in varlist:
        if v in spec.factor_levels:
            ref = spec.references[v]
            ordered = [ref] + [l for l in spec.factor_levels[v] if l != ref]
            for l in ordered:
                out.append((v, l))
        else:
            out.append((v, None))
    return out


def partial(spec, param_names, params, var: str, point: dict, level=None) -> float:
    params = np.asarray(params, dtype=float)

    # ---- factor case -------------------------------------------------------
    if var in spec.factor_levels:
        if level is None:
            raise ValueError(f"factor variable {var!r}: partial() needs level=")
        if level == spec.references[var]:
            return 0.0

        dummy_tag = f"C({var})[T.{level}]"
        acc = 0.0
        for name, beta in zip(map(str, param_names), params):
            if name in ("Intercept", "const"):
                continue
            parts = [p.strip() for p in name.split(":")]
            if dummy_tag not in parts:
                continue

            weight = 1.0
            for part in parts:
                if part == dummy_tag:
                    continue
                m = _TERM_RE.fullmatch(part)
                if m:
                    pvar, plevel = m.group(1), m.group(2)
                    pval = point.get(pvar, spec.references[pvar])
                    weight *= 1.0 if str(pval) == str(plevel) else 0.0
                else:
                    weight *= float(point.get(part, 0.0))
            acc += beta * weight
        return float(acc)

    # ---- continuous case ---------------------------------------------------
    if var not in spec.numeric:
        raise ValueError(f"{var!r} is neither a factor nor a numeric term")

    acc = 0.0
    for name, beta in zip(map(str, param_names), params):
        if name in ("Intercept", "const"):
            continue
        parts = [p.strip() for p in name.split(":")]
        if var not in parts:
            continue
        weight = 1.0
        for part in parts:
            if part == var:
                continue
            m = _TERM_RE.fullmatch(part)
            if m:
                pvar, plevel = m.group(1), m.group(2)
                pval = point.get(pvar, spec.references[pvar])
                weight *= 1.0 if str(pval) == str(plevel) else 0.0
            else:
                weight *= float(point.get(part, 0.0))
        acc += beta * weight
    return float(acc)


def predict_at(spec, param_names, params, point: dict) -> float:
    params = np.asarray(params, dtype=float)
    acc = 0.0

    for name, beta in zip(map(str, param_names), params):
        if name in ("Intercept", "const"):
            acc += beta
            continue
        value = 1.0
        for part in (p.strip() for p in name.split(":")):
            m = _TERM_RE.fullmatch(part)
            if m:
                pvar, plevel = m.group(1), m.group(2)
                pval = point.get(pvar, spec.references[pvar])
                if str(pval) != str(plevel):
                    value = 0.0
                    break
            else:
                value *= float(point.get(part, 0.0))

        acc += beta * value

    return float(acc)


##################################
#                                #
#     Means and coefficients     #
#                                #
##################################


def _mean_var(
    df: pd.DataFrame,
    var: str,
    is_factor: bool,
    group: str,
    cat: str,
    tcol: str,
    tval: float,
    bw: float,
    level: str | None = None,
    wcol: str | None = None,
) -> Tuple[float, float]:
    m = (df[group] == cat) & (df[tcol] > tval - bw) & (df[tcol] <= tval + bw)
    if not m.any():
        return np.nan, np.nan
    if is_factor:
        x = (df[var] == level).astype(float)
    else:
        x = df[var].astype(float)
    xm = x[m].to_numpy(dtype=float)
    if wcol:
        wm = df[wcol].loc[m].to_numpy(dtype=float)
        mean = float(np.sum(xm * wm) / max(1e-300, np.sum(wm)))
        dev = xm - mean
        var = float(np.sum(wm * dev * dev) / max(1e-300, np.sum(wm)))
        return mean, var
    mean = float(xm.mean()) if len(xm) else np.nan
    var = float(xm.var(ddof=1)) if len(xm) > 1 else np.nan
    return mean, var


def calc_means(
    df: pd.DataFrame,
    model,
    groupvar: str,
    groupcat: str,
    timevar: str,
    times: list,
    timebandwidth: float | None = None,
    weights=None,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = spec_from_res(model, df)

    if timebandwidth is None:
        # Stata: 0.1 when the time variable enters the model as a factor
        # (i.time), 0.5 (with a warning) when it is continuous in the model.
        timebandwidth = 0.1 if timevar in spec.factor_levels else 0.5

    if groupcat is None:
        groupcat = spec.factor_levels[groupvar]

    varlist = [
        v
        for v in list(spec.factor_levels) + spec.numeric
        if v not in (groupvar, timevar)
    ]

    times = list(times)
    T = len(times)
    A, B = groupcat[0], groupcat[1]
    wcol = weights if weights is not None and weights in df.columns else None

    names = detail_names(varlist, spec)

    def _rows(cat):
        rows, sdrows = [], []
        for v in varlist:
            if v in spec.factor_levels:
                ref = spec.references[v]
                ordered = [ref] + [l for l in spec.factor_levels[v] if l != ref]
                for l in ordered:
                    row, sdrow = [], []
                    for t in times:
                        m, s = _mean_var(
                            df=df,
                            var=v,
                            is_factor=True,
                            level=l,
                            group=groupvar,
                            cat=cat,
                            tcol=timevar,
                            tval=t,
                            bw=timebandwidth,
                            wcol=wcol,
                        )
                        row.append(m)
                        sdrow.append(s)
                    rows.append(row)
                    sdrows.append(sdrow)
            else:
                row, sdrow = [], []
                for t in times:
                    m, s = _mean_var(
                        df=df,
                        var=v,
                        is_factor=False,
                        level=None,
                        group=groupvar,
                        cat=cat,
                        tcol=timevar,
                        tval=t,
                        bw=timebandwidth,
                        wcol=wcol,
                    )
                    row.append(m)
                    sdrow.append(s)
                rows.append(row)
                sdrows.append(sdrow)
        ones = [1.0] * T
        rows.append(ones)
        sdrows.append(ones)
        return np.array(rows, float), np.array(sdrows, float)

    meansA, sdA = _rows(A)
    meansB, sdB = _rows(B)
    return names, meansA, meansB, sdA, sdB


def calc_coefficients(
    df: pd.DataFrame,
    model,
    basemodel,
    varlist: List[str],
    groupvar: str,
    groupcat: str,
    timevar: str,
    times: List[int],
    atvars: bool,
    meansA: np.ndarray,
    meansB: np.ndarray,
    timebandwidth: float | None = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    spec = spec_from_res(model, df)
    if timebandwidth is None:
        # Stata: 0.1 when the time variable enters the model as a factor
        # (i.time), 0.5 (with a warning) when it is continuous in the model.
        timebandwidth = 0.1 if timevar in spec.factor_levels else 0.5

    times = list(times)
    T = len(times)
    A, B = groupcat[0], groupcat[1]
    det = detail_spec(varlist, spec)
    n = len(det)

    if atvars is True:
        atvars = [v for v in spec.numeric if v not in varlist]
    atvars = list(atvars or [])
    at_means = {v: float(np.nanmean(df[v].astype(float))) for v in atvars}
    mA = meansA[:n]
    mB = meansB[:n]

    def _coefs(cat: str, model_m, mg: np.ndarray) -> np.ndarray:
        out = np.zeros((T, n + 1))
        for j, t in enumerate(times):
            point = {timevar: t, groupvar: cat}
            point.update(at_means)
            # decomp vars at this group's own window means
            seen = set()
            for i, (raw, level) in enumerate(det):
                if raw in seen:
                    continue
                seen.add(raw)
                if level is None:
                    point[raw] = mg[i, j]
                else:
                    m = (
                        (df[groupvar] == cat)
                        & (df[timevar] > t - timebandwidth)
                        & (df[timevar] <= t + timebandwidth)
                    )
                    sub = df.loc[m, raw]
                    if len(sub):
                        mode = (sub.value_counts()).idxmax()
                        # if the mode is NOT the examined level, the
                        # examined level's own dummy is 0 at the at-point
                        # and its coefficient enters via `partial` — so
                        # pin to the mode; examined-level dummies other
                        # than the mode are 0 here as in Stata.
                        point.setdefault(raw, mode)
                    else:
                        point.setdefault(raw, spec.references[raw])
            for i, (raw, level) in enumerate(det):
                if level is None:
                    out[j, i] = partial(
                        spec,
                        model_m.params.index,
                        model_m.params.to_numpy(),
                        raw,
                        point,
                        level=None,
                    )  # model_m.partial(raw, point)
                else:
                    out[j, i] = partial(
                        spec,
                        model_m.params.index,
                        model_m.params.to_numpy(),
                        raw,
                        point,
                        level=level,
                    )  # model_m.partial(raw, point, level=level)
            p0 = {timevar: t, groupvar: cat}
            p0.update(at_means)
            for raw, level in det:
                p0[raw] = spec.references[raw] if level is not None else 0.0
            out[j, n] = predict_at(
                spec, model_m.params.index, model_m.params.to_numpy(), p0
            )  # model_m.predict_at(p0)
        return out

    coefsA = _coefs(A, model, mA)
    coefsB = _coefs(B, model, mB)
    if basemodel is None:
        return coefsA, coefsB
    return coefsA, coefsB, _coefs(A, basemodel, mA), _coefs(B, basemodel, mB)


##################################
#                                #
#         Other utlities         #
#                                #
##################################


def _to_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]


def get_model_method(model_res) -> str:
    method = "ols"
    if isinstance(model_res.model, RandomEffects):
        method = "re"
    elif isinstance(model_res.model, PanelOLS):
        method = "fe" if getattr(model_res.model, "entity_effects", False) else "ols"
    return method


def _unit_stats(df, ivar):
    codes, uniques = pd.factorize(df[ivar].to_numpy())
    cnt = np.bincount(codes).astype(float)
    return codes, cnt


def design_rows(df, param_names):
    param_names = [str(p).strip() for p in param_names]
    n = len(df)
    X = np.zeros((n, len(param_names)), dtype=float)

    for j, name in enumerate(param_names):
        if name in ("Intercept", "const"):
            X[:, j] = 1.0
            continue

        v = np.ones(n, dtype=float)
        for part in (p.strip() for p in name.split(":")):
            m = _TERM_RE.fullmatch(part)
            if m:
                var, level = m.group(1), m.group(2)
                raw = df[var].to_numpy()
                v *= np.array(
                    [1.0 if str(x) == level else 0.0 for x in raw],
                    dtype=float,
                )
            else:
                v *= df[part].to_numpy(dtype=float)
        X[:, j] = v
    return X


def params_from_result(res, spec, param_names=None):
    index = list(map(str, res.params.index))
    if param_names is None:
        param_names = index

    for name in index:
        if name in ["Intercept", "1", "const"]:
            continue
        for part in (p.strip() for p in name.split(":") if p.strip()):
            m = _TERM_RE.fullmatch(part)
            if m:
                var = m.group(1)
                if var not in spec.factor_levels:
                    raise ValueError(
                        f"params_from_result: factor {var!r} in {name!r} "
                        f"is not in spec.factor_levels"
                    )
            else:
                if part not in spec.numeric:
                    raise ValueError(
                        f"params_from_result: numeric term {part!r} in "
                        f"{name!r} is not in spec.numeric"
                    )

    pos = {n: j for j, n in enumerate(param_names)}
    params = np.zeros(len(param_names), dtype=float)

    for name, val in res.params.items():
        name = str(name).strip()
        j = pos.get(name)
        if j is None:
            raise ValueError(
                f"params_from_result: cannot map parameter {name!r} "
                f"onto the supplied param_names"
            )
        params[j] = float(val)

    missing = [
        n
        for j, n in enumerate(param_names)
        if params[j] == 0.0 and n not in res.params.index
    ]
    if missing:
        raise ValueError(f"params_from_result: no coefficient for {missing!r}")
    return params


def re_mean(reffects, df, groupvar, timevar, cat, time, bandwidth, touse=None):
    print(f"DEBUG bandwidth: {bandwidth}")
    if reffects is None or groupvar is None:
        return 0.0

    m = (df[groupvar] == cat) & (
        (df[timevar] > time - bandwidth) & (df[timevar] <= time + bandwidth)
    )
    if touse is not None:
        m &= np.asarray(touse)

    vals = np.asarray(reffects)[m.to_numpy()]
    return float(np.mean(vals)) if len(vals) else 0.0


def _refmat(df, _id_var, _time_var, model, basemodel, groupcat, times, bandwidth):
    print(f"DEBUG bandwidth: {bandwidth}")
    flat = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df

    reffects = None

    spec = spec_from_res(df=df, model=model)

    params = params_from_result(res=model, spec=spec)
    print(f"DEBUG params {params}")
    ivar = list(model.model.exog.index.names)[0]
    odes, cnt = _unit_stats(flat, ivar)
    method = get_model_method(model)
    if method == "re":
        vd = model.variance_decomposition
        su2 = float(vd["Effects"])
        se2 = float(vd["Residual"])
        rho2 = su2 / (su2 + se2) if su2 > 0 else 0.0
        codes, cnt = _unit_stats(flat, ivar)
        y = flat[model.model.formula.split("~")[0].strip()].to_numpy(dtype=float)
        ybar = np.zeros(len(cnt))
        np.add.at(ybar, codes, y)
        ybar /= cnt
        X = design_rows(df, model.params.index.tolist())  # X = spec.design_rows(flat)
        Xbar = np.zeros((len(cnt), X.shape[1]))
        for j in range(X.shape[1]):
            np.add.at(Xbar[:, j], codes, X[:, j])
        Xbar /= cnt[:, None]
        uib = ybar - Xbar @ params
        c = cnt * rho2 / np.maximum(1e-9, 1.0 - rho2 + cnt * rho2)
        reffects = (c * uib)[codes]
    elif method == "fe":
        codes, cnt = _unit_stats(flat, ivar)
        y = flat[spec.depvar].to_numpy(dtype=float)
        ybar = np.zeros(len(cnt))
        np.add.at(ybar, codes, y)
        ybar /= cnt
        X = design_rows(df, model.params.index.tolist())  # X = spec.design_rows(flat)
        Xbar = np.zeros((len(cnt), X.shape[1]))
        for j in range(X.shape[1]):
            np.add.at(Xbar[:, j], codes, X[:, j])
        Xbar /= cnt[:, None]
        reffects = (ybar - Xbar @ params)[codes]

    T = len(times)
    g0, g1 = groupcat
    out = np.zeros((2, T))
    for i, m in enumerate((basemodel if basemodel is not None else model, model)):
        if get_model_method(m) in ("re", "mixed") and reffects is not None:
            for t in range(T):
                # out[i, t] = m.re_mean(g0, times[t], bandwidth) - m.re_mean(
                #     g1, times[t], bandwidth
                # )
                out[i, t] = re_mean(
                    reffects=reffects,
                    df=df,
                    groupvar=_id_var,
                    cat=g0,
                    timevar=_time_var,
                    time=times[t],
                    bandwidth=bandwidth,
                ) - re_mean(
                    reffects=reffects,
                    df=df,
                    groupvar=_id_var,
                    cat=g1,
                    timevar=_time_var,
                    time=times[t],
                    bandwidth=bandwidth,
                )
    return out


def _colname(detail_name: str) -> str:
    if "." in detail_name:
        lv, var = detail_name.split(".", 1)
        return f"{var}_{lv}"
    return detail_name


def _leaf_keys(ch: list, cname: str) -> list:
    if cname == "mpjd":
        return [k for k in ("E_total", "U_total") if k in ch]
    return [
        k for k in ch if not k.startswith("perc_") and k not in ("RE", "coefs_star")
    ]


def _trunc(label: str) -> str:
    return label if len(label) <= 12 else label[:10] + "-" + label[-1]


def _timestr(f: float) -> str:
    f = float(f)
    return str(int(f)) if f == int(f) else ("%g" % f)


##################################
#                                #
#  Panel Oaxaca-Blinder result   #
#                                #
##################################


class PanelOaxacaResults:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def n_detail(self):
        return len(self.names)

    def _block_rows(self, comps, type_name):
        """Rows for one block (level or a change type).

        Returns list of dicts. comps: {name: (n+2,T)} with percentages
        under name 'p'+name. RE rows dropped (carried in `re` col).
        """
        n = self.n_detail
        T = len(self.times)
        rows = []
        comp_names = [
            c
            for c in comps
            if not c.startswith("perc_") and c not in ("RE", "Total", "coefs_star")
        ]
        for cname in comp_names:
            M = comps[cname]
            P = comps.get("perc_" + cname)
            if M is None or M.ndim != 2:
                # kim D1 has shape (1, T): detail rows empty
                continue
            m = M.shape[0]
            detail_n = m - 2
            if detail_n < n:
                # matrix with fewer detail rows (kim D1..): pad
                pass
            for t in range(T):
                vals = {}
                for ci, name in enumerate(self.names):
                    col = _colname(name)
                    vals[col] = M[ci, t] if ci < detail_n else np.nan
                vals["Total"] = M[-1, t]
                vals["time"] = self.times[t]
                vals["Intercept"] = M[n, t] if m == n + 2 else np.nan
                vals["component"] = cname
                vals["percentage"] = "no"
                vals["type"] = type_name
                if self.re:
                    RE = np.asarray(comps["RE"])
                    vals["re"] = RE[0, t] if RE.ndim == 2 else RE[t]
                rows.append(dict(vals))
            if P is not None:
                for t in range(T):
                    vals = {}
                    for ci, name in enumerate(self.names):
                        col = _colname(name)
                        vals[col] = P[ci, t] if ci < detail_n else np.nan
                    vals["Total"] = P[-1, t]
                    vals["time"] = self.times[t]
                    vals["Intercept"] = P[n, t] if m == n + 2 else np.nan
                    vals["component"] = cname
                    vals["percentage"] = "yes"
                    vals["type"] = type_name
                    if self.re:
                        RE = np.asarray(comps["RE"])
                        vals["re"] = RE[0, t] if RE.ndim == 2 else RE[t]
                    rows.append(dict(vals))
        return rows

    def resultsdata(self):
        """Long DataFrame, Stata resultsdata() layout.

        Columns: [detail cols..., Total, time, Intercept, component,
        percentage, type] + `re` if RE model + `bs` if bootstrapped.
        Change blocks first (in requested order), then level block.
        """
        all_rows = []
        for cname, comps in self.change.items():
            all_rows.extend(self._block_rows(comps, cname))
        if self.level is not None:
            all_rows.extend(self._block_rows(self.level, "level"))
        if not all_rows:
            return pd.DataFrame()
        df = pd.DataFrame(all_rows)
        detail_cols = [_colname(x) for x in self.names]
        order = detail_cols + [
            "Total",
            "time",
            "Intercept",
            "component",
            "percentage",
            "type",
        ]
        if self.re:
            order.append("re")
        if "bs" in df.columns:
            order.append("bs")
        return df[order]

    def _fmt_row(self, label, values):
        """One report row: left label + right-aligned 3-decimal value columns
        (missing values printed as a centered dot, Stata style)."""
        cells = []
        for v in values:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                cells.append("%11s" % ".")
                continue
            if not np.isfinite(fv):
                cells.append("%11s" % ".")
            else:
                cells.append("%11.3f" % fv)
        return "%-26s" % _trunc(label) + "".join(cells)

    def _observed_gap(self):
        """Non-parametric group gap per time: windowed (aweight-) mean of the
        dependent variable for group A minus group B, over (t-bw, t+bw].
        This equals the sum of the decomposition components (verified)."""
        y = self.model.spec.depvar
        df, gv, tv = self.df, self.groupvar, self.timevar
        a, b = self.groupcat
        bw = getattr(self, "timebandwidth", 0.1)
        w = getattr(self, "weights", None)
        wcol = w if isinstance(w, str) else None

        def wmean(cat):
            sel = df[gv] == cat
            out = []
            for t in self.times:
                mm = sel & (df[tv] > t - bw) & (df[tv] <= t + bw)
                if not mm.any():
                    out.append(np.nan)
                    continue
                x = df[y].loc[mm].to_numpy(dtype=float)
                if wcol and wcol in df.columns:
                    wt = df[wcol].loc[mm].to_numpy(dtype=float)
                    out.append(float(np.sum(x * wt) / max(1e-300, np.sum(wt))))
                else:
                    out.append(float(np.mean(x)))
            return np.array(out)

        return wmean(a) - wmean(b)

    def _header(self, out, sep):
        out.append(sep)
        out.append(" " * 31 + self.timevar)
        out.append(" " * 30 + "".join("%11s" % _timestr(t) for t in self.times))
        out.append(sep)

    def summary(self):
        """Render the Stata-style decomposition report (mirrors the .ado output:
        'Decomposition of Levels' then 'Decomposition of Change' per change type,
        with Level / Decomp / Decomp % groups and the RE note + citation)."""
        SEP = "\u2500" * 68
        out = []
        gap = self._observed_gap()
        r = self.times.index(self.timeref)
        change_gap = gap - gap[r]
        refe = getattr(self, "refe", "")
        reflag = "RE" if refe == "re" else ("FE" if refe == "fe" else None)
        re_note = {"re": "RE = random effects.", "fe": "FE = fixed effects."}.get(
            refe, ""
        )
        T = len(self.times)

        if self.level is not None:
            lv = self.level
            out.append("Decomposition of Levels")
            out.append("Summary of level decomposition")
            self._header(out, SEP)
            out.append("Level")
            out.append(self._fmt_row("non-parametric", gap))
            out.append(SEP)
            parts = (
                [
                    ("endowment", "Endowments"),
                    ("coefficient", "Coefficients"),
                    ("coefficient_endowment_interaction", "Interaction"),
                ]
                if "coefficient_endowment_interaction" in lv
                else [("explained", "Explained"), ("unexplained", "Unexplained")]
            )
            out.append("Decomp")
            for key, lbl in parts:
                out.append(self._fmt_row(lbl, np.asarray(lv[key], float)[-1]))
            re_row = np.atleast_1d(np.asarray(lv.get("RE", np.zeros(T)), float))[0]
            total = np.atleast_1d(np.asarray(lv["Total"], float))[0]
            if reflag:
                out.append(self._fmt_row(reflag, re_row))
            out.append(self._fmt_row("Total", total))
            out.append(SEP)
            out.append("Decomp %")
            if reflag:
                with np.errstate(invalid="ignore", divide="ignore"):
                    pre = np.where(total != 0, re_row / total * 100.0, np.nan)
            for key, lbl in parts:
                out.append(self._fmt_row(lbl, np.asarray(lv["perc_" + key], float)[-1]))
            if reflag:
                out.append(self._fmt_row(reflag, pre))
            ptot = np.zeros(T)
            for key, _ in parts:
                ptot = ptot + np.asarray(lv["perc_" + key], float)[-1]
            if reflag:
                ptot = ptot + pre
            out.append(self._fmt_row("Total", ptot))
            out.append(SEP)
            if re_note:
                out.append(re_note)
            out.append("")

        for cname, ch in self.change.items():
            out.append("Decomposition of Change")
            out.append("Summary of changes in the outcome")
            self._header(out, SEP)
            out.append("Change")
            out.append(self._fmt_row("non-parametric", change_gap))
            out.append(SEP)
            comppairs = CHANGE_COMP_LABELS.get(
                cname, [(k, k.replace("_", " ")) for k in _leaf_keys(ch, cname)]
            )
            leaves = _leaf_keys(ch, cname)
            out.append("Decomp")
            print(f"DEBUG lbl: {lbl}")
            for key, lbl in comppairs:
                if key not in ch:
                    continue
                out.append(self._fmt_row(lbl, np.asarray(ch[key], float)[-1]))
            dref = (
                np.atleast_1d(np.asarray(ch.get("RE", np.zeros(T)), float))[0]
                if reflag
                else None
            )
            if reflag:
                out.append(self._fmt_row(reflag, dref))
            tot = sum(np.asarray(ch[k], float)[-1] for k in leaves)
            if reflag:
                tot = tot + dref
            out.append(self._fmt_row("Total", tot))
            out.append(SEP)
            out.append("Decomp %")
            ptot = np.zeros(T)
            for key, lbl in comppairs:
                if key not in ch:
                    continue
                prow = np.asarray(ch["perc_" + key], float)[-1]
                out.append(self._fmt_row(lbl, prow))
                ptot = ptot + prow
            if reflag:
                with np.errstate(invalid="ignore", divide="ignore"):
                    pre_c = np.where(tot != 0, dref / tot * 100.0, np.nan)
                out.append(self._fmt_row(reflag, pre_c))
                ptot = ptot + pre_c
            out.append(self._fmt_row("Total", ptot))
            out.append(SEP)
            if re_note:
                out.append(re_note)
            cite = CITATIONS.get(cname)
            if cite:
                out.append(
                    "For an explanation of this change decomposition, please see:"
                )
                out.append(cite)
            out.append("")

        return "\n".join(out)

    def __repr__(self):
        bits = [f"xtoaxaca result: {self.n_detail} variables"]
        if self.level is not None:
            bits.append("levels")
        if self.change:
            bits.append("change(" + ",".join(self.change) + ")")
        return " ".join(bits)
