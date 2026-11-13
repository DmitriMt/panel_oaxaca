from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects
from patsy import dmatrix

__all__ = [
    "PanelOaxacaResults",
    "PanelRegressionResults",
    "calc_means",
    "calc_coefficients",
    "detail_names",
    "get_model_method",
    "_to_list",
    "_refmat",
]

_TERM_RE = re.compile(r"C\(([^,)]+)(?:,[^)]*)?\)\[T\.([^\]]+)\]")
_CALL_RE = re.compile(r"C\(([^,)]+)(?:,[^)]*)?\)")

# Regexes matching patsy factor prefixes
_FACT_PREFIX = re.compile(r"^(i|ib|mi|b)\.(.+)$", re.S)
_CLIST_PREFIX = re.compile(r"^(c|i)\.\((.*)\)$", re.S)

# Patsy parameter name patterns
_PTS_C = re.compile(r"^C\(([^\(\)]+)\)$")

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


CHANGE_COMP_LABELS = {
    "interventionist": [
        ("delta_E", "Endowments"),
        ("delta_C", "Coefficients"),
        ("delta_CE", "Interactions"),
    ],
    "hartmann": [
        ("delta_E", "Endowments"),
        ("delta_C", "Coefficients"),
        ("delta_CE", "Interactions"),
    ],
    "ssm": [
        ("delta_E", "Endowments"),
        ("delta_C", "Coefficients"),
        ("delta_CE", "Interactions"),
    ],
    "kim": [
        ("kim_D1m", "D1"),
        ("kim_D2m", "D2"),
        ("kim_D3m", "D3"),
        ("kim_D4m", "D4"),
        ("kim_D5m", "D5"),
    ],
    "smithwelch": [("sw_1", "i"), ("sw_2", "ii"), ("sw_3", "iii"), ("sw_4", "iv")],
    "mpjd": [
        ("endowment_pure", "E-pure"),
        ("endowment_price", "E-price"),
        ("endowment_total", "E-total"),
        ("coefficient_pure", "U-pure"),
        ("coefficient_price", "U-price"),
        ("coefficient_total", "U-total"),
    ],
    "wellington": [("wl_1", "w1"), ("wl_2", "w2")],
    "interventionist_twofold": [
        ("delta_endowment", "Endowments"),
        ("delta_coefficient", "Unexplained"),
    ],
}


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
    varlist: list,
    model: PanelRegressionResults,
    groupvar: str,
    groupcat: str,
    time_var: str,
    times: list,
    timebandwidth: float | None = None,
    weights=None,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec = model.spec(df)

    if timebandwidth is None:
        # Stata: 0.1 when the time variable enters the model as a factor
        # (i.time), 0.5 (with a warning) when it is continuous in the model.
        timebandwidth = 0.1 if time_var in spec.factor_levels else 0.5

    if groupcat is None:
        groupcat = spec.factor_levels[groupvar]

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
                            tcol=time_var,
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
                        tcol=time_var,
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
    model: PanelRegressionResults,
    basemodel: PanelRegressionResults,
    varlist: List[str],
    groupvar: str,
    groupcat: str,
    time_var: str,
    times: List[int],
    atvars: bool,
    meansA: np.ndarray,
    meansB: np.ndarray,
    timebandwidth: float | None = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
]:
    spec = model.spec(df)
    if timebandwidth is None:
        # Stata: 0.1 when the time variable enters the model as a factor
        # (i.time), 0.5 (with a warning) when it is continuous in the model.
        timebandwidth = 0.1 if time_var in spec.factor_levels else 0.5

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

    def _coefs(cat: str, model, mg: np.ndarray) -> np.ndarray:
        out = np.zeros((T, n + 1))
        for j, t in enumerate(times):
            point = {time_var: t, groupvar: cat}
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
                        & (df[time_var] > t - timebandwidth)
                        & (df[time_var] <= t + timebandwidth)
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
                    out[j, i] = model.partial(df=df, var=raw, point=point, level=None)
                else:
                    out[j, i] = model.partial(df=df, var=raw, point=point, level=level)
            p0 = {time_var: t, groupvar: cat}
            p0.update(at_means)
            for raw, level in det:
                p0[raw] = spec.references[raw] if level is not None else 0.0
            out[j, n] = model.predict_at(point=p0, df=df)
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


def _to_list(x) -> List[str]:
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


def _unit_stats(df: pd.DataFrame, ivar: str) -> Tuple[np.ndarray, np.ndarray]:
    codes, uniques = pd.factorize(df[ivar].to_numpy())
    cnt = np.bincount(codes).astype(float)
    return codes, cnt


def re_mean(
    reffects: np.ndarray,
    df: pd.DataFrame,
    groupvar: str,
    time_var: str,
    cat: str,
    time: int,
    bandwidth: float,
    touse=None,
) -> float:
    if reffects is None or groupvar is None:
        return 0.0

    m = (df[groupvar] == cat) & (
        (df[time_var] > time - bandwidth) & (df[time_var] <= time + bandwidth)
    )
    if touse is not None:
        m &= np.asarray(touse)

    vals = np.asarray(reffects)[m.to_numpy()]
    return float(np.mean(vals)) if len(vals) else 0.0


def _refmat(
    df: pd.DataFrame,
    _id_var: str,
    _time_var: str,
    model: PanelRegressionResults,
    basemodel: PanelRegressionResults,
    groupcat: str,
    times: List[int],
    bandwidth: float,
) -> np.ndarray:
    flat = df.reset_index() if isinstance(df.index, pd.MultiIndex) else df
    reffects = model.reffects(df=flat)

    T = len(times)
    g0, g1 = groupcat
    out = np.zeros((2, T))
    for i, m in enumerate((basemodel if basemodel is not None else model, model)):
        if get_model_method(m) in ("re", "mixed") and reffects is not None:
            for t in range(T):
                out[i, t] = re_mean(
                    reffects=reffects,
                    df=df,
                    groupvar=_id_var,
                    cat=g0,
                    time_var=_time_var,
                    time=times[t],
                    bandwidth=bandwidth,
                ) - re_mean(
                    reffects=reffects,
                    df=df,
                    groupvar=_id_var,
                    cat=g1,
                    time_var=_time_var,
                    time=times[t],
                    bandwidth=bandwidth,
                )
    return out


def _colname(detail_name: str) -> str:
    if "." in detail_name:
        lv, var = detail_name.split(".", 1)
        return f"{var}_{lv}"
    return detail_name


def _leaf_keys(ch: dict, cname: str) -> list:
    if cname == "mpjd":
        return [k for k in ("endowment_total", "coefficient_total") if k in ch]
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
# Forumla scpecification handler #
#                                #
##################################


def _split_patsy_products(rhs: str) -> List[str]:
    out, depth, cur = [], 0, ""
    for ch in rhs:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "+" and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return [p for p in out if p and p not in ("1", "const", "Intercept")]


def _split_terms(rhs: str) -> List[str]:
    terms, depth, cur = [], 0, ""
    for ch in rhs:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "+" and depth == 0:
            terms.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        terms.append(cur.strip())
    return [t for t in terms if t]


def detail_names(varlist: List[str], spec: SimpleNamespace) -> List[str]:
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


##################################
#                                #
#     Formula specification      #
#                                #
##################################


@dataclass(frozen=True)
class Factor:
    kind: str
    var: str
    level: object = None

    @property
    def is_ref(self) -> bool:
        return self.kind == "f" and self.level == "REF"

    def label(self) -> str:
        if self.kind == "c":
            return f"c.{self.var}"
        lvl = "bn" if self.level == "REF" else str(self.level)
        return f"{lvl}.{self.var}"

    def value_at(self, point: dict) -> float:
        if self.kind == "c":
            return float(point.get(self.var, 0.0))
        v = point.get(self.var)
        if v is None:
            return 0.0
        return 1.0 if v == self.level else 0.0


@dataclass(frozen=True)
class Term:
    factors: tuple
    name: str = ""

    def label(self) -> str:
        if self.name:
            return self.name
        if not self.factors:
            return "_cons"
        return "+".join(f.label() for f in self.factors)

    def contains_var(self, var: str) -> bool:
        return any(f.var == var for f in self.factors)

    def has_dummy(self, var: str, level) -> bool:
        target = "REF" if level == "REF" else level
        return any(
            f.kind == "f"
            and f.var == var
            and (f.level if f.level != "REF" else "REF") == target
            for f in self.factors
        )

    def value_at(self, point: dict) -> float:
        v = 1.0
        for f in self.factors:
            v *= f.value_at(point)
        return v

    def factor_value_at(self, point: dict, var: str, level=None) -> float:
        v = 1.0
        for f in self.factors:
            if f.var == var and (
                level is None or (f.level if f.level != "REF" else "REF") == level
            ):
                continue
            v *= f.value_at(point)
        return v


##################################
#                                #
#  Panel Regression results      #
#                                #
##################################


class PanelRegressionResults:
    def __init__(self, result):
        object.__setattr__(self, "_result", result)

    def __getattr__(self, name):
        return getattr(self._result, name)

    def __setattr__(self, name, value):
        if name == "_result" or not hasattr(self._result, name):
            object.__setattr__(self, name, value)
        else:
            setattr(self._result, name, value)

    def __repr__(self):
        return f"{type(self).__name__}({type(self._result).__name__})"

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self._result)))

    @property
    def id_var(self) -> str:
        return self.model.exog.index.names[0]

    @property
    def time_var(self) -> str:
        return self.model.exog.index.names[1]

    def spec(self, df: pd.DataFrame) -> SimpleNamespace:
        """Build an expanded spec (factor_levels + all-level terms + zero‑padded params)

        Uses patsy ``dmatrix`` for factor levels/references, then generates every
        factor‑level combination (including reference dummies) exactly as the
        Stata would.
        """
        rhs = self.model.formula.split("~")[1].strip()
        di = dmatrix(rhs, df, return_type="dataframe").design_info

        # TODO rework clone mapping to more suitable time_var
        _clone_map = {}
        if hasattr(self.model, "exog") and self.model.exog is not None:
            idx_names = list(self.model.exog.index.names or [])
            for cn in idx_names:
                if cn and cn in df.columns:
                    _clone_map[cn] = cn
                if cn and f"{cn}_clone" in df.columns:
                    _clone_map[f"{cn}_clone"] = cn

        factor_levels: dict = {}
        references: dict = {}
        for factor, info in di.factor_infos.items():
            if info.type != "categorical":
                continue
            m = _CALL_RE.match(factor.code)
            var = m.group(1) if m else factor.code
            levels = list(info.categories)
            ref = levels[0]
            mapped = _clone_map.get(var, var)
            references[mapped] = ref
            factor_levels[mapped] = levels

        numerics: list = []
        seen_n: set = set()
        for p in map(str, self.params.index):
            p = p.strip()
            if p in ("Intercept", "const", "_cons"):
                continue
            for term in p.split(":"):
                term = term.strip()
                if _TERM_RE.fullmatch(term):
                    continue
                mapped_term = _clone_map.get(term, term)
                if mapped_term not in seen_n:
                    seen_n.add(mapped_term)
                    numerics.append(mapped_term)

        def _mapvar(v):
            v = v.strip()
            return _clone_map.get(v, v)

        products: list = []
        for prod in _split_patsy_products(rhs):
            raw_toks = [t.strip() for t in prod.split("*") if t.strip()]

            # grouped expressions expantion (exp + exp2) into multiple products
            grouped_indices = [
                i
                for i, tok in enumerate(raw_toks)
                if tok.startswith("(") and tok.endswith(")")
            ]

            if grouped_indices:
                # For each group, expand into one sub-product per inner term
                # Then the outer product becomes a sum of these sub-products
                expanded_products = [""]
                for i, tok in enumerate(raw_toks):
                    if i in grouped_indices:
                        inner = tok[1:-1].strip()
                        inner_terms = [s.strip() for s in inner.split("+") if s.strip()]
                        new_expanded = []
                        for ep in expanded_products:
                            for it in inner_terms:
                                prefix = " * " if ep else ""
                                new_expanded.append(ep + prefix + it)
                        expanded_products = new_expanded
                    else:
                        prefix = " * " if expanded_products[0] else ""
                        expanded_products = [
                            ep + prefix + tok for ep in expanded_products
                        ]

                for ep in expanded_products:
                    toks = []
                    for tok in [t.strip() for t in ep.split(" * ") if t.strip()]:
                        m = _PTS_C.match(tok)
                        if m:
                            toks.append(f"i.{_mapvar(m.group(1))}")
                        elif tok not in ("1", "const", "Intercept"):
                            toks.append(f"c.{_mapvar(tok)}")
                    if toks:
                        products.append("##".join(toks))
            else:
                toks = []
                for tok in raw_toks:
                    m = _PTS_C.match(tok)
                    if m:
                        toks.append(f"i.{_mapvar(m.group(1))}")
                    elif tok not in ("1", "const", "Intercept"):
                        toks.append(f"c.{_mapvar(tok)}")
                if toks:
                    products.append("##".join(toks))
        if not products:
            products = ["1"]  # intercept-only fallback

        stata_rhs = " + ".join(products)

        def _unique_levels_from_spec(var):
            return factor_levels[var]

        def _expand_cross(tok):
            tok = tok.strip()
            if "##" in tok or "*" in tok:
                sep = "##" if "##" in tok else "*"
                names = [n.strip() for n in tok.split(sep)]
                out = []
                for k in range(1, len(names) + 1):
                    for combo in itertools.combinations(names, k):
                        out.append(":".join(combo))
                return out
            return [tok]

        term_map: dict = {}
        term_map[()] = ((), "_cons")

        for tok in _split_terms(stata_rhs):
            for cross_prod in _expand_cross(tok):
                for prod in [cross_prod]:
                    combos = [()]
                    for part in (x.strip() for x in prod.split(":") if x):
                        m = _FACT_PREFIX.match(part)
                        if m and m.group(2).strip() in factor_levels:
                            var = m.group(2).strip()
                            f_list = [
                                Factor(
                                    "f", var, "REF" if lev == references[var] else lev
                                )
                                for lev in factor_levels[var]
                            ]
                        elif part.startswith("c."):
                            var = part[2:]
                            f_list = [Factor("c", var)]
                        elif part in factor_levels:
                            var = part
                            f_list = [
                                Factor(
                                    "f", var, "REF" if lev == references[var] else lev
                                )
                                for lev in factor_levels[var]
                            ]
                        else:
                            f_list = [Factor("c", part)]
                        combos = [c + (f,) for c in combos for f in f_list]

                    for fs in combos:
                        key = tuple(
                            sorted(
                                (
                                    (
                                        f.kind,
                                        f.var,
                                        "REF" if f.level == "REF" else f.level,
                                    )
                                    for f in fs
                                ),
                                key=lambda t: (t[0], t[1], str(t[2])),
                            )
                        )
                        name = "+".join(f.label() for f in fs) if fs else "_cons"
                        if key not in term_map:
                            term_map[key] = (fs, name)

        full_terms = [Term(factors=fs, name=n) for fs, n in term_map.values()]

        def _term_key(t):
            return tuple(
                sorted(
                    (
                        (f.kind, f.var, "REF" if f.level == "REF" else f.level)
                        for f in t.factors
                    ),
                    key=lambda z: (z[0], z[1], str(z[2])),
                )
            )

        key_to_idx = {_term_key(t): j for j, t in enumerate(full_terms)}

        full_params = np.zeros(len(full_terms), dtype=float)

        for raw in map(str, self.params.index):
            name = raw.strip()
            if name in ("Intercept", "const", "_cons"):
                j = key_to_idx.get(())
                if j is not None:
                    full_params[j] = float(self.params[raw])
                continue

            factors = []
            for part in (p.strip() for p in name.split(":")):
                m = _TERM_RE.fullmatch(part)
                if m:
                    var = m.group(1)
                    var = _clone_map.get(var, var)
                    if var in factor_levels:
                        lvl_str = m.group(2)
                        if lvl_str.startswith("T."):
                            lvl_str = lvl_str[2:]
                        actual = lvl_str
                        for lv in factor_levels[var]:
                            if str(lv) == str(lvl_str):
                                actual = lv
                                break
                        ref = references[var]
                        factors.append(
                            Factor(
                                "f", var, "REF" if str(actual) == str(ref) else actual
                            )
                        )
                    else:
                        factors.append(Factor("c", var))
                else:
                    mapped = _clone_map.get(part, part)
                    factors.append(Factor("c", mapped))

            k = _term_key(Term(factors=tuple(factors)))
            j = key_to_idx.get(k)
            if j is not None:
                full_params[j] = float(self.params[raw])

        return SimpleNamespace(
            factor_levels=factor_levels,
            references=references,
            numeric=numerics,
            terms=full_terms,
            params=full_params,
        )

    def terms(self, df: pd.DataFrame) -> list:
        return self.spec(df).terms

    @property
    def method(self) -> str:
        method = "ols"
        if isinstance(self.model, RandomEffects):
            method = "re"
        elif isinstance(self.model, PanelOLS):
            method = "fe" if getattr(self.model, "entity_effects", False) else "ols"
        return method

    @property
    def dep_var(self) -> str:
        return self.model.formula.split("~")[0].strip()

    def design_rows(self, df: pd.DataFrame) -> np.ndarray:
        spec = self.spec(df)
        terms = spec.terms
        n = len(df)
        X = np.zeros((n, len(terms)), dtype=float)
        for j, t in enumerate(terms):
            v = np.ones(n, dtype=float)
            for f in t.factors:
                if f.kind == "c":
                    v = v * df[f.var].to_numpy(dtype=float)
                else:
                    v = v * np.where(df[f.var].to_numpy() == f.level, 1.0, 0.0)
            X[:, j] = v
        return X

    def partial(self, df: pd.DataFrame, var: str, point: dict, level=None) -> float:
        spec = self.spec(df)
        terms = spec.terms
        params = spec.params

        if var in spec.factor_levels:
            if level is None:
                raise ValueError(f"factor variable {var!r}: partial() needs level=")
            for lv in spec.factor_levels[var]:
                if str(lv) == str(level):
                    level = lv
                    break

            acc = 0.0
            for t, b in zip(terms, params):
                if t.has_dummy(var, level):
                    acc += b * t.factor_value_at(point, var, level=level)
            return float(acc)

        acc = 0.0
        for t, b in zip(terms, params):
            if t.contains_var(var):
                acc += b * t.factor_value_at(point, var)
        return float(acc)

    def predict_at(self, df: pd.DataFrame, point: dict) -> float:
        spec = self.spec(df)
        return float(
            np.dot(spec.params, np.array([t.value_at(point) for t in spec.terms]))
        )

    def reffects(self, df: pd.DataFrame) -> np.ndarray:
        spec = self.spec(df)
        params = spec.params

        reffects = None
        if self.method == "re":
            vd = self.variance_decomposition
            su2 = float(vd["Effects"])
            se2 = float(vd["Residual"])
            rho2 = su2 / (su2 + se2) if su2 > 0 else 0.0
            codes, cnt = _unit_stats(df, self.id_var)
            y = df[self.dep_var].to_numpy(dtype=float)
            ybar = np.zeros(len(cnt))
            np.add.at(ybar, codes, y)
            ybar /= cnt
            X = self.design_rows(df)
            Xbar = np.zeros((len(cnt), X.shape[1]))
            for j in range(X.shape[1]):
                np.add.at(Xbar[:, j], codes, X[:, j])
            Xbar /= cnt[:, None]
            uib = ybar - Xbar @ params
            c = cnt * rho2 / np.maximum(1e-9, 1.0 - rho2 + cnt * rho2)
            reffects = (c * uib)[codes]
        elif self.method == "fe":
            codes, cnt = _unit_stats(df, self.id_var)
            y = df[self.dep_var].to_numpy(dtype=float)
            ybar = np.zeros(len(cnt))
            np.add.at(ybar, codes, y)
            ybar /= cnt
            X = self.design_rows(df)
            Xbar = np.zeros((len(cnt), X.shape[1]))
            for j in range(X.shape[1]):
                np.add.at(Xbar[:, j], codes, X[:, j])
            Xbar /= cnt[:, None]
            reffects = (ybar - Xbar @ params)[codes]
        return reffects


##################################
#                                #
#  Panel Oaxaca-Blinder results  #
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
                continue
            m = M.shape[0]
            detail_n = m - 2
            if detail_n < n:
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

    @property
    def resultsdata(self):
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
        y = self.model.dep_var
        df, gv, tv = self.df, self.groupvar, self.time_var
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
        out.append(" " * 31 + self.time_var)
        out.append(" " * 30 + "".join("%11s" % _timestr(t) for t in self.times))
        out.append(sep)

    def summary(self):
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
                if key in leaves:
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

    def _latex_val(self, v, nd=3):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return "."
        if not np.isfinite(fv):
            return "."
        return f"{fv:.{nd}f}"

    def _latex_row(self, label, values, nd=3):
        cells = " & ".join(self._latex_val(v, nd) for v in values)
        return rf"{label} & {cells} \\"

    def _latex_header(self, out):
        T = len(self.times)
        cols = "l" + "c" * T
        out.append(r"\begin{tabular}{" + cols + "}")
        out.append(r"\toprule")
        time_header = " & ".join(str(_timestr(t)) for t in self.times)
        out.append(
            r" & \multicolumn{" + str(T) + r"}{c}{" + str(self.time_var) + r"} \\"
        )
        out.append(r"\cmidrule{2-" + str(T + 1) + "}")
        out.append(r" & " + time_header + r" \\")
        out.append(r"\midrule")

    def _latex_footer(self, out):
        out.append(r"\bottomrule")
        out.append(r"\end{tabular}")
        out.append(r"")

    def summary_as_latex(self, nd=3):
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
            out.append(r"\begin{table}[htbp]")
            out.append(r"\centering")
            out.append(r"\caption{Decomposition of Levels}")
            self._latex_header(out)
            out.append(r"\textit{Level} \\")
            out.append(self._latex_row(r"\quad non-parametric", gap, nd))
            out.append(r"\midrule")
            out.append(r"\textit{Decomp} \\")
            parts = (
                [
                    ("endowment", "Endowments"),
                    ("coefficient", "Coefficients"),
                    ("coefficient_endowment_interaction", "Interaction"),
                ]
                if "coefficient_endowment_interaction" in lv
                else [("explained", "Explained"), ("unexplained", "Unexplained")]
            )
            for key, lbl in parts:
                out.append(
                    self._latex_row(r"\quad " + lbl, np.asarray(lv[key], float)[-1], nd)
                )
            re_row = np.atleast_1d(np.asarray(lv.get("RE", np.zeros(T)), float))[0]
            total = np.atleast_1d(np.asarray(lv["Total"], float))[0]
            if reflag:
                out.append(self._latex_row(r"\quad " + reflag, re_row, nd))
            out.append(self._latex_row(r"\textit{Total}", total, nd))
            out.append(r"\midrule")
            out.append(r"\textit{Decomp \%} \\")
            if reflag:
                with np.errstate(invalid="ignore", divide="ignore"):
                    pre = np.where(total != 0, re_row / total * 100.0, np.nan)
            for key, lbl in parts:
                out.append(
                    self._latex_row(
                        r"\quad " + lbl, np.asarray(lv["perc_" + key], float)[-1], nd
                    )
                )
            if reflag:
                out.append(self._latex_row(r"\quad " + reflag, pre, nd))
            ptot = np.zeros(T)
            for key, _ in parts:
                ptot = ptot + np.asarray(lv["perc_" + key], float)[-1]
            if reflag:
                ptot = ptot + pre
            out.append(self._latex_row(r"\textit{Total \%}", ptot, nd))
            self._latex_footer(out)
            if re_note:
                out.append(r"\small{" + re_note + "}")
            out.append(r"\end{table}")
            out.append("")

        for cname, ch in self.change.items():
            comppairs = CHANGE_COMP_LABELS.get(
                cname, [(k, k.replace("_", " ")) for k in _leaf_keys(ch, cname)]
            )
            leaves = _leaf_keys(ch, cname)
            out.append(r"\begin{table}[htbp]")
            out.append(r"\centering")
            out.append(
                r"\caption{Decomposition of Change --- "
                + cname.replace("_", " ").title()
                + "}"
            )
            self._latex_header(out)
            out.append(r"\textit{Change} \\")
            out.append(self._latex_row(r"\quad non-parametric", change_gap, nd))
            out.append(r"\midrule")
            out.append(r"\textit{Decomp} \\")
            for key, lbl in comppairs:
                if key not in ch:
                    continue
                out.append(
                    self._latex_row(r"\quad " + lbl, np.asarray(ch[key], float)[-1], nd)
                )
            dref = (
                np.atleast_1d(np.asarray(ch.get("RE", np.zeros(T)), float))[0]
                if reflag
                else None
            )
            if reflag:
                out.append(self._latex_row(r"\quad " + reflag, dref, nd))
            tot = sum(np.asarray(ch[k], float)[-1] for k in leaves)
            if reflag:
                tot = tot + dref
            out.append(self._latex_row(r"\textit{Total}", tot, nd))
            out.append(r"\midrule")
            out.append(r"\textit{Decomp \%} \\")
            ptot = np.zeros(T)
            for key, lbl in comppairs:
                if key not in ch:
                    continue
                prow = np.asarray(ch["perc_" + key], float)[-1]
                out.append(self._latex_row(r"\quad " + lbl, prow, nd))
                if key in leaves:
                    ptot = ptot + prow
            if reflag:
                with np.errstate(invalid="ignore", divide="ignore"):
                    pre_c = np.where(tot != 0, dref / tot * 100.0, np.nan)
                out.append(self._latex_row(r"\quad " + reflag, pre_c, nd))
                ptot = ptot + pre_c
            out.append(self._latex_row(r"\textit{Total \%}", ptot, nd))
            self._latex_footer(out)
            if re_note:
                out.append(r"\small{" + re_note + "}")
            cite = CITATIONS.get(cname)
            if cite:
                out.append(r"\small{\textit{Source:} " + cite + "}")
            out.append(r"\end{table}")
            out.append("")

        return "\n".join(out)

    def __repr__(self):
        bits = [f"xtoaxaca result: {self.n_detail} variables"]
        if self.level is not None:
            bits.append("levels")
        if self.change:
            bits.append("change(" + ",".join(self.change) + ")")
        return " ".join(bits)
