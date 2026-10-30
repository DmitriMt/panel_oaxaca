from __future__ import annotations

import numpy as np

__all__ = [
    "levels_3fold",
    "levels_2fold",
    "change_interventionist",
    "change_ssm",
    "change_kim",
    "change_smithwelch",
    "change_mpjd",
    "change_wellington",
    "change_interventionist_twofold",
    "COMPONENT_NAMES",
]

COMPONENT_NAMES = {
    "interventioninst": ["delta_E", "delta_C", "delta_CE"],
    "ssm": ["delta_E", "delta_C", "delta_CE"],
    "kim": ["kim_D1m", "kim_D2m", "kim_D3m", "kim_D4m", "kim_D5m"],
    "smithwelch": ["sw_1", "sw_2", "sw_3", "sw_4"],
    "mpjd": [
        "endowment_pure",
        "endowment_price",
        "coefficient_pure",
        "coefficient_price",
    ],
    "wellington": ["wl_1", "wl_2"],
    "interventionist_twofold": ["delta_endowment", "delta_coefficient"],
}


def _total_row(comp: np.ndarray) -> np.ndarray:
    return np.vstack([comp, comp.sum(axis=0)])


def _finilize(comp_n1: np.ndarray, total: np.ndarray):
    out = _total_row(comp_n1)
    with np.errstate(invalid="ignore"):
        pct = out / total * 100.0
    return out, pct


def levels_3fold(meansA, meansB, coefsA, coefsB, refrow):
    mA, mB = np.asarray(meansA, float), np.asarray(meansB, float)
    cA, cB = np.asarray(coefsA, float).T, np.asarray(coefsB, float).T
    refrow = np.asarray(refrow, float)

    endowment_means = (mA - mB) * cB
    coefficient_means = mB * (cA - cB)
    coefficient_endowment_interacton_means = (mA - mB) * (cA - cB)

    total = (
        endowment_means.sum(axis=0)
        + coefficient_means.sum(axis=0)
        + coefficient_endowment_interacton_means.sum(axis=0)
        + refrow
    )
    endowment, perc_endowment = _finilize(endowment_means, total)
    coefficient, perc_coefficient = _finilize(coefficient_means, total)
    coefficient_endowment_interacton, perc_coefficient_endowment_interacton = _finilize(
        coefficient_endowment_interacton_means, total
    )
    return {
        "endowment": endowment,
        "coefficient": coefficient,
        "coefficient_endowment_interaction": coefficient_endowment_interacton,
        "perc_endowment": perc_endowment,
        "perc_coefficient": perc_coefficient,
        "perc_coefficient_endowment_interacton": perc_coefficient_endowment_interacton,
    }


def levels_2fold(
    meansA, meansB, coefsA, coefsB, sdA=None, sdB=None, weight=None, refrow=None
):
    mA, mB = np.asarray(meansA, float), np.asarray(meansB, float)
    cA, cB = np.asarray(coefsA, float).T, np.asarray(coefsB, float).T
    refrow = np.zeros(mA.shape[1]) if refrow is None else np.asarray(refrow, float)
    if sdA is not None and sdB is not None:
        sdA, sdB = np.asarray(sdA, float), np.asarray(sdB, float)
        W = sdA / (sdA + sdB)
    else:
        W = np.full(mA.shape, float(weight if weight is not None else 0.5))

    ones = np.ones(mA.shape)

    coefs_star = W * cA + (ones - W) * cB
    explained_means = (mA - mB) * coefs_star
    unexplained_means = ((ones - W) * mA + W * mB) * (cA + cB)

    total = explained_means.sum(axis=0) + unexplained_means.sum(axis=0) + refrow
    explained, perc_explained = _finilize(explained_means, total)
    unexplained, perc_unexplained = _finilize(unexplained_means, total)
    return {
        "explained": explained,
        "unexplained": unexplained,
        "perc_explained": perc_explained,
        "perc_unexplained": perc_unexplained,
        "coefs_star": coefs_star,
    }


def _dref(refrow, r):
    refrow = np.asarray(refrow, float)
    return refrow - refrow[r]


def change_interventionist(meansA, meansB, coefsA, coefsB, r, refrow):
    mA, mB = np.asarray(meansA, float), np.asarray(meansB, float)
    cA, cB = np.asarray(coefsA, float).T, np.asarray(coefsB, float).T
    mA_r, mB_r = mA[:, r], mB[:, r]
    cA_r, cB_r = cA[:, r], cB[:, r]

    delta_endowment = (mA - mA_r[:, None]) * cA_r[:, None] - (
        mB - mB_r[:, None]
    ) * cB_r[:, None]
    delta_coefficient = (cA - cA_r[:, None]) * mA_r[:, None] - (
        cB - cB_r[:, None]
    ) * mB_r[:, None]
    delta_coefficient_endowment_interacton = (mA - mA_r[:, None]) * (
        cA - cA_r[:, None]
    ) - (mB - mB_r[:, None]) * (cB - cB_r[:, None])

    dref = _dref(refrow, r)
    total = (
        delta_endowment.sum(axis=0)
        + delta_coefficient.sum(axis=0)
        + delta_coefficient_endowment_interacton.sum(axis=0)
        + dref
    )

    delta_E, perc_delta_E = _finilize(delta_endowment, total)
    delta_C, perc_delta_C = _finilize(delta_coefficient, total)
    delta_CE, perc_delta_CE = _finilize(delta_coefficient_endowment_interacton, total)
    return {
        "delta_E": delta_E,
        "delta_C": delta_C,
        "delta_CE": delta_CE,
        "perc_delta_E": perc_delta_E,
        "perc_delta_C": perc_delta_C,
        "perc_delta_CE": perc_delta_CE,
        "drefmat": dref.reshape(1, -1),
    }


def change_ssm(meansA, meansB, coefsA, coefsB, r, refrow):
    mA, mB = np.asarray(meansA, float), np.asarray(meansB, float)
    cA, cB = np.asarray(coefsA, float).T, np.asarray(coefsB, float).T

    endowment_means = (mA - mB) * cB
    coefficient_means = mB * (cA - cB)
    coefficient_endowment_interacton_means = (mA - mB) * (cA - cB)
    delta_endowment = endowment_means - endowment_means[:, r][:, None]
    delta_coefficient = coefficient_means - coefficient_means[:, r][:, None]
    delta_coefficient_endowment_interacton = (
        coefficient_endowment_interacton_means
        - coefficient_endowment_interacton_means[:, r][:, None]
    )

    dref = _dref(refrow, r)
    total = (
        delta_endowment.sum(axis=0)
        + delta_coefficient.sum(axis=0)
        + delta_coefficient_endowment_interacton.sum(axis=0)
        + dref
    )

    delta_E, perc_delta_E = _finilize(delta_endowment, total)
    delta_C, perc_delta_C = _finilize(delta_coefficient, total)
    delta_CE, perc_delta_CE = _finilize(delta_coefficient_endowment_interacton, total)
    return {
        "delta_E": delta_E,
        "delta_C": delta_C,
        "delta_CE": delta_CE,
        "perc_delta_E": perc_delta_E,
        "perc_delta_C": perc_delta_C,
        "perc_delta_CE": perc_delta_CE,
    }


def chage_kim(meansA, meansB, coefsA, coefsB, r, n_decvars, refrow):
    mA, mB = np.asarray(meansA, float), np.asarray(meansB, float)
    cA, cB = np.asarray(coefsA, float).T, np.asarray(coefsB, float).T

    cA_d, cB_d = cA[:n_decvars], cB[:n_decvars]
    mA_d, mB_d = mA[:n_decvars], mB[:n_decvars]
    cA_r, cB_r = cA_d[:, r], cB_d[:, r]
    mA_r, mB_r = mA_d[:, r], mB_d[:, r]

    kim_D1m = np.atleast_2d(
        (cA[n_decvars, :] - cA[n_decvars, r]) - (cB[n_decvars, :] - cB[n_decvars, r])
    )
    kim_D2m = (
        (cA_d - cA_r[:, None])
        - (cB_d - cB_r[:, None]) * (mA_d - mA_r[:, None])
        + (cB_d - cB_r[:, None]) / 4
    )
    kim_D3m = (
        ((cA_d - cA_r[:, None]) / 2 - (cB_d - cB_r[:, None]) / 2)
        * ((mA_d - mA_r[:, None]) + (cB_d - cB_r[:, None]))
        / 2
    )
    kim_D4m = (
        ((mA_d - mA_r[:, None]) - (mB_d - mB_r[:, None]))
        * ((cA_d - cA_r[:, None]) + (cB_d - cB_r[:, None]))
        / 4
    )
    kim_D5m = (
        ((mA_d - mA_r[:, None]) / 2 - (mB_d - mB_r[:, None]) / 2)
        * ((cA_d - cA_r[:, None]) + (cB_d - cB_r[:, None]))
        / 2
    )

    dref = _dref(refrow, r)
    total = (
        kim_D1m.sum(axis=0)
        + kim_D2m.sum(axis=0)
        + kim_D3m.sum(axis=0)
        + kim_D4m.sum(axis=0)
        + kim_D5m.sum(axis=0)
        + dref
    )

    def _with_total(matrix):
        return np.vstack([matrix, np.zeros(matrix.shape[1]), matrix.sum(axis=0)])

    kim_D1m = np.atleast_2d(kim_D1m)
    kim_D2m = _with_total(kim_D2m)
    kim_D3m = _with_total(kim_D3m)
    kim_D4m = _with_total(kim_D4m)
    kim_D5m = _with_total(kim_D5m)

    return {
        "kim_D1m": kim_D1m,
        "kim_D2m": kim_D2m,
        "kim_D3m": kim_D3m,
        "kim_D4m": kim_D4m,
        "kim_D5m": kim_D5m,
        "perc_kim_D1m": kim_D1m / total * 100.0,
        "perc_kim_D2m": kim_D2m / total * 100.0,
        "perc_kim_D3m": kim_D3m / total * 100.0,
        "perc_kim_D4m": kim_D4m / total * 100.0,
        "perc_kim_D5m": kim_D5m / total * 100.0,
    }


def change_smithwelch(meansA, meansB, coefsA, coefsB, r, refrow):
    mA, mB = np.asarray(meansA, float), np.asarray(meansB, float)
    cA, cB = np.asarray(coefsA, float).T, np.asarray(coefsB, float).T

    cA_r, cB_r = cA[:, r], cB[:, r]
    mA_r, mB_r = mA[:, r], mB[:, r]

    sw_1_m = ((mA - mB) - (mA_r[:, None] - mB_r[:, None])) * cB_r[:, None]
    sw_2_m = (mA - mA_r[:, None]) * (cA_r[:, None] - cB_r[:, None])
    sw_3_m = (mA - mB) * (cB - cB_r[:, None])
    sw_4_m = mA * ((cA - cB) - (cA_r[:, None] - cB_r[:, None]))

    dref = _dref(refrow, r)
    total = (
        sw_1_m.sum(axis=0)
        + sw_2_m.sum(axis=0)
        + sw_3_m.sum(axis=0)
        + sw_4_m.sum(axis=0)
        + dref
    )
    sw_1, perc_sw_1 = _finilize(sw_1_m, total)
    sw_2, perc_sw_2 = _finilize(sw_2_m, total)
    sw_3, perc_sw_3 = _finilize(sw_3_m, total)
    sw_4, perc_sw_4 = _finilize(sw_4_m, total)

    return {
        "sw_1": sw_1,
        "sw_2": sw_2,
        "sw_3": sw_3,
        "sw_4": sw_4,
        "perc_sw_1": perc_sw_1,
        "perc_sw_2": perc_sw_2,
        "perc_sw_3": perc_sw_3,
        "perc_sw_4": perc_sw_4,
    }


def change_mpjd(meansA, meansB, coefsA, coefsB, r, refrow):
    mA, mB = np.asarray(meansA, float), np.asarray(meansB, float)
    cA, cB = np.asarray(coefsA, float).T, np.asarray(coefsB, float).T

    cA_r, cB_r = cA[:, r], cB[:, r]
    mA_r, mB_r = mA[:, r], mB[:, r]

    endowment_pure = ((mA - mA_r[:, None]) - (mB - mB_r[:, None])) * cA
    endowment_price = (mA_r[:, None] - mB_r[:, None]) * (cA - cA_r[:, None])
    coefficient_pure = mB * ((cA - cA_r[:, None]) - (cB - cB_r[:, None]))
    coefficient_price = (mB - mB_r[:, None]) * (cA_r[:, None] - cB_r[:, None])

    dref = _dref(refrow, r)
    total = (
        endowment_pure.sum(axis=0)
        + endowment_price.sum(axis=0)
        + coefficient_pure.sum(axis=0)
        + coefficient_price.sum(axis=0)
        + dref
    )

    endowment_pure, perc_endowment_pure = _finilize(endowment_pure, total)
    endowment_price, perc_endowment_price = _finilize(endowment_price, total)
    coefficient_pure, perc_coefficient_pure = _finilize(coefficient_pure, total)
    coefficient_price, perc_coefficient_price = _finilize(coefficient_price, total)
    return {
        "endowment_pure": endowment_pure,
        "endowment_price": endowment_price,
        "coefficient_pure": coefficient_pure,
        "coefficient_price": coefficient_price,
        "coefficient_total": endowment_pure + endowment_price,
        "coefficient_total": coefficient_pure + coefficient_price,
        "perc_endowment_pure": perc_endowment_pure,
        "perc_endowment_price": perc_endowment_price,
        "perc_coefficient_pure": perc_coefficient_pure,
        "perc_coefficient_price": perc_coefficient_price,
        "perc_endowment_total": (perc_endowment_pure + perc_endowment_price)
        / total
        * 100,
        "perc_coefficient_total": (perc_coefficient_pure + perc_coefficient_price)
        / total
        * 100,
    }


def change_wellington(meansA, meansB, coefsA, coefsB, r, refrow):
    mA, mB = np.asarray(meansA, float), np.asarray(meansB, float)
    cA, cB = np.asarray(coefsA, float).T, np.asarray(coefsB, float).T

    cA_r, cB_r = cA[:, r], cB[:, r]
    mA_r, mB_r = mA[:, r], mB[:, r]

    wl_1m = (mA - mA_r[:, None]) * cA - (mB - mB_r[:, None]) * cB
    wl_2m = (cA - cA_r[:, None]) * mA_r[:, None] - (cB - cB_r[:, None]) * mB_r[:, None]

    dref = _dref(refrow, r)
    total = wl_1m.sum(axis=0) + wl_2m.sum(axis=0) + dref
    wl_1, perc_wl_1 = _finilize(wl_1m, total)
    wl_2, perc_wl_2 = _finilize(wl_2m, total)
    return {"wl_1": wl_1, "wl_2": wl_2, "perc_wl_1": perc_wl_1, "perc_wl_2": perc_wl_2}


def change_interventionist_twofold(
    meansA, meansB, coefsA, coefsB, r, sdA=None, sdB=None, weight=None, refrow=None
):
    mA, mB = np.asarray(meansA, float), np.asarray(meansB, float)
    cA, cB = np.asarray(coefsA, float).T, np.asarray(coefsB, float).T

    if sdA is not None and sdB is not None:
        sdA, sdB = np.asarray(sdA, float), np.asarray(sdB, float)
        W = sdA / (sdA + sdB)
    else:
        W = np.full(mA.shape, float(weight if weight is not None else 0.5))

    ones = np.ones(mA.shape)
    cA_r, cB_r = cA[:, r], cB[:, r]
    mA_r, mB_r = mA[:, r], mB[:, r]

    coefs_star = W * cA + (ones - W) * cB
    delta_endowment_means = ((mA - mA_r[:, None]) - (mB - mB_r[:, None])) * coefs_star[
        :, r
    ][:, None]
    delta_coefficient_means = (
        mA * cA
        - mB * cB
        + (
            mB_r[:, None] * W * cB_r[:, None]
            - mA_r[:, None] * (ones - W) * cA_r[:, None]
        )
        + (mB - mB_r[:, None] - mA) * W * cA_r[:, None]
        + (mA_r[:, None] + mB - mA) * (ones - W) * cB_r[:, None]
    )
    dref = _dref(refrow, r)
    total = (
        delta_endowment_means.sum(axis=0) + delta_coefficient_means.sum(axis=0) + dref
    )
    delta_endowment, perc_delta_endowment = _finilize(delta_endowment_means, total)
    delta_coefficient, perc_delta_coefficient = _finilize(
        delta_coefficient_means, total
    )
    return {
        "delta_endowment": delta_endowment,
        "delta_coefficient": delta_coefficient,
        "perc_delta_endowment": perc_delta_endowment,
        "perc_delta_coefficient": perc_delta_coefficient,
        "coefs_star": coefs_star,
    }
