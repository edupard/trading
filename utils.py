import numpy as np


def ema(prev_ema, val, gamma):
    return val * gamma + (1-gamma) * prev_ema;


def arr_ema(arr, gamma):
    def _ema(prev_ema, val):
        return ema(prev_ema, val, gamma)

    v_ema = np.frompyfunc(_ema, 2, 1)

    return v_ema.accumulate(arr, dtype=np.object).astype(np.float)


def _roll_fwd(prev, val):
    return prev if val <= 0 else val


_v_roll_fwd = np.frompyfunc(_roll_fwd, 2, 1)


def roll_arr_fwd(arr):
    return _v_roll_fwd.accumulate(arr, dtype=np.object).astype(np.float)


def roll_arr_bwd(arr):
    intermediate = roll_arr_fwd(arr[::-1])
    return intermediate[::-1]


def _calc_pct(new, old):
    if old > 0:
        return (new - old) / old
    return 0


_v_pct = np.vectorize(_calc_pct, otypes=[np.float])


def pct_chg(arr):
    prev_arr = np.roll(arr, 1)
    prev_arr[0] = 0
    return _v_pct(arr, prev_arr)

