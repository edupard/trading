import numpy as np


def ema(prev_ema, val, gamma):
    return val * gamma + (1-gamma) * prev_ema;


def arr_ema(arr, gamma):
    def _ema(prev_ema, val):
        return ema(prev_ema, val, gamma)

    v_ema = np.frompyfunc(_ema, 2, 1)

    return v_ema.accumulate(arr, dtype=np.object).astype(np.float)


def arr_rema(arr, gamma):
    intermediate = arr_ema(arr[::-1], gamma)
    rema = intermediate[::-1]
    last= rema[-1]
    # take next
    rema = np.roll(rema, -1)
    rema[-1] = last

    return rema


def _roll_fwd(prev, val):
    return prev if val <= 0 else val


_v_roll_fwd = np.frompyfunc(_roll_fwd, 2, 1)


def roll_arr_fwd(arr):
    return _v_roll_fwd.accumulate(arr, dtype=np.object).astype(np.float)


def roll_arr_bwd(arr):
    intermediate = roll_arr_fwd(arr[::-1])
    return intermediate[::-1]


def _pct(old, new):
    if old > 0:
        return (new - old) / old
    return 0


_v_pct = np.vectorize(_pct, otypes=[np.float])


def pct_chg(arr):
    prev_arr = np.roll(arr, 1)
    prev_arr[0] = 0
    return _v_pct(prev_arr, arr)


def pct_diff(prev, next):
    return _v_pct(prev, next)


def new_state(state, value, threshold):
    if value > threshold:
        return 1.
    elif value < -threshold:
        return -1.
    return state


def get_state(value, threshold):

    value = np.insert(value, 0, 0., axis=0)

    def _new_state(state, value):
        return new_state(state, value, threshold)

    _v_new_state = np.frompyfunc(_new_state, 2, 1)

    state = _v_new_state.accumulate(value, dtype=np.object).astype(np.float)
    return state[1:]

