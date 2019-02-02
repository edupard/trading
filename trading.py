import numpy as np

def new_position(state, value, threshold):
    if value > threshold:
        return 1.
    elif value < -threshold:
        return -1.
    return state


def get_position(value, threshold):

    value = np.insert(value, 0, 0., axis=0)

    def _new_state(state, value):
        return new_position(state, value, threshold)

    _v_new_state = np.frompyfunc(_new_state, 2, 1)

    state = _v_new_state.accumulate(value, dtype=np.object).astype(np.float)
    return state[1:]


def get_pnl(pos, px):
    # get trades array
    prev_pos = np.roll(pos, 1)
    prev_pos[0] = 0.0
    trades = pos - prev_pos

    # cash
    cash_flow = -trades * px
    cash = np.cumsum(cash_flow)

    # mv
    mv = pos * px

    pl = mv + cash

    return pl
