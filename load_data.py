import numpy as np


def hms(hhmmss):
    h = (hhmmss - hhmmss % 10000) / 10000
    mmss = hhmmss % 10000
    m = (mmss - mmss % 100) / 100
    s = mmss % 100
    return (h, m, s)


SECONDS_IN_DAY = 24 * 60 * 60

TEST_IDX = 0 + 0 * 60 + 10 * 60 *60

# when it close to 1 - we use most recent data
GAMMA = 0.7


def ema(prev_ema, value):
    return value * GAMMA + (1-GAMMA) * prev_ema


def roll_prev_px(next_value, value):
    return value if value > 0 else next_value


def load_data(yyyymmdd):
    # date, time, px, vol
    csv_content = np.loadtxt('data/test.csv', skiprows=1, delimiter=',')
    date = csv_content[:, 0]
    date_match = date == yyyymmdd
    csv_content = csv_content[date_match, :]

    # create target array:
    # [px, vol: SECONDS_IN_DAY]
    # aggregate seconds
    data = np.zeros([2, SECONDS_IN_DAY])

    amount = np.zeros([SECONDS_IN_DAY])
    vol = np.zeros([SECONDS_IN_DAY])
    px = np.zeros([SECONDS_IN_DAY])

    time = csv_content[:, 1]
    h, m, s = hms(time)
    idx = s + m * 60 + h * 3600
    idx = idx.astype(dtype=np.int)

    tr_vol = csv_content[:, 3]
    tr_px = csv_content[:, 2]
    tr_amount = tr_vol * tr_px

    np.add.at(vol, idx, tr_vol)
    np.add.at(amount, idx, tr_amount)

    non_zero_idx = np.nonzero(amount)

    px[non_zero_idx] = amount[non_zero_idx] / vol[non_zero_idx]

    data[0, :] = px
    data[1, :] = vol

    ema_ufunc = np.frompyfunc(ema, 2, 1)

    ema_volume = ema_ufunc.accumulate(vol, dtype=np.object)

    roll_prev_px_ufunc = np.frompyfunc(roll_prev_px, 2, 1)

    rolled_px_reversed = roll_prev_px_ufunc.accumulate(px[::-1], dtype=np.object)
    rolled_px = rolled_px_reversed[::-1]

    data[0,:] = px
    data[1,:] = vol

    # now we need to normalize volume
    # do exponential moving average
    # https://en.wikipedia.org/wiki/Moving_average

    # do ema with quiet high speed - need to forget high volumes quickly
    # ie we need to be sensitive to quick changes
    # so we have smooth non zero curve
    # then we need to feed in % change for this curve to get 0 mean input
    # plot graphs so i can pick ema factor properly

    # prices
    # first - fill in gaps

    # no reason to do ema for prices:
    # a) you need to see price changes immediatelly
    # b) NN can derieve ema if it needs
    # c) EMA is reversible
    # d) volumes are just tricky because you need to normalize
    # e) feed in % change to get 0 mean input


    return data


# h, m, s = hms(np.array([101223, 141500]))
data = load_data(20190124)
i = 0
