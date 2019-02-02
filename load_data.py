import numpy as np
import preprocess


def hms(hhmmss):
    h = (hhmmss - hhmmss % 10000) / 10000
    mmss = hhmmss % 10000
    m = (mmss - mmss % 100) / 100
    s = mmss % 100
    return (h, m, s)


AVG_VOLUME_PER_SECOND = 114237.239471109
AVG_VOLUME_PER_SECOND_STDDEV = 39145.5378031463


SECONDS_IN_DAY = 24 * 60 * 60

BEG_IDX = 0 + 0 * 60 + 10 * 60 * 60
END_IDX = 0 + 0 *60 + 19 * 60 *60 - 1


def load_data(yyyymmdd, PX_GAMMA):
    # date, time, px, vol
    csv_content = np.loadtxt('data/test.csv', skiprows=1, delimiter=',')
    date = csv_content[:, 0]
    date_match = date == yyyymmdd
    csv_content = csv_content[date_match, :]

    # create target array:
    # [px, vol: SECONDS_IN_DAY]
    # aggregate seconds
    data = np.zeros([SECONDS_IN_DAY, 6])

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

    px = preprocess.roll_arr_fwd(px)
    px = preprocess.roll_arr_bwd(px)

    px_pct = preprocess.pct_chg(px)

    vol_normalized = (vol - AVG_VOLUME_PER_SECOND) / AVG_VOLUME_PER_SECOND_STDDEV

    px_rema = preprocess.arr_rema(px, PX_GAMMA)

    value = preprocess.pct_diff(px, px_rema)

    return vol, vol_normalized, px, px_rema, px_pct, value

