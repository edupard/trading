import numpy as np


def hms(hhmmss):
    h = (hhmmss - hhmmss % 10000) / 10000
    mmss = hhmmss % 10000
    m = (mmss - mmss % 100) / 100
    s = mmss % 100
    return (h, m, s)


SECONDS_IN_DAY = 24 * 60 * 60

TEST_IDX = 0 + 0 * 60 + 10 * 60 *60


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

    gross_amount = np.zeros([SECONDS_IN_DAY])
    gross_volume = np.zeros([SECONDS_IN_DAY])
    avg_px = np.zeros([SECONDS_IN_DAY])

    time = csv_content[:, 1]
    h, m, s = hms(time)
    idx = s + m * 60 + h * 3600
    idx = idx.astype(dtype=np.int)

    vol = csv_content[:, 3]
    px = csv_content[:, 2]
    amount = vol * px

    np.add.at(gross_volume, idx, vol)
    np.add.at(gross_amount, idx, amount)

    non_zero_idx = np.nonzero(gross_amount)

    avg_px[non_zero_idx] = gross_amount[non_zero_idx] / gross_volume[non_zero_idx]

    data[0, :] = avg_px
    data[1, :] = gross_volume
    return data


# h, m, s = hms(np.array([101223, 141500]))
data = load_data(20190124)
i = 0
