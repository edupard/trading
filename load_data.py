import numpy as np


def hms(hhmmss):
    h = (hhmmss - hhmmss % 10000) / 10000
    mmss = hhmmss % 10000
    m = (mmss - mmss % 100) / 100
    s = mmss % 100
    return (h, m, s)


def load_data():
    # date, time, px, vol
    csv_content = np.loadtxt('data/test.csv', skiprows=1, delimiter=',')
    # create target array:
    # [px, vol: 24*60*60]
    # aggregate seconds
    data = np.zeros([2, 24 * 60 * 60])
    time = csv_content[:, 1]
    h, m, s = hms(time)
    idx = s + m * 60 + h * 3600
    idx = idx.astype(dtype=np.int)

    vol = csv_content[:, 3]

    data[0, idx] = vol
    return data


# h, m, s = hms(np.array([101223, 141500]))
data = load_data()
i = 0
