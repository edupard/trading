from finam.download import download, Market
import datetime
import dateutil.relativedelta

em=408262

MONTH_CODES = {
    1 : "F",
    2 : "G",
    3 : "H",
    4 : "J",
    5 : "K",
    6 : "M",
    7 : "N",
    8 : "Q",
    9 : "U",
    10 : "V",
    11 : "X",
    12 : "Z"
}

for y in range(2010, 2019):
    for m in range(1, 13):
        y = 2016
        m = 11
        code = "ICE.BRN %d%s-ICE" % (y %10, MONTH_CODES[m])
        filename = "BRENT-%02d.%02d.csv" % (m, y)
        checkpoint_date = datetime.date(y, m, 1)
        fromDate = checkpoint_date - dateutil.relativedelta.relativedelta(month=3)
        toDate = checkpoint_date + dateutil.relativedelta.relativedelta(month=2)
        # %Y %m %d
        fromDate = datetime.date(2016, 8, 1)
        toDate = datetime.date(2016, 11, 30)
        download(em,
                 filename,
                 code,
                 Market.ICE,
                 fromDate,
                 toDate)
        print(code)
        break
    break