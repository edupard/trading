import datetime
from enum import Enum
from urllib import urlretrieve, urlencode
from urlparse import parse_qs, urlparse


class Market(Enum):
    ICE = 31
    BONDS = 2
    COMMODITIES = 24
    CURRENCIES = 45
    ETF = 28
    ETF_MOEX = 515
    FUTURES = 14  # non-expired futures
    FUTURES_ARCHIVE = 17  # expired futures
    FUTURES_USA = 7
    INDEXES = 6
    SHARES = 1
    SPB = 517
    USA = 25


class Timeframe(Enum):
    TICKS = 1
    MINUTES1 = 2
    MINUTES5 = 3
    MINUTES10 = 4
    MINUTES15 = 5
    MINUTES30 = 6
    HOURLY = 7
    DAILY = 8
    WEEKLY = 9
    MONTHLY = 10


IMMUTABLE_PARAMS = {
    'f': 'table',
    'e': '.csv',
    'dtf': '1',
    'tmf': '1',
    'MSOR': '1',
    'mstimever': '0',
    'sep': '1',
    'sep2': '1',
    'at': '1',
    'apply' : 0
}

DEFAULT_EXPORT_HOST = 'export.finam.ru'


def _build_url(params):
    url = ('http://{}/table.csv?{}&{}'
           .format(DEFAULT_EXPORT_HOST,
                   urlencode(IMMUTABLE_PARAMS),
                   urlencode(params)))
    return url


def download(em,
             filename,
             code,
             market,
             start_date=datetime.date(2017, 1, 1),
             end_date=datetime.date.today(),
             timeframe=Timeframe.MINUTES1):
    params = {
        'p': timeframe,
        'em': em,
        'market': market,
        'df': start_date.day,
        'mf': start_date.month - 1,
        'yf': start_date.year,
        'from' : start_date.strftime('%d.%m.%Y'),
        'dt': end_date.day,
        'mt': end_date.month - 1,
        'yt': end_date.year,
        'to': end_date.strftime('%d.%m.%Y'),
        'cn': code,
        'code': code,
        # I would guess this param denotes 'data format'
        # that differs for ticks only
        'datf': 6 if timeframe == Timeframe.TICKS else 5,

    }
    url = _build_url(params)
    good_url = "http://export.finam.ru/ICE.BRN%206X-ICE_160801_161130.csv?market=31&em=408262&code=ICE.BRN+6X-ICE&apply=0&df=1&mf=7&yf=2016&from=01.08.2016&dt=30&mt=10&yt=2016&to=30.11.2016&p=2&f=ICE.BRN+6X-ICE_160801_161130&e=.csv&cn=ICE.BRN+6X-ICE&dtf=1&tmf=1&MSOR=1&mstimever=0&sep=1&sep2=1&datf=5&at=1"

    url_params = parse_qs(urlparse(url).query)
    good_params = parse_qs(urlparse(good_url).query)

    for k, v in good_params.iteritems():
        if k in url_params:
            value = url_params[k]
            if value != v:
                print ("%s expected: %s but was %s" % (k, v, value))
        else:
            print("%s is missing, expected %s" % (k, v))

    for k, v in url_params.iteritems():
        if k not in good_params:
            print("%s is redundant: %s" % (k, v))

    urlretrieve(url, filename)

