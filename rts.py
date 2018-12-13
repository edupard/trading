import urllib.request
import datetime
from enum import IntEnum
from urllib.parse import urlencode


class Market(IntEnum):
 """
 Markets mapped to ids used by finam.ru export
 List is incomplete, extend it when needed
 """

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


class Timeframe(IntEnum):
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
        'd': 'd',
        'f': 'table',
        'e': '.csv',
        'dtf': '1',
        'tmf': '3',
        'MSOR': '0',
        'mstime': 'on',
        'mstimever': '1',
        'sep': '3',
        'sep2': '1',
        'at': '1'
}

DEFAULT_EXPORT_HOST = 'export.finam.ru'


def _build_url(params):
 url = ('http://{}/table.csv?{}&{}'
        .format(DEFAULT_EXPORT_HOST,
                urlencode(IMMUTABLE_PARAMS),
                urlencode(params)))
 return url


def download(code,
             market,
             start_date=datetime.date(2017, 1, 1),
             end_date=datetime.date.today(),
             timeframe=Timeframe.MINUTES1):

 params = {
            'p': timeframe.value,
            'em': 17455,
            'market': market.value,
            'df': start_date.day,
            'mf': start_date.month - 1,
            'yf': start_date.year,
            'dt': end_date.day,
            'mt': end_date.month - 1,
            'yt': end_date.year,
            'cn': code,
            'code': code,
            # I would guess this param denotes 'data format'
            # that differs for ticks only
            'datf': 6 if timeframe == Timeframe.TICKS.value else 5
        }
 url = _build_url(params)
 urllib.request.urlretrieve(url, "./test.csv")

 return _build_url(params)


test = download('SPFB.RTS', Market.FUTURES_ARCHIVE)
print(test)

FORMAT = '%y%m%d'
FORMAT_QUERY = '%d.%m.%Y'

BEG = datetime.date(2017, 1, 1)
END = datetime.date(2018, 12, 12)

TICKER = 'SPFB.RTS'
BASE_URL = 'http://export.finam.ru'
MARKET = 14
EM=17455
url = '%s/%s_%s_%s.txt?market=%d&em=17455&code=%s&apply=0&df=%d&mf=%d&yf=%d&from=%s&dt=%d&mt=%d&yt=%d&to=%s&p=2&f=%s_%s_%s&e=.txt&cn=%s&dtf=1&tmf=1&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=1&at=1' % \
      (BASE_URL,
       TICKER,
       BEG.strftime(FORMAT),
       END.strftime(FORMAT),
       MARKET,
       TICKER,
       BEG.day,
       BEG.month - 1,
       BEG.year,
       BEG.strftime(FORMAT_QUERY),
       END.day,
       END.month - 1,
       END.year,
       END.strftime(FORMAT_QUERY),
       TICKER,
       BEG.strftime(FORMAT),
       END.strftime(FORMAT),
       TICKER
       )

sample_url = 'http://export.finam.ru/SPFB.RTS_170101_181212.txt?market=14&em=17455&code=SPFB.RTS&apply=0&df=1&mf=0&yf=2017&from=01.01.2017&dt=12&mt=11&yt=2018&to=12.12.2018&p=2&f=SPFB.RTS_170101_181212&e=.txt&cn=SPFB.RTS&dtf=1&tmf=1&MSOR=1&mstime=on&mstimever=1&sep=1&sep2=1&datf=1&at=1'

print(url)
print(sample_url)

for i in range(max(len(url), len(sample_url))):
 if url[i] != sample_url[i]:
  print(sample_url[i: len(sample_url) - i])
  break


test = sample_url.startswith(url)

urllib.request.urlopen(url)
