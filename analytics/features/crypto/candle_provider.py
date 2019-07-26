from collections import defaultdict
from collections import namedtuple
import pandas as pd
from sgmtradingcore.util.misc import round_datetime


def make_ohlc_candles(timestamps, values, period):
    """

    Create OHLC dataframe based on a numerical series.
    'timestamps' and 'values' must have the same len
    :param timestamps: list of datetime
    :param values: list of int or floats
    :param period: timedelta
    :return:
    """
    if len(timestamps) != len(values):
        raise ValueError("Bad len")

    values_by_period = defaultdict(list)
    for t, v in zip(timestamps, values):
        period_start = round_datetime(t, period)
        values_by_period[period_start].append(v)
    index = []
    opens = []
    closes = []
    highs = []
    lows = []
    for timestamp, v in values_by_period.iteritems():
        index.append(timestamp)
        open = None
        close = None
        if len(v) > 0:
            open = v[0]
            close = v[-1]
        min_price = min(v)
        max_price = max(v)

        opens.append(open)
        closes.append(close)
        highs.append(max_price)
        lows.append(min_price)

    out = pd.DataFrame(data={'open': opens,
                             'close': closes,
                             'high': highs,
                             'low': lows},
                       index=[index])
    out.sort_index(inplace=True)
    out.index.name = 'timestamp'
    return out


def heiken_ashi(df, prev_open=None, prev_close=None, freq='5m'):
    """
              Heikin-Ashi Candle Calculations
          HA_Close = (Open + High + Low + Close) / 4
          HA_Open = (previous HA_Open + previous HA_Close) / 2
          HA_Low = minimum of Low, HA_Open, and HA_Close
          HA_High = maximum of High, HA_Open, and HA_Close

                  Heikin-Ashi Calculations on First Run
          HA_Close = (Open + High + Low + Close) / 4
          HA_Open = (Open + Close) / 2
          HA_Low = Low
          HA_High = High
    :param df: current DataFrame
    :param prev_open: previous open value
    :param prev_close: previous close value
    :param freq: frequency for calculating time threshold in case there is no date for some intervals
    :return: Heikin Ashi dataframe
    """
    # check case when we have no data for specific time intervals by adding delta column
    if len(df) == 0:
        return df
    time_dict = {'5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}  # in minutes
    serr_diff = df.index.to_series().diff()
    df['delta'] = serr_diff.dt.total_seconds().div(60, fill_value=0)
    thres = time_dict[freq]

    df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close'])/4

    pair = namedtuple('pair', ['ha_open', 'ha_close'])
    if prev_open is None:
        previous_row = pair(df.ix[0, 'open'], df.ix[0, 'close'])
    else:
        previous_row = pair(prev_open, prev_close)
    i = 0

    for row in df.itertuples():
        if row.delta > thres:
            ha_open = (row.open + row.close) / 2
        else:
            ha_open = (previous_row.ha_open + previous_row.ha_close) / 2
        df.ix[i, 'ha_open'] = ha_open
        previous_row = pair(ha_open, row.ha_close)
        i += 1
    df['ha_high'] = df[['ha_open', 'ha_close', 'high']].max(axis=1)
    df['ha_low'] = df[['ha_open', 'ha_close', 'low']].min(axis=1)
    df = df.drop(['open', 'low', 'high', 'close', 'delta'], axis=1)
    df = df[['ha_low', 'ha_open', 'ha_close', 'ha_high']]
    return df


def simple_moving_average(df, window, freq, col):
    """
    :param df: Timeseries DataFrame
    :param window: type(int) , rolling window for the moving average
    :param freq: type(str)   ,required frequency for downsampling the data (used in order to deal with data gaps_
    :param col:  type (str) ,name of the column on which we computer the MA
    :return: Timeseries DataFrame containing MA (moving average) as a columns.
            Note: In the case we have gaps in the data, the moving average will reset.
    """
    if freq == '15m':
        time_window = "{0}{1}".format(15*window, 'min')
    elif freq == '1h':
        time_window = "{0}{1}".format(window, 'h')
    elif freq == '4h':
        time_window = "{0}{1}".format(4*window, 'h')
    elif freq == '1d':
        time_window = "{0}{1}".format(window, 'd')
    else:
        raise ValueError("frequency {} not supported yet".format(freq))

    if len(df):
        df1 = df.copy()
        df1['MA'] = df1[col].rolling(time_window, min_periods=window).mean()
    else:
        df1 = pd.DataFrame(columns=['MA'])

    df1 = df1[['MA']]
    return df1


def bollinger_bands(df, window, freq):
    """
    :param df: OHLC DataFrame
    :param window: type(int), moving average window
    :param freq: type(str), ohlc timeframe
    :return: datframe containing all the 3 bands which are calculated as below:

    middle_band = X window moving average
    upper_band = X moving average + (X window standard deviation of price x 2)
    lower_band = X moving average - (X standard deviation of price x 2)
    """

    if freq == '15m':
        time_window = "{0}{1}".format(15*window, 'min')
    elif freq == '1h':
        time_window = "{0}{1}".format(window, 'h')
    elif freq == '4h':
        time_window = "{0}{1}".format(4*window, 'h')
    elif freq == '1d':
        time_window = "{0}{1}".format(window, 'd')
    else:
        raise ValueError("frequency {} not supported yet".format(freq))

    if len(df):
        df1 = df.copy()
        df1['middle_band'] = df1['close'].rolling(time_window, min_periods=window).mean()
        df1['STD'] = df1['close'].rolling(time_window, min_periods=window).std()
        df1['upper_band'] = df1['middle_band'] + (2 * df1['STD'])
        df1['lower_band'] = df1['middle_band'] - (2 * df1['STD'])
        df1 = df1.drop(['open', 'high', 'low', 'close', 'STD'], axis=1)
    else:
        df1 = pd.DataFrame(columns=['middle_band', 'lower_band', 'upper_band'])

    df1 = df1[['lower_band', 'middle_band', 'upper_band']]
    return df1
