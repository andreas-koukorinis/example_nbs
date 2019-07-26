import pytz
import pandas as pd
from datetime import datetime, timedelta
from sgmtradingcore.analytics.features.crypto.runner import CryptoFeatureRunner
from sgmtradingcore.analytics.features.request import FeatureRequest


def get_empty_candles_report(ticker, start_dt, end_dt, freq, lob_or_trades="trades"):
    """
    :param ticker: crypto exchange ticker
    :param start_dt (datetime): start date (must be rounded to the nearest day)
    :param end_dt (datetime): end_date  (must be rounded to the nearest day)
    :param freq (str): candles' frequency, currently supported: '15m', '1h', '4h', '1d'.
    :param lob_or_trades (str): 'trades' if we want the candles based on trades, 'lob' if we want candles
    based on lob
    :return: The function finds the missing candles for the specified interval and frequency. It returns a
    dataframe containing the following columns:
    - expected_candles (number)
    - missing_number ( how many candles we are missing)
    - missing_candles (the actual intervals/single timestamps we are missing)
    """
    if lob_or_trades == "lob":
        request = [FeatureRequest('OHLC',
                       {'frequency': freq,
                        'base_ts':{'is_feature_base_ts' :False,
                                   'base_ts_name': 'l1_mid'}
                       },{})]
    elif lob_or_trades == "trades":
        request = [FeatureRequest('OHLCTrades',
                                  {'frequency': freq},
                                  {},
                                  )]
    else:
        raise ValueError("lob_or_trades {} not supported yet".format(lob_or_trades))
    runner = CryptoFeatureRunner()
    df = runner.get_merged_dataframes(request, start_dt, end_dt)
    #  drop None/Nan values, ie. no candles
    df = df.dropna()
    if freq == '15m':
        freq = '15min'
    #  round dates ro nearest hour
    start_dt = datetime(*start_dt.timetuple()[:4])
    end_dt = datetime(*end_dt.timetuple()[:4])
    expected_timestamps = pd.date_range(start=start_dt, end=end_dt, freq=freq)[:-1]
    actual_timestamps = df.index
    expected_candles_nr = len(expected_timestamps)
    actual_candles_nr = len(actual_timestamps)
    missing_number = expected_candles_nr - actual_candles_nr
    missing_candles = list(set(expected_timestamps) - set(actual_timestamps))
    missing_candles = sorted(missing_candles)
    if freq != '1d':
          missing_candles = get_date_range(missing_candles, freq)
    report_dict = {'expected_candles': expected_candles_nr,
                   'missing_number': missing_number, 'missing_candles': [missing_candles]}
    df = pd.DataFrame.from_dict(report_dict, orient='index')
    df = df.reindex(index=['expected_candles', 'missing_candles', 'missing_number'])
    return df


def get_date_range(dates, freq):
    time_dict = {'15min': 15, '1h': 60, '4h': 240, '1d': 1440}
    first, last = dates[0], dates[0]
    date_ranges = []
    min = time_dict[freq]
    # Loop over the sorted list from the second value
    group_number = 0
    for d in dates[1:]:
        # "last" date
        if d - last != timedelta(minutes=min):
            if group_number != 0:
                date_ranges.append(tuple(sorted({first, last})))
            first, last = d, d
            group_number = 0
        else:
            group_number += 1
            last = d

    if group_number:
        date_ranges.append(tuple(sorted({first, last})))
    return date_ranges


def main():
    from stratagemdataprocessing.crypto.enums import get_last_valid_datetime, get_first_valid_datetime
    time_frames = ['15m', '1h', '4h', '1d']
    ticker = 'BTCUSD.PERP.BMEX'
    final_df = pd.DataFrame()

    #  get dataframe with all frequences. Round the date to the nearest day.
    _FIRST_VALID_DATETIMES = get_first_valid_datetime()
    _LAST_VALID_DATETIMES = get_last_valid_datetime()
    for frequency in time_frames:
        start_dt = _FIRST_VALID_DATETIMES[ticker]
        end_dt = _LAST_VALID_DATETIMES[ticker]
        df = get_empty_candles_report(ticker, start_dt, end_dt, frequency, lob_or_trades="trades")
        final_df = pd.concat([final_df, df], axis=1)
    final_df.columns = time_frames
    final_df.to_csv("Bitmex-Trades-2018.csv")


if __name__ == "__main__":
    main()
