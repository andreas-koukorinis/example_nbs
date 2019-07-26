import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from stratagemdataprocessing.crypto.enums import get_last_valid_datetime, get_first_valid_datetime
from sgmtradingcore.analytics.features.crypto.runner import CryptoFeatureRunner
from sgmtradingcore.analytics.features.request import FeatureRequest
from stratagemdataprocessing.crypto.market.arctic.arctic_storage import ArcticStorage
from stratagemdataprocessing.dbutils.mongo import MongoPersister
from sgmtradingcore.analytics.features.crypto.candle_provider import make_ohlc_candles


def are_similar(v1, v2, threshold_perc):
    if v1 is None or v2 is None or np.isnan(v1) or np.isnan(v2):
        return True
    diff = abs(v2-v1)
    if diff <= v1 * threshold_perc:
        return True
    return False


def row_is_similar(row, col_pairs, threshold_perc):
    return all([are_similar(row[p[0]], row[p[1]], threshold_perc) for p in col_pairs])


def get_compared_candles(df, cols1, cols2, threshold_perc=0.01):
    if len(cols1) != len(cols2):
        raise ValueError("Bad len")

    out = df.copy()
    col_pairs = [[a, b] for a, b in zip(cols1, cols2)]

    out['are_similar'] = out.apply(row_is_similar, args=(col_pairs, threshold_perc), axis=1)
    return out


def log_different(row):
    if not row['are_similar']:
        # logging.info("{}".format(row))
        # logging.info("diff {}".format(row.name))
        pass
    pass


def compare_candles(ticker, df, cols1, cols2):
    compared_df = get_compared_candles(df, cols1, cols2)
    n = len(compared_df)
    num_good = np.sum(compared_df['are_similar'])
    num_bad = n-num_good
    logging.info("{}: {}/{} different candles".format(ticker, num_bad, n))
    compared_df.apply(log_different, axis=1)


def main():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - ' '%(levelname)s - %(message)s')

    tickers = ['BTCUSD.SPOT.BITS', 'BTCUSD.SPOT.BITF', 'BTCUSD.PERP.BMEX']
    frequency = '1d'

    runner = CryptoFeatureRunner()

    request = [
        FeatureRequest('OHLCTrades',
                       {
                           'frequency': frequency,
                       },
                       {},
                       ),
        # FeatureRequest('OHLC',
        #                {
        #                    'frequency': frequency,
        #                    'base_ts': {
        #                        'is_feature_base_ts': False,
        #                        'base_ts_name': TSInput.L1_MID,
        #                    }
        #                },
        #                {},
        #                prefix='t_'
        #                )
    ]

    _FIRST_VALID_DATETIMES = get_first_valid_datetime()
    _LAST_VALID_DATETIMES = get_last_valid_datetime()

    for ticker in tickers:
        runner = CryptoFeatureRunner()
        start_dt = _FIRST_VALID_DATETIMES[ticker]
        end_dt = _LAST_VALID_DATETIMES[ticker]
        df = runner.get_merged_dataframes(request, start_dt, end_dt)
        column_names = [
            'open',
            'high',
            'low',
            'close'
        ]
        column_names_t = [
            't_open',
            't_high',
            't_low',
            't_close'
        ]
        column_names_a = [
            'a_open',
            'a_high',
            'a_low',
            'a_close'
        ]

        mongo_client = MongoPersister.init_from_config('arctic_crypto', auto_connect=True)
        arctic = ArcticStorage(mongo_client.client)
        arctic_trades_df = arctic.load_trades(ticker, start_dt, end_dt)
        timestamps = arctic_trades_df.index.to_pydatetime().tolist()
        prices = arctic_trades_df['price'].tolist()

        arctic_candles = make_ohlc_candles(timestamps=timestamps, values=prices, period=timedelta(days=1))
        arctic_candles.columns = ['a_' + str(col) for col in arctic_candles.columns]

        df1 = pd.merge(df, arctic_candles)

        compare_candles(ticker, df1, column_names, column_names_a)

    # TODO do the same comparison with clean data


if __name__ == "__main__":
    main()