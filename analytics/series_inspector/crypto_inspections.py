import pytz
import logging
from datetime import timedelta, datetime
from sgmtradingcore.analytics.series_inspector.common import InspectionStatus, InspectionResult
from stratagemdataprocessing.crypto.market.cassandra_data import get_cassandra_lob_ticks, get_cassandra_connection, \
    get_cassandra_trades_ticks
from sgmtradingcore.analytics.series_inspector.inspections import Inspection


class MissingRealtimeCryptoData(Inspection):
    """
    Check that there is no gap in cassandra crypto data, lob and trades.
    """
    EMAILS = ['lorenzo@stratagem.co', 'crypto@stratagem.co']

    def __init__(self, tickers, data_types, max_ok_diff=timedelta(seconds=240)):
        super(MissingRealtimeCryptoData, self).__init__()
        self.tickers = tickers
        # ['lob', 'trades']
        self.data_types = data_types
        for t in self.data_types:
            if t not in ('lob', 'trades'):
                raise ValueError("Bad data_type: {}".format(self.data_types))
        self.max_ok_diff = max_ok_diff

        self._conn = None

    def inspect_series(self, start_dt, end_dt):
        ret = []
        for ticker in self.tickers:
            for data_type in self.data_types:
                ticks = self._get_ticks(ticker, start_dt, end_dt, data_type)
                ret.extend(self._get_results(ticks, ticker, data_type, start_dt, end_dt))

        return ret

    def _get_results(self, ticks, ticker, data_type, start_dt, end_dt):
        ret = []
        last_ts = start_dt
        max_diff = timedelta(seconds=0)
        max_diff_start_ts = start_dt

        for tick in ticks:
            current_ts = self._get_ts(tick, data_type)
            diff = current_ts - last_ts
            if diff > max_diff:
                max_diff = diff
                max_diff_start_ts = last_ts

            last_ts = tick.received_at

        _end_dt = min(datetime.now(tz=pytz.utc), end_dt)
        diff = _end_dt - last_ts
        if diff > max_diff:
            max_diff = diff
            max_diff_start_ts = last_ts

        if len(ticks) == 0:
            ret.append(InspectionResult(
                InspectionStatus.ERROR,
                '{} {} {}'.format(ticker, len(ticks), data_type),
                '{} {} from cassandra has no ticks between {} and {}'.format(
                    ticker, data_type, start_dt, end_dt)))
        elif max_diff <= self.max_ok_diff:
            ret.append(InspectionResult(
                InspectionStatus.OK,
                '{} {} {} OK'.format(ticker, len(ticks), data_type),
                '{} {} from cassandra has no gap of more than 60 seconds'.format(ticker, data_type)))
        else:
            ret.append(InspectionResult(
                InspectionStatus.ERROR,
                '{} {} {}: {} seconds gap'.format(ticker, len(ticks), data_type, int(max_diff.total_seconds())),
                '{} {} from cassandra has a gap of {} seconds, starting at {}'.format(
                    ticker, data_type, max_diff.total_seconds(), max_diff_start_ts)))
        return ret

    def _get_ticks(self, ticker, start_dt, end_dt, data_type):
        if self._conn is None:
            self._conn = get_cassandra_connection()

        logging.info('Fetching from cassandra {} {}: {} to {}'.format(ticker, data_type, start_dt, end_dt))
        if data_type == 'lob':
            ticks = get_cassandra_lob_ticks(self._conn, ticker, start_dt, end_dt)
        else:
            ticks = get_cassandra_trades_ticks(self._conn, ticker, start_dt, end_dt)
        logging.info("Fetched {} ticks".format(len(ticks)))
        return ticks

    @staticmethod
    def _get_ts(tick, data_type):
        if data_type == 'lob':
            return tick.received_at
        elif data_type == 'trades':
            return tick.happened_at
        else:
            raise ValueError("Bad data_type: {}".format(data_type))
