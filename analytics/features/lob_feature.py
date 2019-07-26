import pandas as pd
import tqdm

from sgmtradingcore.analytics.features.market_indicators import MarketFeature
from sgmtradingcore.analytics.features.storage import InMemoryFeatureStorage, PickleFileFeatureStorage
from sgmtradingcore.exchange.lob.command_extractor import CommandExtractor
from sgmtradingcore.exchange.lob.lob import Lob
from sgmtradingcore.providers.odds_providers import get_odds_from_sticker
from stratagemdataprocessing.parsing.common.stickers import extract_tradeable_bookmaker, BOOKMAKER_ABBR


class LobOrdersFeature(MarketFeature):

    def __init__(self, runner, params, **kwargs):
        storage = InMemoryFeatureStorage(self, parent_cache=PickleFileFeatureStorage(self))
        super(LobOrdersFeature, self).__init__(runner, params, storage, **kwargs)

    def _compute_by_sticker(self, sticker):

        tradeable, bookmaker = extract_tradeable_bookmaker(sticker)

        ticks = get_odds_from_sticker(sticker, odds_cache=self._runner.shared_objects()['odds_cache'])
        if ticks is None:
            return pd.DataFrame([])

        extractor = CommandExtractor(BOOKMAKER_ABBR[bookmaker])
        lob = Lob()

        for ii, tick in tqdm.tqdm(enumerate(ticks), desc='ticks'):

            commands = extractor.process_tick(tick)

            for command in commands:
                lob.process_command(command)

        return lob.to_dataframe(compact=True)