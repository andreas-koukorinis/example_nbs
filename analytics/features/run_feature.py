import logging
from sgmtradingcore.analytics.features.runner import FeatureRunner
from sgmtradingcore.analytics.features.request import FeatureRequest
from stratagemdataprocessing.enums.odds import Bookmakers


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - ''%(levelname)s - %(message)s')

    # These are the shared objects than can be used by all the Features.

    requests = [
        FeatureRequest('BookPressure', {'bookmakers': [Bookmakers.BETFAIR,
                                                       Bookmakers.MATCHBOOK,
                                                       Bookmakers.BETDAQ]},
                       {},
                       prefix=''),
        FeatureRequest('MicroPrice', {'bookmakers': [Bookmakers.BETFAIR,
                                                     Bookmakers.MATCHBOOK,
                                                     Bookmakers.BETDAQ]},
                       {}),
        FeatureRequest('MicroPrice', {'bookmakers': [Bookmakers.BETFAIR,
                                                     Bookmakers.MATCHBOOK,
                                                     Bookmakers.BETDAQ],
                                      'flipped': True},
                       {},
                       prefix='ggg_'),
    ]

    stickers = ['S-EGSM2465098-FT1X2-X']

    runner = FeatureRunner()

    df = runner.get_dataframes_by_stickers(requests, stickers)

    print 'returned'
    print df[stickers[0]]
