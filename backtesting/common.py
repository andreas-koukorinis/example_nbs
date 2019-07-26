from sgmtradingcore.strategies.dummy.strategy import DummyStrategy

__author__ = 'lorenzo belli'

PARAM_CLASS_MAP = {
    'butterfly': DummyStrategy,
}

# If the market making framework is installed then we will try to add the
# MM genetic algorithm to the mapping for use with spark:
try:
    from sgmmktmaking.Trading_Strategy.optimisation.ga import MMChooseParamsGA
    PARAM_CLASS_MAP['market_making'] = MMChooseParamsGA

except ImportError:
    pass
