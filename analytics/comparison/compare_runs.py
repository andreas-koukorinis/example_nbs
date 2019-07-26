import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import imread, subplot, imshow, show
import matplotlib.pyplot as plt

import pprint
from sgmtradingcore.analytics.comparison.trades_stats import \
    StrategyRunStatsHelper
from sgmtradingcore.analytics.comparison.postprocess_stats import \
    get_biggest_dict_diffs
from sgmtradingcore.analytics.comparison.presentation import RunStatsPresenter

BETA = '552f6a9139fdca41ca28b01a'
TRADING = '54da2b5fd47e6bff0dade9b4'


def main():
    """
    Compare two backtests and a realtime run
    :return:
    """
    strategy_name = 'tennis_sip_template_Challenger'
    strategy_desc = 'tennis_sip_template_clay_Challenger_v1'
    strategy_code = 'challenger_set1_clay_february2018'

    mnemonic1 = 'lorenzo_exsim_delay'
    mnemonic2 = 'lorenzo_trades_matcher_delay_20'
    trading_user_id = TRADING
    start_date = '2018-04-01'
    end_date = '2018-06-01'

    compare_two_backtest_and_realtime(strategy_name, strategy_desc,
                                      strategy_code, start_date,
                                      end_date, trading_user_id, mnemonic1,
                                      mnemonic2)


def compare_backtest_and_realtime(strategy_name, strategy_desc,
                                  strategy_code, start_date,
                                  end_date, trading_user_id,
                                  mnemonic):
    pp = pprint.PrettyPrinter(indent=2)
    # strategy_name = 'tennis_sip_template_WTA'
    # strategy_desc = 'tennis_sip_template_WTA_v1'
    # strategy_code = 'wta_set1_august2017_fixed_new'
    helper = StrategyRunStatsHelper()
    instructions, orders = helper.load_orders_and_instructions(
        strategy_name, strategy_desc, strategy_code, mnemonic,
        trading_user_id, start_date, end_date)
    stats_bkt = helper.get_stats(strategy_name, strategy_desc, strategy_code,
                                 mnemonic, trading_user_id,
                                 start_date, end_date, instructions, orders)

    mnemonic_prod = 'prod'
    instructions, orders = helper.load_orders_and_instructions(strategy_name,
                                                               strategy_desc,
                                                               strategy_code,
                                                               mnemonic_prod,
                                                               trading_user_id,
                                                               start_date,
                                                               end_date,
                                                               is_prod=True)
    stats_prod = helper.get_stats(strategy_name, strategy_desc, strategy_code,
                                  mnemonic_prod, trading_user_id,
                                  start_date, end_date, instructions, orders)
    del instructions
    del orders
    print "Biggest event pnl diffs with prod are are"
    pp.pprint(get_biggest_dict_diffs(stats_prod.per_event_pnl,
                                     stats_bkt.per_event_pnl))
    print "Biggest event vols diffs are"
    pp.pprint(get_biggest_dict_diffs(stats_prod.per_event_vol,
                                     stats_bkt.per_event_vol))
    print "Biggest daily diffs are"
    pp.pprint(
        get_biggest_dict_diffs(stats_prod.pnl_by_date,
                               stats_bkt.pnl_by_date))
    print "vol_by_bookmaker prod"
    pp.pprint(stats_prod.vol_by_bookmaker)
    print "vol_by_bookmaker backtest"
    pp.pprint(stats_bkt.vol_by_bookmaker)
    print "pnl_by_bookmaker prod"
    pp.pprint(stats_prod.pnl_by_bookmaker)
    print "pnl_by_bookmaker backtest"
    pp.pprint(stats_bkt.pnl_by_bookmaker)
    print "pnl_by_side prod"
    pp.pprint(stats_prod.pnl_by_side)
    print "pnl_by_side backtest"
    pp.pprint(stats_bkt.pnl_by_side)

    presenter = RunStatsPresenter()
    filepath = presenter.make_daily_pnl_graph(
        [stats_bkt, stats_prod])
    img = mpimg.imread(filepath)
    fig = plt.imshow(img)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.show()


def compare_two_backtest_and_realtime(strategy_name, strategy_desc,
                                      strategy_code, start_date,
                                      end_date, trading_user_id, mnemonic1,
                                      mnemonic2):
    pp = pprint.PrettyPrinter(indent=2)
    # strategy_name = 'tennis_sip_template_WTA'
    # strategy_desc = 'tennis_sip_template_WTA_v1'
    # strategy_code = 'wta_set1_august2017_fixed_new'
    helper = StrategyRunStatsHelper()
    instructions, orders = helper.load_orders_and_instructions(
        strategy_name, strategy_desc, strategy_code, mnemonic1,
        trading_user_id, start_date, end_date)
    stats_exsim = helper.get_stats(strategy_name, strategy_desc, strategy_code,
                                   mnemonic1, trading_user_id,
                                   start_date, end_date, instructions, orders)
    # TODO split loading of orders/instruction and get_Stats cesause sometime
    #  we have to filter loading...
    instructions, orders = helper.load_orders_and_instructions(
        strategy_name, strategy_desc, strategy_code, mnemonic2,
        trading_user_id,
        start_date, end_date)
    stats_tm = helper.get_stats(strategy_name, strategy_desc, strategy_code,
                                mnemonic2, trading_user_id,
                                start_date, end_date, instructions, orders)
    mnemonic = 'prod'
    instructions, orders = helper.load_orders_and_instructions(strategy_name,
                                                               strategy_desc,
                                                               strategy_code,
                                                               mnemonic,
                                                               trading_user_id,
                                                               start_date,
                                                               end_date,
                                                               is_prod=True)
    stats_prod = helper.get_stats(strategy_name, strategy_desc, strategy_code,
                                  mnemonic, trading_user_id,
                                  start_date, end_date, instructions, orders)
    del instructions
    del orders
    print "Biggest event pnl diffs with prod are are"
    pp.pprint(get_biggest_dict_diffs(stats_prod.per_event_pnl,
                                     stats_tm.per_event_pnl))
    print "Biggest event vols diffs are"
    pp.pprint(get_biggest_dict_diffs(stats_exsim.per_event_vol,
                                     stats_tm.per_event_vol))
    print "Biggest event pnl diffs are"
    pp.pprint(get_biggest_dict_diffs(stats_exsim.per_event_pnl,
                                     stats_tm.per_event_pnl))
    print "Biggest daily diffs are"
    pp.pprint(
        get_biggest_dict_diffs(stats_exsim.pnl_by_date, stats_tm.pnl_by_date))
    print "vol_by_bookmaker prod"
    pp.pprint(stats_prod.vol_by_bookmaker)
    print "vol_by_bookmaker exsim"
    pp.pprint(stats_exsim.vol_by_bookmaker)
    print "vol_by_bookmaker trades_matcher"
    pp.pprint(stats_tm.vol_by_bookmaker)
    print "pnl_by_bookmaker prod"
    pp.pprint(stats_prod.pnl_by_bookmaker)
    print "pnl_by_bookmaker exsim"
    pp.pprint(stats_exsim.pnl_by_bookmaker)
    print "pnl_by_bookmaker trades_matcher"
    pp.pprint(stats_tm.pnl_by_bookmaker)
    print "pnl_by_side prod"
    pp.pprint(stats_prod.pnl_by_side)
    print "pnl_by_side exsim"
    pp.pprint(stats_exsim.pnl_by_side)
    print "pnl_by_side trades_matcher"
    pp.pprint(stats_tm.pnl_by_side)
    presenter = RunStatsPresenter()
    filepath = presenter.make_daily_pnl_graph(
        [stats_exsim, stats_tm, stats_prod])
    img = mpimg.imread(filepath)
    fig = plt.imshow(img)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
