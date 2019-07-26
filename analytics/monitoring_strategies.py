from collections import defaultdict
from datetime import datetime, timedelta
from operator import itemgetter

import numpy as np
from bokeh.models import HoverTool, ColumnDataSource, LinearAxis, Range1d, PrintfTickFormatter, \
    DatetimeTickFormatter, NumeralTickFormatter
from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter
from bokeh.plotting import Figure, output_file, show
from bokeh.layouts import column

from sgmtradingcore.analytics.incremental_metrics import IncrementalMetrics
from sgmtradingcore.analytics.performance import get_trading_days
from sgmtradingcore.execution.monitoring import json_to_bet_info
from sgmtradingcore.providers.odds_providers import get_historical_odds, OddsStream
from sgmtradingcore.strategies.strategy_base import TennisStrategyFilter, FootballStrategyFilter
from stratagemdataprocessing import data_api
from stratagemdataprocessing.enums.odds import Bookmakers
from stratagemdataprocessing.enums.odds import Sports
from stratagemdataprocessing.parsing.common.stickers import parse_sticker


DAYS_PER_YEAR = 365

TRADING_USERS = {'algosports': '562f5bef497aee1c22000001',
                 'stratagem': '54da2b5fd47e6bff0dade9b4'}

POUND_SYMBOL = "(" + u'\u00A3' + ")"


class MonitoringMetrics:
    """
    API to be called by monitoring strategies notebook
    """

    def __init__(self):
        self.metrics = None
        self.settled_orders = None

        # load strategies from rpc service
        strategies_resp = data_api.get_strategies()

        # create maps
        self.strategies = {}
        self.sport_strategies_map = {}
        self.strategy_style = {}
        self.competitions = {}
        self.has_source = {}
        for strategy in strategies_resp:
            name = strategy['name']
            self.strategies[name] = strategy['descriptions']
            self.sport_strategies_map[name] = strategy['sport']
            self.strategy_style[name] = strategy['style']
            self.competitions[name] = strategy['competitions']
            self.has_source[name] = strategy['has_source']

        self.map_trade_id_to_orders = defaultdict(list)
        self.map_trade_id_to_stickers = defaultdict(set)

        self.strategies_names = self.strategies.keys()
        descriptions = self.strategies.values()
        self.strategies_descriptions = []
        for desc in descriptions:
            self.strategies_descriptions.extend(desc)

    @staticmethod
    def load_summary_metrics(trading_user):
        # load summary metrics from rpc service
        dt = datetime.now() - timedelta(days=2)
        str_date = dt.strftime('%Y-%m-%d')
        strategy_metrics = data_api.get_metrics(TRADING_USERS[trading_user], str_date)

        metrics_map = {}
        for item in strategy_metrics:
            name = item['name']
            metrics_map[item['trading_user_id'], name] = item
        return metrics_map

    def get_sport_filter(self, sport, strategy, strategy_descr=None):

        if strategy_descr is None:
            competitions = self.competitions[strategy][strategy]
        else:
            competitions = self.competitions[strategy][strategy_descr]

        if sport == Sports.TENNIS:
            sport_filter = TennisStrategyFilter(competitions)
        elif sport == Sports.FOOTBALL:
            sport_filter = FootballStrategyFilter(competitions)

        return sport_filter

    def calculate_metrics(self, strategy, strategy_descr, start_dt, end_dt, sport, style, trading_user):
        ret_success = True
        trading_user_id = TRADING_USERS[trading_user]
        settled_orders_expanded = data_api.get_settled_orders(trading_user_id, start_dt, end_dt, strategy,
                                                              strategy_descr)
        currencies = data_api.get_currency_rates([])

        orders = [json_to_bet_info(order_expanded, currencies=currencies) for order_expanded in
                  settled_orders_expanded if order_expanded['order_info']['sticker'] != u'']  # filter empty sticker

        settled_orders = sorted(orders, key=lambda x: x.settled_dt)

        if len(settled_orders) == 0:
            ret_success = False

        capital_timeseries = data_api.get_capital_timeseries(
            trading_user_id, sport, style, start_dt, end_dt, strategy, strategy_descr)

        if self.has_source[strategy]:
            if sport == Sports.FOOTBALL or sport == Sports.TENNIS:
                sport_filter = self.get_sport_filter(sport, strategy, strategy_descr)
                trading_days, total = get_trading_days(sport, start_dt, end_dt, sport_filter)
        else:
            step = timedelta(days=1)
            trading_days = []
            total = DAYS_PER_YEAR
            while start_dt <= end_dt:
                trading_days.append(start_dt)
                start_dt += step

        metrics = IncrementalMetrics(capital_timeseries, trading_days, total)

        for order in settled_orders:
            metrics.flat_capital_metrics_incremental([order])

        # this is to round up the final day as the incremental metrics normally
        # do so when they see a new day
        metrics.update_return_structures(True)
        metrics.flat_capital_metrics_incremental([])

        self.metrics = metrics
        self.settled_orders = settled_orders

        return ret_success

    def get_total_compound(self):
        if self.metrics is None:
            raise ValueError("Calculate metrics needs to be called before this method")
        sorted_values = sorted(self.metrics._cumsum_log_return_by_day.iteritems(), key=itemgetter(0))
        date, cumsum_log_return = zip(*sorted_values)
        x_date = list(date)
        y_cumsum_log_return = list(cumsum_log_return)

        # data compound return
        exp_cumsum_log_return = np.expm1(y_cumsum_log_return)
        return x_date, exp_cumsum_log_return

    def get_total_pnl(self):
        if self.metrics is None:
            raise ValueError("Calculate metrics needs to be called before this method")
        sorted_values = sorted(self.metrics._total_pnl_by_day.iteritems(), key=itemgetter(0))
        date, total_pnl = zip(*sorted_values)
        x_date = list(date)
        y_total_pnl = list(total_pnl)
        return x_date, y_total_pnl

    def get_capital_by_day(self, dates):
        if self.metrics is None:
            raise ValueError("Calculate metrics needs to be called before this method")
        # get the dates we have orders
        date_orders = []
        for i in dates:
            date_orders.append((i, self.metrics._capital_by_day[i]))
        date, capital = zip(*date_orders)
        x_date = list(date)
        y_capital= list(capital)
        return x_date, y_capital

    def construct_figure(self, x, y, metric):
        # workaround to format date in the hover tool at the moment bokeh
        source = ColumnDataSource(data=dict(x=x, y=y, time=[e.strftime('%d %b %Y') for e in x]))

        hover = HoverTool(
            tooltips=[
                ("Date", "@time"),
                ("Return", "@y{0.0%}"),
            ]
        )

        # create a new plot with a a datetime axis type
        p2 = Figure(x_axis_type="datetime", title=metric,toolbar_location="above",
                    tools=[hover, 'box_zoom, box_select, crosshair,resize, reset, save,  wheel_zoom'])

        # add renderers
        p2.circle(x, y, size=8, color='black', alpha=0.2, legend=metric, source=source)
        p2.line(x, y, color='navy', legend=metric, source=source)

        # NEW: customize by setting attributes
        # p2.title = metric
        p2.legend.location = "top_left"
        p2.grid.grid_line_alpha = 0
        p2.xaxis.axis_label = 'Date'
        p2.yaxis.axis_label = metric
        p2.ygrid.band_fill_color = "olive"
        p2.ygrid.band_fill_alpha = 0.1
        p2.xaxis.formatter = DatetimeTickFormatter(formats={'days': ['%d %b'], 'months': ['%b %Y']})
        p2.yaxis.formatter = NumeralTickFormatter(format="0.0%")
        # format="0.0%"

        return p2

    def construct_total_pnl_figure(self, x, y, t):
        str_total_pnl = "Total Pnl " + POUND_SYMBOL
        # workaround to format date in the hover tool at the moment bokeh does not supported in the tool tips
        time = [e.strftime('%d %b %Y') for e in x]
        source_total_pnl = ColumnDataSource(data=dict(x=x, y=y, time=time))

        tooltips_total_pnl = [
                ("Date", "@time"),
                ("Total Pnl", "@y{0.00}"),
            ]

        tooltips_capital = [
                ("Date", "@time"),
                ("Capital", "@y{0.00}"),
            ]

        # create a new pnl plot
        p2 = Figure(x_axis_type="datetime", title="Total Pnl/Capital Allocated " + POUND_SYMBOL,
                    toolbar_location="above", tools=['box_zoom, box_select, crosshair, resize, reset, save,  wheel_zoom'])
        # add renderers
        r1 = p2.circle(x, y, size=8, color='black', alpha=0.2, legend=str_total_pnl, source=source_total_pnl)
        r11 = p2.line(x, y, color='navy', legend=str_total_pnl, source=source_total_pnl)

        # add renderers to the HoverTool instead of to the figure so we can have different tooltips for each glyph
        p2.add_tools(HoverTool(renderers=[r1, r11], tooltips=tooltips_total_pnl))

        max_total_pnl = max(y)
        min_total_pnl = min(y)

        # offset to adjust the plot so the max and min ranges are visible
        offset = (max(abs(max_total_pnl), abs(min_total_pnl))) * 0.10
        p2.y_range = Range1d(min_total_pnl - offset, max_total_pnl + offset)

        # NEW: customize by setting attributes
        # p2.title = "Total Pnl/Capital Allocated " + POUND_SYMBOL
        p2.legend.location = "top_left"
        p2.grid.grid_line_alpha = 0
        p2.xaxis.axis_label = 'Date'
        p2.yaxis.axis_label = str_total_pnl
        p2.ygrid.band_fill_color = "olive"
        p2.ygrid.band_fill_alpha = 0.1
        p2.xaxis.formatter = DatetimeTickFormatter(formats={'days': ['%d %b'], 'months': ['%b %Y']})
        # formatter without exponential notation
        p2.yaxis.formatter = PrintfTickFormatter(format="%.0f")

        # secondary axis
        max_capital = max(t)
        min_capital = min(t)
        # offset to adjust the plot so the max and min ranges are visible
        offset = (max(abs(max_capital), abs(min_capital))) * 0.10
        p2.extra_y_ranges = {"capital": Range1d(start=min_capital - offset, end=max_capital + offset)}

        # formatter without exponential notation
        formatter = PrintfTickFormatter()
        formatter.format = "%.0f"

        # formatter=NumeralTickFormatter(format="0,0"))
        p2.add_layout(LinearAxis(y_range_name="capital", axis_label="Capital allocated " + POUND_SYMBOL,
                                 formatter=formatter), 'right')

        # create plot for capital series
        source_capital = ColumnDataSource(data=dict(x=x, t=t, time=time))
        r2 = p2.square(x, t, size=8, color='green', alpha=0.2, legend="Capital " + POUND_SYMBOL, y_range_name="capital",
                  source=source_capital)
        r22 = p2.line(x, t, color='green', legend="Capital " + POUND_SYMBOL, y_range_name="capital", source=source_capital)

        # add renderers to the HoverTool instead of to the figure so we can have different tooltips for each glyph
        p2.add_tools(HoverTool(renderers=[r2, r22], tooltips=tooltips_capital))

        return p2

    def construct_best_odds_figure(self, x, y, z, t, trade_id, sticker):
        # workaround to format date in the hover tool at the moment bokeh does not supported in the tool tips
        source_back_odds = ColumnDataSource(data=dict(x=x, y=y, time=[e.strftime('%d-%m-%Y %H:%M:%S') for e in x]))

        hover = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("x", "@time"),
                ("y", "@y"),
            ],
            active=False
        )

        # create a new plot with a a datetime axis type
        p2 = Figure(plot_width=1000, plot_height=600, title=sticker, toolbar_location="above", x_axis_type="datetime",
                    tools=[hover, 'box_zoom, box_select, crosshair, resize, reset, save,  wheel_zoom'])
        orders = self.map_trade_id_to_orders[trade_id, sticker]
        for order in orders:
            a = [order.placed_dt]
            b = [order.price]
            source_placed = ColumnDataSource(data=dict(a=a, b=b, time=[a[0].strftime('%d-%m-%Y %H:%M:%S')]))
            p2.square(a, b, legend="placed", fill_color="red", line_color="red", size=8, source=source_placed)

        p2.circle(x, y, size=8, color='navy', alpha=0.2, legend="best back odds", source=source_back_odds)
        p2.line(x, y, color='navy', legend="best back odds", source=source_back_odds)

        # lay odds
        source_lay_odds = ColumnDataSource(data=dict(z=z, t=t, time=[e.strftime('%d-%m-%Y %H:%M:%S') for e in z]))
        p2.triangle(z, t, size=8, color='green', alpha=0.2, legend="best lay odds", source=source_lay_odds)
        p2.line(z, t, color='green', legend="best lay odds", source=source_lay_odds)

        # NEW: customize by setting attributes
        # p2.title = sticker
        p2.legend.location = "top_left"
        p2.grid.grid_line_alpha = 0
        p2.xaxis.axis_label = 'Date'
        p2.yaxis.axis_label = "Best odds"
        p2.ygrid.band_fill_color = "olive"
        p2.ygrid.band_fill_alpha = 0.1

        return p2

    def construct_oders_table(self):
        if self.metrics is None:
            raise ValueError("Can't create table of orders without generating metrics")
        list_orders = self.settled_orders
        stickers = []
        sizes = []
        sides = []
        prices = []
        pnls = []
        placed = []
        settled = []
        exec_market_ids = []
        exec_bet_ids = []
        exec_bookmakers = []
        exec_providers = []
        trade_ids = []

        for order in list_orders:
            stickers.append(order.sticker)
            if order.is_back:
                side = 'B'
            else:
                side = 'L'
            sides.append(side)
            sizes.append(order.size)
            prices.append(order.price)
            pnls.append(order.pnl)
            placed.append(order.placed_dt.strftime("%b %d %Y %H:%M:%S"))
            settled.append(order.settled_dt.strftime("%b %d %Y %H:%M:%S"))
            # exec details
            exec_market_ids.append(order.execution_details.get('market_id', ''))
            exec_bet_ids.append(order.execution_details['bet_id'])
            exec_bookmakers.append(order.execution_details['bookmaker'])
            exec_providers.append(order.execution_details['provider'])
            trade_ids.append(order.trade_id)
            self.map_trade_id_to_orders[order.trade_id, order.sticker].append(order)
            self.map_trade_id_to_stickers[order.trade_id].add(order.sticker)

        data = dict(stickers=stickers, sides=sides, sizes=sizes, prices=prices, pnls=pnls, placed=placed,
                    settled=settled,
                    exec_market_ids=exec_market_ids, exec_bet_ids=exec_bet_ids, exec_bookmakers=exec_bookmakers,
                    exec_providers=exec_providers, trade_ids=trade_ids)

        source = ColumnDataSource(data)

        columns = [
            # TableColumn(field="dates", title="Date", formatter=DateFormatter()),
            TableColumn(field="trade_ids", title="TradeID", width=150),
            TableColumn(field="stickers", title="Sticker", width=120),
            TableColumn(field="placed", title="Placed", width=110),
            TableColumn(field="settled", title="Settled", width=110),
            TableColumn(field="sides", title="Side", width=20),
            TableColumn(field="sizes", title="Size", width=40),
            TableColumn(field="prices", title="Price", width=40),
            TableColumn(field="pnls", title="Pnl", width=50, formatter=NumberFormatter(format='0.00')),
            TableColumn(field="exec_market_ids", title="MarketID", width=50),
            TableColumn(field="exec_bet_ids", title="BetID", width=50),
            TableColumn(field="exec_bookmakers", title="Bookmaker", width=30),
            TableColumn(field="exec_providers", title="Providers", width=30),
        ]

        return DataTable(source=source, columns=columns, editable=True, height=280)

    def market_data_stickers_figures_for_trade_id(self, trade_id, stickers_for_trade_id):
        # get all historical odds for the stickers
        # sticker is
        best_odds_figs = []
        for sticker in stickers_for_trade_id:
            sport, market_scope, market, params, _ = parse_sticker(sticker)
            stream = OddsStream(sport, market_scope[1], market, *params,
                                bookmaker=Bookmakers.BETFAIR)

            odds = get_historical_odds([stream])[stream]

            # Calculate best back and lay for a unique sticker
            back_dt = []
            lay_dt = []
            back_odds = []
            lay_odds = []
            for oddsTick in odds:
                if len(oddsTick.back) > 0:
                    back_odds.append(oddsTick.back[0].o)
                    back_dt.append(oddsTick.timestamp)

                if len(oddsTick.lay) > 0:
                    lay_odds.append(oddsTick.lay[0].o)
                    lay_dt.append(oddsTick.timestamp)

            # best odds fig
            best_odds_fig = self.construct_best_odds_figure(back_dt, back_odds, lay_dt, lay_odds, trade_id, sticker)
            best_odds_figs.append(best_odds_fig)

        return best_odds_figs

    def construct_summary_table(self, trading_user):
        trading_user_id = TRADING_USERS[trading_user]
        summary_metrics_map = self.load_summary_metrics(trading_user)
        strategies_names = []
        strategies_names.extend(self.strategies_names)
        strategies_names.extend(self.strategies_descriptions)

        daily_pnls = []
        day_returns_on_capital = []
        day_drawdown = []
        day_volatility = []
        day_sharpe_ratio = []

        weekly_pnls = []
        weekly_returns_on_capital = []
        weekly_drawdown = []
        weekly_volatility = []
        weekly_sharpe_ratio = []

        month_to_date_pnls = []
        month_to_date_returns_on_capital = []
        month_to_date_drawdown = []
        month_to_date_volatility = []
        month_to_date_sharpe_ratio = []

        year_to_date_pnls = []
        year_to_date_returns_on_capital = []
        year_to_date_drawdown = []
        year_to_date_volatility = []
        year_to_date_sharpe_ratio = []

        # strategies
        strategies_name_no_in_data = []
        for strategy_name in strategies_names:
            item = summary_metrics_map.get((trading_user_id, strategy_name), None)
            if item is None:
                strategies_name_no_in_data.append(strategy_name)
                continue
            # create list of data for a column
            daily_metrics = item['metrics']['daily']

            daily_pnls.append(daily_metrics['netPnl'])
            day_returns_on_capital.append(daily_metrics['retOnCapital'])
            day_drawdown.append(daily_metrics['drawDownPct'])
            day_volatility.append(daily_metrics['volatility'])
            day_sharpe_ratio.append(daily_metrics['sharpeRatio'])

            # weekly
            weekly_metrics = item['metrics']['weekly']

            weekly_pnls.append(weekly_metrics['netPnl'])
            weekly_returns_on_capital.append(weekly_metrics['retOnCapital'])
            weekly_drawdown.append(weekly_metrics['drawDownPct'])
            weekly_volatility.append(weekly_metrics['volatility'])
            weekly_sharpe_ratio.append(weekly_metrics['sharpeRatio'])

            # month to date
            month_to_date_metrics = item['metrics']['mtd']

            month_to_date_pnls.append(month_to_date_metrics['netPnl'])
            month_to_date_returns_on_capital.append(month_to_date_metrics['retOnCapital'])
            month_to_date_drawdown.append(month_to_date_metrics['drawDownPct'])
            month_to_date_volatility.append(month_to_date_metrics['volatility'])
            month_to_date_sharpe_ratio.append(month_to_date_metrics['sharpeRatio'])

            # year to date
            year_to_date_metrics = item['metrics']['ytd']

            year_to_date_pnls.append(year_to_date_metrics['netPnl'])
            year_to_date_returns_on_capital.append(year_to_date_metrics['retOnCapital'])
            year_to_date_drawdown.append(year_to_date_metrics['drawDownPct'])
            year_to_date_volatility.append(year_to_date_metrics['volatility'])
            year_to_date_sharpe_ratio.append(year_to_date_metrics['sharpeRatio'])

        strategies_names = [x for x in strategies_names if x not in strategies_name_no_in_data]

        data = dict(strategies_names=strategies_names, daily_pnls=daily_pnls,
                    day_returns_on_capital=day_returns_on_capital,
                    day_drawdown=day_drawdown,
                    day_volatility=day_volatility,
                    day_sharpe_ratio=day_sharpe_ratio,
                    weekly_pnls=weekly_pnls, weekly_returns_on_capital=weekly_returns_on_capital,
                    weekly_drawdown=weekly_drawdown, weekly_volatility=weekly_volatility,
                    weekly_sharpe_ratio=weekly_sharpe_ratio,
                    month_to_date_pnls=month_to_date_pnls,
                    month_to_date_returns_on_capital=month_to_date_returns_on_capital,
                    month_to_date_drawdown=month_to_date_drawdown, month_to_date_volatility=month_to_date_volatility,
                    month_to_date_sharpe_ratio=month_to_date_sharpe_ratio,
                    year_to_date_pnls=year_to_date_pnls,
                    year_to_date_returns_on_capital=year_to_date_returns_on_capital,
                    year_to_date_drawdown=year_to_date_drawdown, year_to_date_volatility=year_to_date_volatility,
                    year_to_date_sharpe_ratio=year_to_date_sharpe_ratio
                    )

        source = ColumnDataSource(data)

        columns = [
            # TableColumn(field="dates", title="Date", formatter=DateFormatter()),
            TableColumn(field="strategies_names", title="Strategy"),
            TableColumn(field="daily_pnls", title="Day PNL " + POUND_SYMBOL,
                        formatter=NumberFormatter(format="0.00"), width=80),
            TableColumn(field="day_returns_on_capital", title="Day Return on Capital (%)",
                        formatter=NumberFormatter(format='0.00'), width=160),
            TableColumn(field="day_drawdown", title="Day Drawdown", formatter=NumberFormatter(format='0.00')),
            TableColumn(field="day_volatility", title="Day Volatility", formatter=NumberFormatter(format='0.00')),
            TableColumn(field="day_sharpe_ratio", title="Day Sharpe Ratio", formatter=NumberFormatter(format='0.00')),

            TableColumn(field="weekly_pnls", title="Weekly PNL " + POUND_SYMBOL, formatter=NumberFormatter(format='0.00'),
                        width=100),
            TableColumn(field="weekly_returns_on_capital", title="Weekly Return on Capital (%)",
                        formatter=NumberFormatter(format='0.00'), width=180),
            TableColumn(field="weekly_drawdown", title="Weekly Drawdown", formatter=NumberFormatter(format='0.00')),
            TableColumn(field="weekly_volatility", title="Weekly Volatility", formatter=NumberFormatter(format='0.00')),
            TableColumn(field="weekly_sharpe_ratio", title="Weekly Sharpe Ratio", formatter=NumberFormatter(
                format='0.00')),

            TableColumn(field="month_to_date_pnls", title="MTD PNL " + POUND_SYMBOL,
                        formatter=NumberFormatter(format='0.00'), width=85),
            TableColumn(field="month_to_date_returns_on_capital", title="MTD Return on Capital (%)",
                        formatter=NumberFormatter(format='0.00'), width=160),
            TableColumn(field="month_to_date_drawdown", title="MTD Drawdown", formatter=NumberFormatter(format='0.00')),
            TableColumn(field="month_to_date_volatility", title="MTD Volatility", formatter=NumberFormatter(
                format='0.00')),
            TableColumn(field="month_to_date_sharpe_ratio", title="MTD Sharpe Ratio", formatter=NumberFormatter(
                format='0.00')),

            TableColumn(field="year_to_date_pnls", title="YTD PNL " + POUND_SYMBOL, formatter=NumberFormatter(format='0.00'),
                        width=80),
            TableColumn(field="year_to_date_returns_on_capital", title="YTD Return on Capital (%)",
                        formatter=NumberFormatter(format='0.00'), width=160),
            TableColumn(field="year_to_date_drawdown", title="YTD Drawdown", formatter=NumberFormatter(format='0.00')),
            TableColumn(field="year_to_date_volatility", title="YTD Volatility", formatter=NumberFormatter(
                format='0.00')),
            TableColumn(field="year_to_date_sharpe_ratio", title="YTD Sharpe Ratio", formatter=NumberFormatter(
                format='0.00')),
        ]

        return DataTable(width=1300, source=source, columns=columns, fit_columns=True)


def main():
    from stratagemdataprocessing.parsing.common.stickers import extract_event
    monitor = MonitoringMetrics()

    # populate the dictionaries that are used as loading the data in the dropdowns
    output_file("dashboard.html", title="dashboard")

    trading_user = 'stratagem'
    trading_user_id = TRADING_USERS[trading_user]

    strategy = 'tennis_sip'
    strategy_descr = None

    # style = StrategyStyle.to_str(StrategyStyle.INPLAY)
    style = monitor.strategy_style[strategy]
    start_dt = datetime(2016, 7, 15, 0, 0, 0)
    end_dt = datetime(2016, 7, 20, 0, 0, 0)
    sport = monitor.sport_strategies_map[strategy]
    Sports.to_str(sport)

    # SUMMARY METRICS "
    metrics_table = monitor.construct_summary_table(trading_user)
    show(metrics_table)

    success = monitor.calculate_metrics(strategy, strategy_descr, start_dt, end_dt, sport, style, trading_user)
    if not success:
        print "no orders"
        exit(-1)

    # total compound
    x_date, y_total_compound = monitor.get_total_compound()
    print "date=", x_date
    print "total compound=", y_total_compound
    exp_cumsum_log_return = np.expm1(y_total_compound)
    compound_fig = monitor.construct_figure(x_date, exp_cumsum_log_return, 'Compound Return (%)')

    # total pnl
    x_date, y_total_pnl = monitor.get_total_pnl()
    x_capital_date, y_capital = monitor.get_capital_by_day(x_date)

    print "pnl=", y_total_pnl
    print "capital=", y_capital
    total_pnl_fig = monitor.construct_total_pnl_figure(x_date, y_total_pnl, y_capital)

    # table of orders
    data_table = monitor.construct_oders_table()

    # supposing we have a trade_id
    trade_id = '57091add30b3743fa87dcd22'
    stickers_for_trade_id = monitor.map_trade_id_to_stickers[trade_id]
    best_odds_figs = monitor.market_data_stickers_figures_for_trade_id(trade_id, stickers_for_trade_id)

    # print stickers information
    for sticker in stickers_for_trade_id:
        print "STICKER = ", sticker
        orders = monitor.map_trade_id_to_orders[trade_id, sticker]
        for order in orders:
            event = extract_event(order.sticker)
            print "bet id: %s, event id: %s, placed dt: %s, settled dt: %s, size: %s, price: %s, pnl: %s" % (
                order.execution_details['bet_id'], event, order.placed_dt, order.settled_dt, order.size,
                order.price, order.pnl)

    # plot stickers
    show(column(compound_fig, total_pnl_fig, data_table, metrics_table, *best_odds_figs))


if __name__ == '__main__':
    main()
