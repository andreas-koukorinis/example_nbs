from collections import defaultdict, OrderedDict
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
import pytz
import re
import numpy as np
from bson.objectid import ObjectId

from bokeh.io import output_file, gridplot, show
from bokeh.models.layouts import Column
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, TapTool, CustomJS, SaveTool
from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter, Select, Tabs, Panel
from bokeh.palettes import Spectral8
from bokeh.models.formatters import DatetimeTickFormatter

from sgmtradingcore.backtesting.persistence import load_optimization_results, load_optimization
from sgmtradingcore.analytics.metrics import flat_capital_traces, cumulative_capital_traces, flat_capital_metrics
from stratagemdataprocessing.dbutils.mongo import MongoPersister
from sgmtradingcore.analytics.performance import apply_betfair_commission
from sgmtradingcore.analytics.parameters import MultiParamSet
from sgmtradingcore.backtesting.scoring import Scoring


def connect_to_mongo(database):
    mongo_client = MongoPersister.init_from_config(database, auto_connect=True)
    return mongo_client

def convert_backtest_orders(orders):
    result = []

    for order in orders:
        result.append({
            'dt': order.placed_dt,
            'stake': order.size if order.is_back else order.size / order.odds,
            'pnl': order.pnl,
        })

    return result


def get_backtest_results(conn, name):
    results = load_optimization_results(conn, name)

    return results


def plot_timeseries_return(mongo_connection, optimization_name, top=10,
                           param_filter_fn=None, flat_staking=True, config_ids=None):

    pipeline = [{'$match': {'name': optimization_name}},
                {'$sort': {'updated_dt': -1}},
                {'$limit': 1},
                {'$unwind': '$optimization'},
                {'$project': {'config_id': '$optimization.config_id',
                              'start_date': '$optimization.start_date',
                              'end_date': '$optimization.end_date',
                              'score': '$optimization.score',
                              'metrics': '$optimization.metrics'}}]
    if config_ids is not None:
        pipeline.append({'$match': {'config_id': {'$in': [ObjectId(cfg) for cfg in config_ids]}}})
        top = len(config_ids)
    pipeline.append({'$sort': {'score': -1}})

    optimization_runs = mongo_connection['backtest_optimization'].aggregate(pipeline)

    seen_configs = []
    count = 0
    plot_data = []
    all_metrics = []
    for run in optimization_runs:
        start_date = run['start_date']
        end_date = run['end_date']
        config_id = run['config_id']
        if config_id in seen_configs:
            continue
        seen_configs.append(config_id)

        params = mongo_connection['backtest_configurations'].find_one({'_id': run['config_id']})

        if param_filter_fn is not None:
            if not param_filter_fn(params):
                continue

        results = mongo_connection['backtest_results'].find({'config_id': config_id,
                                                         'date': {'$lte': end_date, '$gte': start_date}})
        if results.count() > 0 and params is not None:
            dict_result = []

            for i in results:
                for o in i['orders']:
                    dict_result.append({'date': i['date'], 'pnl': o.get('pnl', np.nan), 'sticker': o['sticker'],
                                        'is_back': o['is_back'], 'stake': o['size'], 'odds': o['odds']})

            tbl = pd.DataFrame(dict_result)
            tbl['dt'] = tbl['date'].map(
                        lambda x: datetime.combine(datetime.strptime(x, '%Y%m%d'), datetime.min.time()))

            starting_capital = 10000
            if flat_staking:
                tbl.loc[:, 'capital'] = starting_capital
                timeseries = flat_capital_traces(tbl)
                data = pd.DataFrame({'config': str(config_id),
                                     'return': timeseries['cum_return'],
                                     'date': list(timeseries.index)})
            else:
                # Accumulate the capital
                cap_df = tbl.groupby('dt').pnl.sum().to_frame().rename(columns={'pnl': 'capital'})
                cap_df = cap_df.reset_index().sort('dt')
                cap_df.loc[:, 'capital'] = starting_capital + np.concatenate([[0.], cap_df.capital.cumsum().values[:-1]])
                tbl = tbl.merge(cap_df)

                timeseries = cumulative_capital_traces(tbl, starting_capital)
                data = pd.DataFrame({'config': str(config_id),
                                     'return': timeseries['cum_ret'],
                                     'date': list(timeseries.index)})
            plot_data.append(data)

            met_df = flat_capital_metrics(tbl.dropna(subset=['pnl']))
            met_df = met_df.rename(columns={'unitary_stake_return': 'unit_return', 'average_trade_win': 'avg_trade_win',
                            'average_trade_loss': 'avg_trade_loss', 'cum_return': 'cum_ret',
                            'volatility (not annualised)': 'vol_not_ann',
                            'sharpe_ratio': 'sharpe_ratio',
                            'maximum_drawdown': 'max_drawdown', 'drawdown_duration (days)': 'drawdown_days',
                            'maximum_runup': 'max_runup', 'runup_duration (days)': 'runup_days'})
            met_df.loc[:, 'config'] = str(run['config_id'])
            all_metrics.append(met_df)

            count += 1
            if count >= top:
                break

    if count == 0:
        raise Exception('Mongo cursor empty, no data were found')

    plot_data = pd.concat(plot_data)
    all_metrics = pd.concat(all_metrics)
    plot_data = plot_data.merge(all_metrics, how='left', on='config')

    # Create data sources for plotting
    configs = list(set(plot_data.config))
    plot_data.loc[:, 'colour'] = 'grey'
    plot_data.loc[:, 'alpha'] = 0.5
    plot_data.loc[:, 'size'] = 5
    s = ColumnDataSource(data=plot_data)
    s2 = ColumnDataSource({'config': configs,
                           'date': [plot_data.date.loc[plot_data.config == cfg].values for cfg in configs],
                           'return': [plot_data['return'].loc[plot_data.config == cfg].values for cfg in configs],
                           'colour': ['grey']*len(configs),
                           'alpha': [0.0]*len(configs)
                          })

    # Add plots
    p1 = figure(plot_width=900, plot_height=400, title="Cumulative returns",
                tools=[TapTool(), SaveTool()], x_axis_type='datetime')
    p1.circle('date', 'return', source=s, color='colour', alpha='alpha', size='size')
    p1.multi_line(xs='date', ys='return', source=s2, color='colour', alpha='alpha')
    p1.yaxis.axis_label = 'Return'

    # Metrics
    metric_formats = OrderedDict([('n_trades', '0'), ('n_win', '0'), ('n_loss', '0'), ('hit_ratio', '0.000'),
                                  ('avg_trade_win', '0.0000'), ('avg_trade_loss', '0.0000'),
                                  ('unit_return', '0.000'), ('total_pnl', '0.00'),
                                  ('cr_trade', '0.000'), ('cr_day', '0.000'), ('cum_ret', '0.000'),
                                  ('vol_not_ann', '0.000'), ('cum_ret/vol', '0.000'),
                                  ('max_drawdown', '0.000'), ('drawdown_days', '0'), ('max_runup', '0.000'),
                                  ('runup_days', '0')])
    cols = ['config'] + metric_formats.keys()

    # Add table of metrics
    s3 = ColumnDataSource(all_metrics.iloc[[]])
    tbl_columns = [TableColumn(field='config', title='config', width=100)]
    tbl_columns += [TableColumn(field=mm, title=mm, width=100, formatter=NumberFormatter(format=fmt))
                    for mm, fmt in metric_formats.iteritems()]
    data_table = DataTable(source=s3, columns=tbl_columns, width=900, height=200, fit_columns=False)

    cb_code = '''
        var cfgs = [];
        var ind = cb_obj.get('selected')['1d'].indices;
        var d = cb_obj.get('data');
        var d2 = s2.get('data');
        var d3 = s3.get('data');
        var cols = __COLS__;
        for (i = 0; i < d['config'].length; i++){
            if ((ind.indexOf(i) != -1) && (cfgs.indexOf(d['config'][i]) == -1)){
                cfgs.push(d['config'][i]);
            }
        }
        for (i = 0; i < d['config'].length; i++){
            if (cfgs.indexOf(d['config'][i]) == -1){
                d['size'][i] = 5;
                d['colour'][i] = "grey";
                d['alpha'][i] = 0.1;
            } else {
                d['size'][i] = 10;
                d['colour'][i] = "blue";
                d['alpha'][i] = 0.9;
            }
        }
        for (i = 0; i < d2['config'].length; i++){
            if (cfgs.indexOf(d2['config'][i]) == -1){
                d2['colour'][i] = "grey";
                d2['alpha'][i] = 0.0;
            } else {
                d2['colour'][i] = "blue";
                d2['alpha'][i] = 0.9;
            }
        }
        for (j = 0; j < cols.length; j++){
            d3[cols[j]] = [];
        }
        var cfgs_tbl = [];
        for (i = 0; i < d['config'].length; i++){
            if ((cfgs.indexOf(d['config'][i]) != -1) && (cfgs_tbl.indexOf(d['config'][i]) == -1)){
                cfgs_tbl.push(d['config'][i]);
                for (j = 0; j < cols.length; j++){
                    d3[cols[j]].push(d[cols[j]][i])
                }
            }
        }
        s2.trigger('change');
        s3.trigger('change');
        dt.trigger('change');
    '''
    cb_code = cb_code.replace('__COLS__', '["' + '","'.join(cols) + '"]')
    s.callback = CustomJS(args={'s2': s2, 's3': s3, 'dt': data_table}, code=cb_code)

    return Column(p1, data_table)


def create_plot_and_table_from_bets(bets_df, flat_staking=True, run_label=None, config_id=None,
                                    run_metrics=None, net=True, starting_capital=10000):
    if net:
        prog = re.compile('^[ST]-E([A-Za-z0-9]+)')
        bets_df.loc[:, 'event'] = [prog.match(s).group(1) for s in bets_df.sticker]
        prog = re.compile('^[ST]-E([A-Za-z0-9]+)-([A-Za-z0-9]+)')
        bets_df.loc[:, 'market'] = [prog.match(s).group(2) for s in bets_df.sticker]
        bets_df = pd.concat(apply_betfair_commission([row for _, row in bets_df.iterrows()]), axis=1).transpose()
        bets_df['pnl'] -= bets_df['commission']

    if flat_staking:
        bets_df.loc[:, 'capital'] = starting_capital
        timeseries = flat_capital_traces(bets_df)
        data = pd.DataFrame({'return': timeseries['cum_return'],
                             'date': list(timeseries.index)})
    else:
        # Accumulate the capital
        cap_df = bets_df.groupby('dt').pnl.sum().to_frame().rename(columns={'pnl': 'capital'})
        cap_df = cap_df.reset_index().sort('dt')
        cap_df.loc[:, 'capital'] = starting_capital + np.concatenate([[0.], cap_df.capital.cumsum().values[:-1]])
        bets_df = bets_df.merge(cap_df)

        bets_df.loc[:, 'return'] = bets_df.pnl
        timeseries = cumulative_capital_traces(bets_df, starting_capital)
        data = pd.DataFrame({'return': timeseries['cum_ret'],
                             'date': list(timeseries.index)})

    # Create the time series plot
    s = ColumnDataSource(data=data)
    title = "Cumulative returns"
    if run_label is not None and config_id is not None:
        title += " - Run {} ({})".format(run_label, config_id)
    p = figure(plot_width=900, plot_height=300,
               title=title, tools=[TapTool(), SaveTool()], x_axis_type='datetime')
    p.circle('date', 'return', source=s, color='dodgerblue', size=8)
    p.line('date', 'return', source=s, color='dodgerblue', line_width=1)
    p.yaxis.axis_label = 'Return'

    wrong_odds = (bets_df.odds <= 1.) & (abs(bets_df.pnl) >= 0.01)
    if wrong_odds.any():
        wrong_odds_back = wrong_odds & (bets_df.is_back)
        wrong_odds_lay = wrong_odds & (~bets_df.is_back)
        print 'Warning: {}(b)/{}(l) Odds <= 1 in config {}'.format(wrong_odds_back.sum(), wrong_odds_lay.sum(),
                                                                   str(config_id))
        bets_df.loc[wrong_odds_back, 'odds'] = 1.
        bets_df = bets_df.loc[~wrong_odds_lay, :]

    if not wrong_odds.all():
        bets_df = bets_df.loc[(bets_df.odds > 1) | (abs(bets_df.pnl) >= 0.01), :]
        met_df = flat_capital_metrics(bets_df.dropna(subset=['pnl']))
        if met_df is not None:
            met_df = met_df.rename(columns={'unitary_stake_return': 'unit_return', 'average_trade_win': 'avg_trade_win',
                                            'average_trade_loss': 'avg_trade_loss', 'cum_return': 'cum_ret',
                                            'volatility (not annualised)': 'vol_not_ann',
                                            'sharpe_ratio': 'sharpe_ratio',
                                            'maximum_drawdown': 'max_drawdown',
                                            'drawdown_duration (days)': 'drawdown_days',
                                            'maximum_runup': 'max_runup', 'runup_duration (days)': 'runup_days'})
            if run_metrics is not None:
                met_df['time_in_market'] = run_metrics.get('time_in_market', np.nan)
                met_df['rina'] = run_metrics.get('rina', np.nan)
                met_df.loc[:, 'run_label'] = run_label
    else:
        met_df = pd.DataFrame(run_metrics, index=[0]).rename(columns={'cum_return': 'cum_ret',
                                                                      'drawdown': 'max_drawdown',
                                                                      'volatility': 'vol_not_ann'})
        met_df.loc[:, 'sharpe_ratio'] = met_df.cum_ret / met_df.vol_not_ann
        met_df.loc[:, 'run_label'] = run_label

    return p, met_df


def metrics_to_table(metrics_df):
    metric_formats = OrderedDict([('n_trades', '0'), ('n_win', '0'), ('n_loss', '0'), ('hit_ratio', '0.000'),
                                  ('avg_trade_win', '0.0000'), ('avg_trade_loss', '0.0000'),
                                  ('unit_return', '0.000'), ('total_pnl', '0.00'),
                                  ('cr_trade', '0.000'), ('cr_day', '0.000'), ('cum_ret', '0.000'),
                                  ('vol_not_ann', '0.000'), ('sharpe_ratio', '0.000'),
                                  ('max_drawdown', '0.000'), ('drawdown_days', '0'), ('max_runup', '0.000'),
                                  ('runup_days', '0'), ('time_in_market', '0.000'), ('rina', '0.000')])

    # Add table of metrics
    s = ColumnDataSource(metrics_df)
    tbl_columns = [TableColumn(field='run_label', title='run', width=100)]
    tbl_columns += [TableColumn(field=mm, title=mm, width=100, formatter=NumberFormatter(format=fmt))
                    for mm, fmt in metric_formats.iteritems() if mm in metrics_df.columns]
    data_table = DataTable(source=s, columns=tbl_columns, width=900, height=200, fit_columns=False)
    return data_table


def plot_metrics_and_returns_single_config(mongo_connection, config_id, opt_name, run_label,
                                           net=True, flat_staking=True):
    params = mongo_connection['backtest_configurations'].find_one({'_id': config_id})

    query = [
        {'$match': {'name': opt_name}},
        {'$unwind': '$optimization'},
        {'$match': {'optimization.config_id': config_id}},
        {'$project': {'metrics': '$optimization.metrics', '_id': 0}}
    ]
    run_metrics = list(mongo_connection['backtest_optimization'].aggregate(query))[0]
    run_metrics = run_metrics.get('metrics', {})

    query = {'config_id': config_id, 'optimization_name': opt_name}
    results = mongo_connection['backtest_results'].find(query)
    if results.count() > 0 and params is not None:
        dict_result = []

        for i in results:
            for o in i['orders']:
                dict_result.append({'date': i['date'], 'pnl': o.get('pnl', np.nan), 'sticker': o['sticker'],
                                    'is_back': o['is_back'], 'stake': o['size'], 'odds': o['odds'],
                                    'is_close': o['details']['sizing']['is_close']})

        tbl = pd.DataFrame(dict_result)
        tbl['dt'] = tbl['date'].map(
            lambda x: datetime.combine(datetime.strptime(x, '%Y%m%d'), datetime.min.time()))

        p, met_df = create_plot_and_table_from_bets(tbl, flat_staking=flat_staking, run_label=run_label,
                                                    config_id=config_id, run_metrics=run_metrics)

        return p, met_df, params


def plot_metrics_and_returns(mongo_connection, study_name, merged_config,
                             flat_staking=True, start_date=None, end_date=None, net=True):

    metrics_df = []
    plots = []
    for run_label in sorted(merged_config.keys()):
        opt_name = '{}_{}'.format(study_name, run_label)

        config_id = merged_config[run_label]
        if isinstance(config_id, basestring):
            config_id = ObjectId(config_id)

        p, met_df, params = plot_metrics_and_returns_single_config(mongo_connection, config_id, opt_name, run_label,
                                                           net=net, flat_staking=flat_staking)
        plots.append(p)
        metrics_df.append(met_df)

    if len(plots) == 0:
        raise Exception('No results found')

    # Add metrics table
    metric_formats = OrderedDict([('n_trades', '0'), ('n_win', '0'), ('n_loss', '0'), ('hit_ratio', '0.000'),
                                  ('avg_trade_win', '0.0000'), ('avg_trade_loss', '0.0000'),
                                  ('unit_return', '0.000'), ('total_pnl', '0.00'),
                                  ('cr_trade', '0.000'), ('cr_day', '0.000'), ('cum_ret', '0.000'),
                                  ('vol_not_ann', '0.000'), ('sharpe_ratio', '0.000'),
                                  ('max_drawdown', '0.000'), ('drawdown_days', '0'), ('max_runup', '0.000'),
                                  ('runup_days', '0'), ('time_in_market', '0.000'), ('rina', '0.000')])
    metrics_df = pd.concat(metrics_df)

    # Add table of metrics
    s = ColumnDataSource(metrics_df)
    tbl_columns = [TableColumn(field='run_label', title='run', width=100)]
    tbl_columns += [TableColumn(field=mm, title=mm, width=100, formatter=NumberFormatter(format=fmt))
                    for mm, fmt in metric_formats.iteritems() if mm in metrics_df.columns]
    data_table = DataTable(source=s, columns=tbl_columns, width=900, height=200, fit_columns=False)
    plots.append(data_table)

    # Add table of parameters
    s = ColumnDataSource(pd.DataFrame(params['variable_params'], index=[0]))
    tbl_columns = [TableColumn(field=cc, title=cc, width=100)
                   for cc in sorted(params['variable_params'].keys()) if cc != 'score']
    data_table = DataTable(source=s, columns=tbl_columns, width=900, height=100, fit_columns=False)
    plots.append(data_table)

    return Column(*plots)


def plot_multiple_metrics_and_returns(mongo_connection, study_name, merged_configs, **kw):
    tabs = [Panel(child=plot_metrics_and_returns(mongo_connection, study_name, mc, **kw), title=nn)
            for nn, mc in merged_configs.iteritems()]
    return Tabs(tabs=tabs)


def get_parameter_sets(mongo_connection, optimization_name, start_date, end_date,
                       number_runs=None, score_fn=None, min_score=-np.inf):

    optimization_runs = \
        mongo_connection['backtest_optimization'].find({'name': optimization_name,
                                                        'optimization': { '$elemMatch':
                                                                              {'start_date': {'$gte': start_date},
                                                                               'end_date': {'$lte': end_date}}
                                                                          }
                                                        }, {'optimization': 1})
    if optimization_runs.count() == 0:
        raise Exception('Mongo cursor empty, no data were found')

    optimization_run = optimization_runs[0]['optimization']
    if number_runs:
        optimization_run = sorted(optimization_run, key = lambda run: run['score'], reverse=True)[0:number_runs]
    else:
        optimization_run = sorted(optimization_run, key = lambda run: run['score'], reverse=True)

    grouped_runs = defaultdict(list)

    for run in optimization_run:
        start_date = run['start_date']
        end_date = run['end_date']
        grouped_runs[(start_date, end_date)].append(run)

    grouped_param_sets = {}
    for key in grouped_runs:

        data_static = []
        data_variable = []
        metrics = []
        configs = []
        gen = []
        for run in grouped_runs[key]:
            params = mongo_connection['backtest_configurations'].find_one({'_id': run['config_id']})
            if params:
                variable_param_values = params['variable_params']
                static_param_values = params['static_params']

                variable_param_values.update({'score': run['score']})

                data_variable.append(variable_param_values)
                configs.append(str(run['config_id']))
                metrics.append(run['metrics'] if run['metrics'] is not None else {})
                gg = run.get('optimization_params', {}).get('generation', -1)
                if gg < 0:
                    gg = run.get('optimization_params', {}).get('iteration', -1)
                gen.append(gg)

        static_param_values.update({'start_date': key[0], 'end_date': key[1]})
        data_static.append(static_param_values)

        data_static = pd.DataFrame.from_records(data_static)
        data_variable = pd.DataFrame.from_records(data_variable, index=configs).drop_duplicates()
        metrics = pd.DataFrame.from_records(metrics)
        metrics.loc[:, 'config'] = configs
        metrics = metrics.drop_duplicates().set_index('config')

        with pd.option_context('mode.use_inf_as_null', True):
            data_variable = data_variable.dropna(subset=['score'])
        data_variable = data_variable.fillna(-999)

        if score_fn is not None:
            metrics.loc[:, 'n_trades'] = metrics.n_win + metrics.n_loss
            metrics.loc[:, 'sharpe_ratio'] = metrics['cum_return'] / metrics['volatility']
            metrics.loc[:, 'maximum_drawdown'] = metrics['drawdown']
            score_df = metrics.apply(score_fn, axis=1).to_frame()
            score_df.columns = ['score']
            del data_variable['score']
            data_variable = data_variable.merge(score_df, left_index=True, right_index=True, how='left')

        data_variable.loc[:, 'score'] = np.maximum(data_variable.score, min_score)

        grouped_param_sets[key] = MultiParamSet(data_variable, aux_df=metrics, n_top_clusters=5,
                                                cluster_by_score=[0.5, 0.8, 1.0], max_clusters=10, gen=gen)

    return grouped_param_sets


def param_sets_from_configs(config_ids, mongo_client=None):
    connect_here = mongo_client is None
    if connect_here:
        mongo_client = MongoPersister.init_from_config('trading_dev', auto_connect=True)
        mongo_client.connect()

    # Get the parameter sets from the configuration IDs
    param_sets = {}
    for cfg_id in config_ids:
        params = mongo_client['backtest_configurations'].find_one({'_id': ObjectId(cfg_id)})
        if params:
            param_values = params['variable_params']
            param_sets[cfg_id] = param_values

    if connect_here:
        mongo_client.close()

    return param_sets


def backtest_run_report(bets_df, starting_capital=10000, net=True, flat_staking=True):

    bkt_names = sorted(list(set(bets_df.name)))
    nbkt = len(bkt_names)

    bets_df = bets_df.rename(columns={'date': 'dt'})
    bets_df.loc[:, 'dt'] = bets_df.dt.map(parser.parse).map(lambda dt: dt.replace(tzinfo=None))
    bets_df.loc[:, 'fixture_id'] = bets_df.event

    if net:
        # Add commission
        bets_out = []
        for _, bdf in bets_df.groupby('name'):
            bets = apply_betfair_commission([row for _, row in bdf.iterrows()])
            bets_out.append(pd.DataFrame(bets))
        bets_df = pd.concat(bets_out)
        bets_df.loc[:, 'pnl'] = bets_df.pnl - bets_df.commission

    # Get metrics
    # TODO: Need to add capital to data frame here to be accurate
    cm = flat_capital_metrics(bets_df, groupby='name', capital=starting_capital)
    cm = cm.rename(columns={'unitary_stake_return': 'unit_return', 'average_trade_win': 'avg_trade_win',
                            'average_trade_loss': 'avg_trade_loss', 'cum_return %': 'cum_ret%',
                            'volatility (not annualised) %': 'vol_not_ann%',
                            'sharpe_ratio': 'sharpe_ratio',
                            'maximum_drawdown %': 'max_drawdown%', 'drawdown_duration (days)': 'drawdown_days',
                            'maximum_runup %': 'max_runup%', 'runup_duration (days)': 'runup_days'})

    metric_formats = [OrderedDict([('n_trades', '0'), ('n_win', '0'), ('n_loss', '0'), ('hit_ratio', '0.000')]),
                      OrderedDict([('avg_trade_win', '0.0000'), ('avg_trade_loss', '0.0000'),
                                   ('unit_return', '0.000'), ('total_pnl', '0.00')]),
                      OrderedDict([('cr_trade', '0.000'), ('cr_day', '0.000'), ('cum_ret%', '0.0'),
                                   ('vol_not_ann%', '0.0'), ('cum_ret/vol', '0.000')]),
                      OrderedDict([('max_drawdown%', '0.0'), ('drawdown_days', '0'), ('max_runup%', '0.0'),
                                   ('runup_days', '0')])]
    tbls = []
    for mf in metric_formats:
        cm_page = cm.loc[:, ['name'] + mf.keys()]
        s = ColumnDataSource(cm_page)
        cols = [TableColumn(field=cc, title=cc, formatter=NumberFormatter(format=fmt), width=150)
                for cc, fmt in mf.iteritems()]
        cols.insert(0, TableColumn(field='name', title='bkt_name', width=100))
        tbls.append(DataTable(source=s, columns=cols, row_headers=True, height=30*(nbkt + 1), width=1000))
    metrics_tbl = Column(*tbls)

    # Get capital trace
    if flat_staking:
        ct = flat_capital_traces(bets_df, groupby='name', capital=starting_capital)
        ct = ct.rename(columns={'cum_return %': 'cum_ret'})
    else:
        bets_df.loc[:, 'return'] = bets_df.pnl
        ct = cumulative_capital_traces(bets_df, starting_capital, groupby='name')

    # Add initial value
    start_date = min(ct.index.get_level_values('date')) - timedelta(days=1)
    start_df = pd.DataFrame({'name': bkt_names, 'date': start_date,
                             'cum_ret': 0.}).set_index('name')
    ct = pd.concat([start_df, ct.reset_index('date')]).sort('date')

    ct2 = ct.set_index('date', append=True).unstack('name').fillna(method='ffill')
    ct2.columns = ct2.columns.droplevel(0)
    s2 = ColumnDataSource(ct2)

    ct.loc[:, 'colour'] = [Spectral8[bkt_names.index(nn)] for nn in ct.index.get_level_values('name')]
    s1 = ColumnDataSource(ct.reset_index())
    s1.data['date_str'] = [dt.strftime('%m/%d') for dt in s1.data['date']]

    hover = HoverTool(tooltips=[('backtest', '@name'),
                                ('date', '@date_str'),
                                ('cum. return %', '@cum_ret{0.0}')])

    p = figure(height=400, width=900, x_axis_type='datetime', tools=[hover, SaveTool()])
    for ii, bkt in enumerate(bkt_names):
        p.line(x='date', y=bkt, color=Spectral8[ii], legend=bkt, source=s2, line_width=2)
    p.circle(x='date', y='cum_ret', color='colour', source=s1, size=10)
    p.legend.location = 'bottom_left'
    #p.legend.label_text_font_size = '6pts'
    p.xaxis.formatter = DatetimeTickFormatter(formats={'days': ['%d %b'], 'months': ['%b %Y']})
    p.yaxis.axis_label = 'Cumulative return %'
    p.yaxis.axis_label_text_font_size = '20pts'

    return metrics_tbl, p


# Function to get merged configurations from different runs (matched by the variable parameters)
def get_merged_configurations(mongo_connection, study_name, min_score=-np.inf):
    params_df = []
    run_labels = []
    fold = 0
    done = False
    scores = {}
    while not done:
        fold_study_name = study_name + '_{}'.format(fold)
        try:
            optimization = load_optimization(mongo_connection, fold_study_name, return_none=False)
        except Exception:
            done = True
            continue

        run_lab = str(fold)
        run_labels.append(run_lab)
        config_ids = [str(run.strategy_config_id) for run in optimization.optimization_runs]
        pdf = param_sets_from_configs(config_ids)
        pdf = pd.DataFrame(pdf.values(), index=pdf.keys())
        pdf.loc[:, 'study'] = run_lab
        pdf = pdf.set_index('study', append=True)
        params_df.append(pdf)

        scores.update({(str(run.strategy_config_id), run_lab): max(run.score, min_score)
                       for run in optimization.optimization_runs})

        fold += 1

    oos_study_name = study_name + '_OOS'
    run_num = 0
    done = False
    while not done:
        run_oos_study_name = oos_study_name + '_{}'.format(run_num)
        try:
            optimization = load_optimization(mongo_connection, run_oos_study_name, return_none=False)
        except ValueError:
            done = True
            continue

        run_lab = 'OOS_{}'.format(run_num)
        run_labels.append(run_lab)
        config_ids = [str(run.strategy_config_id) for run in optimization.optimization_runs]
        pdf = param_sets_from_configs(config_ids)
        pdf = pd.DataFrame(pdf.values(), index=pdf.keys())
        pdf.loc[:, 'study'] = run_lab
        pdf = pdf.set_index('study', append=True)
        params_df.append(pdf)

        scores.update({(str(run.strategy_config_id), run_lab): max(run.score, min_score)
                       for run in optimization.optimization_runs})

        run_num += 1

    params_df = pd.concat(params_df)
    try:
        del params_df['score']
    except KeyError:
        pass
    params_df = params_df.fillna(-999.)

    merged_configs = [dict(d.index.swaplevel(0, 1).tolist())
                      for _, d in params_df.groupby(params_df.columns.tolist())
                      if len(d) == len(run_labels)]

    mc_scores = [[scores[(mc[rl], rl)] for rl in run_labels] for mc in merged_configs]
    x = sorted(zip(merged_configs, mc_scores), cmp=lambda a, b: cmp(np.mean(b[1]), np.mean(a[1])))
    merged_configs, mc_scores = zip(*x)

    fmt = '(' + ', '.join(['{:.2f}']*len(run_labels)) + ')'
    mc_names = ['{}. '.format(ii + 1) + fmt.format(*mcs) for ii, (mc, mcs) in enumerate(zip(merged_configs, mc_scores))]
    merged_configs = OrderedDict([(k, v) for k, v in zip(mc_names, merged_configs)])

    return merged_configs


def plot_scoring_test(mongo_connection, study_name):
    kw = {'maximum_drawdown': ([-0.05, -0.2, 25.], {'truncate_left': True, 'truncate_right': True, 'control': -2.0}),
          'cum_return': ([0.2, 0.0, 25.], {'truncate_left': True, 'truncate_right': True, 'control': -2.0}),
          'sharpe_ratio': ([3., 0., 50.], {'truncate_left': True, 'truncate_right': True, 'control': 0.0})}

    score_fn = lambda m: Scoring.calculate_general_score(m, **kw)
    run_label = '0'
    opt_name = '{}_{}'.format(study_name, run_label)
    x = get_parameter_sets(mongo_connection, opt_name, '20150101', '20161001', score_fn=score_fn)
    mps = x.values()[0]
    mps._n_top_clusters = 20
    mps._cluster_by_score = [0.0, 0.2, 0.5, 0.8, 1.0]

    tabs = []
    z = sorted(zip(mps.selected, mps.selected_indexes()), cmp=lambda a, b: cmp(mps.scores[b[0]], mps.scores[a[0]]))
    for idx, cfg in z:
        p, met_df, _ = plot_metrics_and_returns_single_config(mongo_connection, ObjectId(cfg), opt_name, run_label,
                                                              net=True, flat_staking=False)
        # Add metrics table
        metric_formats = OrderedDict([('n_trades', '0'), ('n_win', '0'), ('n_loss', '0'), ('hit_ratio', '0.000'),
                                      ('avg_trade_win', '0.0000'), ('avg_trade_loss', '0.0000'),
                                      ('unit_return', '0.000'), ('total_pnl', '0.00'),
                                      ('cr_trade', '0.000'), ('cr_day', '0.000'), ('cum_ret', '0.000'),
                                      ('vol_not_ann', '0.000'), ('sharpe_ratio', '0.000'),
                                      ('max_drawdown', '0.000'), ('drawdown_days', '0'), ('max_runup', '0.000'),
                                      ('runup_days', '0'), ('time_in_market', '0.000'), ('rina', '0.000')])
        # Add table of metrics
        s = ColumnDataSource(met_df)
        tbl_columns = [TableColumn(field='run_label', title='run', width=100)]
        tbl_columns += [TableColumn(field=mm, title=mm, width=100, formatter=NumberFormatter(format=fmt))
                        for mm, fmt in metric_formats.iteritems() if mm in met_df.columns]
        data_table = DataTable(source=s, columns=tbl_columns, width=900, height=200, fit_columns=False)

        tabs.append(Panel(child=Column(p, data_table), title='Score={:.1f}'.format(mps.scores[idx])))

    show(Tabs(tabs=tabs))


if __name__ == '__main__':
    output_file('/tmp/presentation.html')

    mongo_connection = connect_to_mongo('trading_dev')
    # plot_scoring_test(mongo_connection, 'butterfly_pso_spark_10_june_ok')

    # show(x.values()[0].plot_params_2d())
    # mc = {'a': {'0': '56fd3298734edd055ab8e4d8', 'OOS_0': '56fd8d44abf88f8ba50a2961', 'OOS_1': '56fd90daabf88f8b9a0a2b6c'},
    #       'b': {'0': '56fd3aceabf88f50589a5cbd', 'OOS_0': '56fd8cd7734edd23e316db25', 'OOS_1': '56fd90dbabf88f8ba50a2b5a'}}
    # show(plot_multiple_metrics_and_returns(mongo_connection, 'football_butterfly_short_20160331', mc,
    #                                        flat_staking=False))
    study_name = 'ess_lth_lead_grid_search_20170114'
    merged_configs = get_merged_configurations(mongo_connection, study_name)
    p = plot_multiple_metrics_and_returns(mongo_connection, study_name, merged_configs, flat_staking=True, net=True)
    show(p)

    # configs = ['56e17400abf88f1801464d3c',
    #        '56e1617c734edd6315d81b70',
    #        '56e1bfd5734edd0ebcd826b6',
    #        '56e2463a734edd3ba4d826b6',
    #        '56e15f16734edd632dd820e4',
    #        '56e180c9abf88f5391462465',
    #        '56e17f8cabf88f537c461f51',
    #        '56e17f68abf88f5385461f23']
    # folder = '/Users/dedwards/TempData/butterfly_backtests'
    # subfolders = {cfg: '{}_a_160101_160301'.format(cfg) for cfg in configs}
    # bkt_names = subfolders.keys()
    #
    # bets_df = []
    # for name, subf in subfolders.iteritems():
    #     bets = pd.DataFrame.from_csv('{}/{}/bets.csv'.format(folder, subf))
    #     bets.loc[:, 'name'] = name
    #     bets_df.append(bets)
    # bets_df = pd.concat(bets_df)
    #
    # metrics_tbl, cum_ret_plot = backtest_run_report(bets_df, flat_staking=False, starting_capital=10000)
    # show(cum_ret_plot)

    # print param_sets_from_configs(['56fbec30734edd6224a1ba05'])
