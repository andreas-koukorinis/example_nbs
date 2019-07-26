__author__ = 'dedwards'

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from itertools import groupby

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.neighbors import KernelDensity

from bokeh.models import ColumnDataSource, HoverTool, BoxZoomTool, ResetTool, \
    CustomJS, TapTool, BoxSelectTool, LassoSelectTool, TableColumn, NumberFormatter, DataTable, SaveTool
from bokeh.models.widgets import Select, Panel, Tabs
from bokeh.plotting import figure, gridplot
from bokeh.layouts import column
from bokeh.models.layouts import Row, Column
from bokeh.io import show, output_file
from bokeh.palettes import YlOrRd9, PuBu9, Spectral11, BrBG11
from bokeh.charts import Bar

from sgmtradingcore.analytics.clustering import choose_clustering

pd.options.mode.chained_assignment = None

CLUSTER_COLS = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
                '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#cccc99', '#b15928',
                '#8dd3c7', '#ccccb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462',
                '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f']


class ParamSet(object):
    """
    Class to hold a set of Params with an associated score
    """
    def __init__(self, params, score, types, levels):
        self.params = params
        self.score = score
        self.types = types
        self.levels = levels
        self.names = self.types.keys()

    def convert_to_matrix_row(self, with_score=False):
        row = []
        for nn in self.names:
            if self.types[nn] == 'cat':
                row.extend([1 if self.params[nn] == l else 0 for l in self.levels[nn]])
            else:
                row.append(self.params[nn])
        if with_score:
            row.append(self.score)
        return row

    def __str__(self):
        msg = []
        for cc, typ in self.types.iteritems():
            p = self.params[cc]
            if typ == 'cont':
                msg.append('{:.2f}'.format(p))
            elif typ == 'ord':
                msg.append('{}'.format(self.levels[cc][int(p)]))
            else:
                msg.append('{}'.format(p))

        return '{' + ', '.join(msg) + '}'


class MultiParamSet(object):
    """
    Class to hold many sets of parameters for scaling, clustering etc.
    """
    def __init__(self, params_df, aux_df=None, n_top_clusters=3, min_clusters=3, max_clusters=20,
                 cluster_by_score=None, n_random_to_select=2, gen=None, score_name='Score'):
        """
        Initialize from data frame of parameters with optional score
        :param params_df:
        :return:
        """
        params_df = params_df.sort('score', ascending=False)
        self.names = [cc for cc in params_df.columns if cc != 'score']

        self.types = {}
        self.levels = {}
        for nn in self.names:
            values, typ, levels = self.get_param_info_from(params_df[nn])
            self.types[nn] = typ
            self.levels[nn] = levels
            params_df.loc[:, nn] = values

        self.df = params_df
        self.index = params_df.index.tolist()
        if aux_df is None:
            self.aux_df = None
        else:
            self.aux_df = aux_df.loc[self.df.index, :]

        self.param_sets = []
        for _, row in params_df.iterrows():
            params = {nn: row[nn] for nn in self.names}
            score = row.get('score', np.nan)
            self.param_sets.append(ParamSet(params, score, self.types, self.levels))
        self.num_param_sets = len(self.param_sets)
        self.generations = gen

        self._scaler = None
        self._param_mat = None
        self._param_mat_2d = None
        self._clusters = None

        self._cluster_by_score = cluster_by_score
        self._n_top_clusters = n_top_clusters
        self._min_clusters = min_clusters
        self._max_clusters = max_clusters
        self._n_random_to_select = n_random_to_select

        self._top_clusters = None
        self._selected = None

        self._score_name = score_name

    @staticmethod
    def get_param_info_from(values):
        def f_is_numeric(z):
            try:
                int(z)
                return True
            except ValueError:
                return False
        if len(set(values)) > 10:
            typ = 'cont'
        elif all([f_is_numeric(v) for v in values]):
            if any([float(v) - int(v) > 1e-3 for v in values]):
                typ = 'cont'
            else:
                typ = 'ord'
        else:
            typ = 'cat'

        if typ == 'ord':
            levels = sorted(list(set(values)), cmp=lambda a, b: cmp(int(a), int(b)))
            values = values.map(lambda x: levels.index(x))
            levels = [str(l) for l in levels]
        elif typ == 'cat':
            values = [str(v) for v in values]
            levels = list(set(values))
        else:
            levels = None

        return values, typ, levels

    @property
    def param_mat(self):
        if self._param_mat is None:
            mat = np.matrix([ps.convert_to_matrix_row() for ps in self.param_sets])
            self._scaler = StandardScaler()
            self._param_mat = self._scaler.fit_transform(mat)
        return self._param_mat

    @property
    def param_mat_2d(self):
        if self._param_mat_2d is None:
            dim_red_obj = TSNE(learning_rate=100, init='pca', n_components=2)
            # dim_red_obj = MDS()
            self._param_mat_2d = dim_red_obj.fit_transform(self.param_mat)
        return self._param_mat_2d

    @property
    def clusters(self):
        if self._clusters is None:
            cbs = [0., 1.] if self._cluster_by_score is None else self._cluster_by_score

            clusters = [-1]*self.num_param_sets

            score_min = min([s for s in self.scores if np.isfinite(s)])
            score_max = max(self.scores) + 0.01
            nc = 0
            for ii in range(len(cbs) - 1):
                s0 = score_min + cbs[ii] * (score_max - score_min)
                s1 = score_min + cbs[ii + 1] * (score_max - score_min)
                idx = [ii for ii, s in enumerate(self.scores) if s0 <= s < s1]
                if len(idx) < 10:
                    clusts = [0]*len(idx)
                else:
                    _, clusts, _, _ = choose_clustering(self.param_mat[idx, :], AgglomerativeClustering,
                                                    {'n_clusters': range(self._min_clusters, self._max_clusters)})
                for ii, cs in zip(idx, clusts):
                    clusters[ii] = cs + nc
                if len(clusts) > 0:
                    nc += max(clusts) + 1

            self._clusters = clusters
        return self._clusters

    @property
    def scores(self):
        return [ps.score for ps in self.param_sets]

    @property
    def top_clusters(self):
        if self._top_clusters is None:
            # Group the scores by cluster
            scores = defaultdict(lambda: [])
            for c, s in zip(self.clusters, self.scores):
                scores[c].append(s)

            # Choose the best scoring clusters (mean and maximum)
            scores = [(c, np.mean(s), np.max(s)) for c, s in scores.iteritems()]
            top_on_mean = [x[0] for x in sorted(scores, cmp=lambda a, b: cmp(a[1], b[1]))][-self._n_top_clusters:]
            top_on_max = [x[0] for x in sorted(scores, cmp=lambda a, b: cmp(a[2], b[2]))][-self._n_top_clusters:]
            self._top_clusters = list(set(top_on_max + top_on_mean))
        return self._top_clusters

    @property
    def selected(self):
        if self._selected is None:
            # Get the best score in each top cluster
            selected = defaultdict(lambda: (-1, -np.inf))
            for ii, (c, s) in enumerate(zip(self.clusters, self.scores)):
                if c in self.top_clusters:
                    if s > selected[c][1]:
                        selected[c] = (ii, s)
            self._selected = [sel[0] for sel in selected.itervalues()]

            # Add some random members of each top cluster
            sel_indexes = [self.index[ii] for ii in self._selected]
            for tc in self.top_clusters:
                idx = [ii for ii, c in enumerate(self.clusters) if c == tc and self.index[ii] not in sel_indexes]
                if len(idx) > 0:
                    self._selected += list(np.random.choice(idx, min(self._n_random_to_select, len(idx)), replace=False))

        return  self._selected

    def selected_indexes(self):
        return [self.index[ii] for ii in self.selected]

    def highest_scoring_param_set(self):
        return np.argmax(self.scores)

    def dist_between_param_sets(self, idx1, idx2):
        x1 = self.param_mat[idx1, :]
        x2 = self.param_mat[idx2, :]
        return np.sqrt(((x1 - x2)**2).sum())

    def plot_dist_vs_score(self):
        best_idx = self.highest_scoring_param_set()

        data = [(self.dist_between_param_sets(idx, best_idx), ps.score, str(ps), self.clusters[idx], idx)
                for idx, ps in enumerate(self.param_sets)]
        data = zip(*data)
        s = ColumnDataSource({'dist': data[0], 'score': data[1], 'param_str': data[2], 'cluster': data[3],
                              'colour': [cluster_cmap(c) for c in data[3]],
                              'size': [15 if ii in self.selected else 5 for ii in data[4]]})
        hover = HoverTool(tooltips=[('dist', '@dist{0.00}'), ('score', '@score{0.00}'), ('p', '@param_str'),
                                    ('cluster', '@cluster')])
        p = figure(width=900, height=500, tools=[hover, BoxZoomTool(), ResetTool(), SaveTool()],
                   title='Parameter Sets: distance vs. score')
        p.scatter(x='dist', y='score', color='colour', source=s, size='size', fill_alpha=0.3)
        p.xaxis.axis_label = 'Distance from best'
        p.yaxis.axis_label = 'Score'
        return p

    def plot_params_2d(self):
        data = [(self.param_mat_2d[idx, 0], self.param_mat_2d[idx, 1], ps.score, str(ps), self.clusters[idx], idx)
                for idx, ps in enumerate(self.param_sets)]
        data = zip(*data)

        hover = HoverTool(tooltips=[('score', '@score{0.00}'), ('p', '@param_str'), ('cluster', '@cluster')])
        p = figure(width=900, height=500, tools=[hover, BoxZoomTool(), ResetTool(), SaveTool()],
                   title='Parameter Sets (2D Embedding)')

        # Plot the convex hull and centroid of each top cluster
        for tc in self.top_clusters:
            mat = self.param_mat_2d[[ii for ii, c in enumerate(self.clusters) if c == tc], :]
            if mat.shape[0] > 2:
                ch = ConvexHull(mat)
                c_scores = [s for c, s in zip(self.clusters, self.scores) if c == tc]
                for sm in ch.simplices:
                    s_str = ['mean={:.2f}, max={:.2f}'.format(np.mean(c_scores), np.max(c_scores))]*2
                    s2 = ColumnDataSource({'x': ch.points[sm, 0], 'y': ch.points[sm, 1],
                                           'p': ['']*2, 'cluster': [tc]*2, 'score': s_str})
                    p.line(x='x', y='y', color=cluster_cmap(tc), source=s2)

        s = ColumnDataSource({'x': data[0], 'y': data[1], 'score': data[2],
                              'param_str': data[3],
                              'colour_score': cmap(data[2], rev=True),
                              'cluster': data[4],
                              'colour_cluster': [cluster_cmap(c) for c in data[4]],
                              'colour': ['black']*self.num_param_sets,
                              'size': [15 if ii in self.selected else 5 for ii in data[5]]})
        s.data['colour'] = s.data['colour_cluster']

        cb = CustomJS(args={'s': s}, code='''
            var choice = cb_obj.get('value');
            var d = s.get('data');
            var colour_label = 'colour_cluster';
            if (choice == 'Score'){
              colour_label = 'colour_score';
            }
            for (i = 0; i < d['colour'].length; i++) {
                d['colour'][i] = d[colour_label][i];
            }
            s.trigger('change');
        ''')
        select = Select(title='Colour by:', options=['Clusters', 'Score'], value='Clusters', callback=cb)

        p.scatter(x='x', y='y', color='colour', source=s, size='size', fill_alpha=0.3)

        p.xaxis.major_tick_line_color = None
        p.yaxis.major_tick_line_color = None
        p.xaxis.minor_tick_line_color = None
        p.yaxis.minor_tick_line_color = None
        p.xaxis.major_label_text_color = None
        p.yaxis.major_label_text_color = None
        return Column(p, select)

    def plot_study(self, with_table=True):
        plot_data = self.df.copy()
        plot_data.loc[:, 'colour'] = 'grey'
        plot_data['score'] = self.scores
        plot_data['name'] = self.index
        plot_data = plot_data.reset_index(drop=True)

        for cc in self.names:
            if self.types[cc] == 'ord':
                plot_data.loc[:, cc] = [self.levels[cc][x] for x in plot_data[cc]]

        s = ColumnDataSource(plot_data)
        s.add(self.clusters, name='cluster')
        s.add([cluster_cmap(c) for c in self.clusters], name='colour')
        sel_ind = self.selected_indexes()
        s.add([10 if nn in sel_ind else 5 for nn in plot_data['name']], name='size')

        grid_plots = []
        row_plots = []
        n_per_row = 3
        for column in self.names:
            column = str(column)
            # if self.types[column] == 'ord':
            #     s.data[column] = [x + 1 for x in s.data[column]]
            fig = figure(width=300, height=300, x_range=self.levels[column],
                         tools=[TapTool(), BoxSelectTool(), LassoSelectTool(), SaveTool()])
            fig.circle(source=s, x=column, y='score', color='colour', size='size', fill_alpha=0.3)
            fig.xaxis.axis_label = column
            fig.xaxis.axis_label_text_font_size = '10pt'
            fig.xaxis.major_label_text_font_size = '8pt'
            fig.yaxis.major_label_text_font_size = '8pt'
            if self.types[column] != 'cont' and \
                    (max([len(l) for l in self.levels[column]]) * len(self.levels[column]) > 40):
                fig.xaxis.major_label_orientation = np.pi / 8.

            if len(row_plots) >= n_per_row:
                grid_plots.append(row_plots)
                row_plots = []
            if len(row_plots) == 0:
                fig.yaxis.axis_label = self._score_name
                fig.yaxis.axis_label_text_font_size = '10pt'
            row_plots.append(fig)

        if len(row_plots) > 0:
            row_plots += [None] * (n_per_row - len(row_plots))
            grid_plots.append(row_plots)

        if with_table:
            tbl_columns = [TableColumn(field=column, title=column, formatter=NumberFormatter(format='0.00'))
                           if column == 'score' else
                           TableColumn(field=column, title=column)
                           if column == 'name' else
                           TableColumn(field=column, title=column, formatter=NumberFormatter(format='0.000'))
                           if self.types[column] == 'cont' else
                           TableColumn(field=column, title=column)
                           for column in ['score'] + self.names + ['name']]
            data_table = DataTable(source=s, columns=tbl_columns, width=900, height=300)
            return Column(gridplot(grid_plots), data_table)
        else:
            return gridplot(grid_plots)

    def plot_top_cluster_info(self):
        tabs = []
        for cluster in self.top_clusters:
            # Get all data for the selected cluster
            pdf = self.df.loc[[c == cluster for c in self.clusters], :]
            if self.aux_df is not None:
                pdf = pdf.merge(self.aux_df, left_index=True, right_index=True)
                aux_cols = self.aux_df.columns.tolist()
            else:
                aux_cols = []

            colour = cluster_cmap(cluster)

            # For each parameter and the score, create a plot
            grid_plots = []
            row_plots = []
            n_per_row = 3
            for column in ['score'] + self.names + aux_cols:
                if column == 'score' or column in aux_cols or self.types[column] == 'cont':
                    # Density plot
                    p = figure(width=300, height=300, tools=[SaveTool()])
                    add_density_plot(p, pdf[column], color=colour)
                    p.yaxis.axis_label = 'density'
                else:
                    if self.types[column] == 'ord':
                        pdf.loc[:, column] = [self.levels[column][x] for x in pdf[column]]

                    # "Bar" chart
                    p = figure(width=300, height=300, tools=[SaveTool()], x_range=self.levels[column])
                    for l, count in Counter(pdf[column]).iteritems():
                        p.line(x=[l, l], y=[0, count], color=colour, line_width=2)
                        p.scatter(x=[l], y=[count], color=colour, size=10)
                    p.yaxis.axis_label = 'count'
                    if (max([len(l) for l in self.levels[column]]) * len(self.levels[column]) > 40):
                        p.xaxis.major_label_orientation = np.pi / 8.

                p.xaxis.axis_label = column
                p.xaxis.axis_label_text_font_size = '10pt'
                p.xaxis.major_label_text_font_size = '8pt'
                p.yaxis.major_label_text_color = None
                p.yaxis.axis_label_text_font_size = '10pt'
                if len(row_plots) >= n_per_row:
                    grid_plots.append(row_plots)
                    row_plots = []
                row_plots.append(p)

            if len(row_plots) > 0:
                row_plots += [None] * (n_per_row - len(row_plots))
                grid_plots.append(row_plots)

            tabs.append(Panel(child=gridplot(grid_plots), title='Cluster {}'.format(cluster)))
        return Tabs(tabs=tabs)

    def selected_table(self):
        pdf = self.df.iloc[self.selected, :]
        for cc in self.names:
            if self.types[cc] == 'ord':
                pdf.loc[:, cc] = [self.levels[cc][x] for x in pdf[cc]]
        return parameter_table(pdf)

    def plot_generations(self):
        if self.generations is None:
            return None

        ugens = list(set(self.generations))
        gen_scores = []
        for gen in ugens:
            try:
                score = max([s for g, s in zip(self.generations, self.scores) if g == gen])
            except ValueError:
                score = -np.inf
            gen_scores.append(max(gen_scores + [score]))

        p1 = figure(width=450, height=300, tools=[SaveTool()])
        p1.line(ugens, gen_scores, color='black')
        p1.scatter(ugens, gen_scores, color='black')

        p2 = figure(width=450, height=300, tools=[SaveTool()], x_range=p1.x_range, y_range=p1.y_range)
        for cluster in self.top_clusters:
            gen_scores = []
            for gen in ugens:
                cg_scores = [s for g, s, c in zip(self.generations, self.scores, self.clusters)
                             if g == gen and c == cluster]
                try:
                    gen_scores.append(max(gen_scores + cg_scores))
                except ValueError:
                    gen_scores.append(-np.inf)
            p2.line(ugens, gen_scores, color=cluster_cmap(cluster))
            p2.scatter(ugens, gen_scores, color=cluster_cmap(cluster))

        for p in [p1, p2]:
            p.xaxis.axis_label = 'Generation'
            p.xaxis.axis_label_text_font_size = '10pt'
            p.yaxis.axis_label = 'Score'
            p.yaxis.axis_label_text_font_size = '10pt'

        return Row(p1, p2)


def parameter_table(pdf):
    pdf = pdf.transpose()
    pdf.index.name = 'param'
    pdf = pdf.reset_index()
    pdf.columns = pdf.columns.astype('str')

    s = ColumnDataSource(pdf)
    cols = [TableColumn(title=cc, field=cc, width=100) for cc in pdf.columns]
    tbl = DataTable(source=s, columns=cols, width=900, height=350, row_headers=False, fit_columns=False)
    return column(tbl)


def cmap(x, rng=None, palette=PuBu9, rev=False):
    if rng is None:
        rng = [min(x), max(x)]
    rng = sorted(rng, reverse=rev)
    rng[1] += 0.01 * (rng[1] - rng[0])
    return [palette[int(np.floor(len(palette) * (xx - rng[0]) / (rng[1] - rng[0])))] for xx in x]


def cluster_cmap(c):
    if c < 0:
        return '#888888'
    elif c >= len(CLUSTER_COLS):
        return '#000000'
    else:
        return CLUSTER_COLS[c]


def add_density_plot(p, x, **kwargs):
    x = x[~np.isnan(x)]
    bw = 0.75 * np.std(x) if np.std(x) > 0 else 0.001
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(x.reshape(-1, 1))
    xx = np.linspace(x.min(), x.max(), num=100)
    dist = np.exp(kde.score_samples(xx.reshape(-1, 1)))
    xx = np.concatenate([[x.min()], xx, [x.max()]])
    dist = np.concatenate([[0.], dist, [0.]])
    s = ColumnDataSource({'x': xx, 'y': dist})
    p.line(x='x', y='y', source=s, **kwargs)
    p.patch(x='x', y='y', fill_alpha=0.3, source=s, **kwargs)


if __name__ == '__main__':
    output_file('/tmp/parameters.html')
    params_df = pd.DataFrame.from_csv('/Users/dedwards/TempData/mps_params.csv')
    aux_df = pd.DataFrame.from_csv('/Users/dedwards/TempData/mps_aux.csv')
    mps = MultiParamSet(params_df, aux_df=aux_df, n_top_clusters=5, max_clusters=10, cluster_by_score=[0.6, 0.8, 1.0])
    show(mps.plot_params_2d())
