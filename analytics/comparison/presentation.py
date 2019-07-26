# Presentation layer for strategy comparison
import matplotlib.pyplot as plt
import tempfile


class RunStatsPresenter(object):
    def __init__(self, tmp_dir=None):
        self._file_dir = tmp_dir or tempfile.mkdtemp(prefix="compare_")
        print "Your graphs are in {}".format(self._file_dir)

    def make_daily_pnl_graph(self, stats_objs):

        for stats in stats_objs:
            assert set(stats.pnl_by_date.keys()) == set(stats_objs[0].pnl_by_date.keys())
        sorted_dates = sorted(stats_objs[0].pnl_by_date.keys())

        labels = [''] * len(sorted_dates)
        for i in range(0, len(labels), 7):
            labels[i] = sorted_dates[i]

        for stats in stats_objs:
            plt.plot(stats.daily_cumulative_pnl,
                     label='{}'.format(stats.mnemonic),
                     alpha=0.5)
        # ax.set_xticks(sorted_dates)
        plt.xticks(range(len(labels)), labels, rotation=37)
        plt.legend()
        plt.title('Daily cumulative PNL')
        file_path = '%s/%s' % (self._file_dir, "daily_cumulative_PNL.png")
        plt.savefig(file_path, dpi=800)
        # plt.show()
        plt.close()
        print "Created image {}".format(file_path)
        return file_path
