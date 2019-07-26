
def compare_per_event_pnl(stats1, stats2):
    all_event_ids = set()
    all_event_ids.update(set(stats1.per_event_pnl))
    all_event_ids.update(set(stats2.per_event_pnl))

    pnl_diffs = get_numerical_dict_diffs(stats1.per_event_pnl, stats2.per_event_pnl)
    return pnl_diffs


def get_numerical_dict_diffs(d1, d2):
    """Get the difference d1-d2 for both keys.

    If a key is missing it assumes zero

    """
    all_keys = set()
    all_keys.update(set(d1))
    all_keys.update(set(d2))

    diffs = {}
    for key in all_keys:
        d1v = d1.get(key, 0)
        d2v = d2.get(key, 0)

        diffs[key] = d1v - d2v
    return diffs


def get_bigger_abs_values(d, num_entries=10):
    """
    Get the 10 bigger key and values of the dict d with the biggest abs values
    """
    abs_values_and_key = []
    for k, v in d.iteritems():
        abs_values_and_key.append((abs(v), k))

    abs_values_and_key = sorted(abs_values_and_key, reverse=True)

    ret = {}
    for _, key in abs_values_and_key[:num_entries]:
        ret[key] = d[key]
    return ret


def get_biggest_dict_diffs(d1, d2, num_entries=10):
    """
    Return a dict with the num_entries biggest differences in the input dicts
    """
    daily_diffs = get_numerical_dict_diffs(d1, d2)
    return get_bigger_abs_values(daily_diffs)
