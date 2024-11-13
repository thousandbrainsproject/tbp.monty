# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import matplotlib.pyplot as plt

"""
Profiled experiments output .csv files which can be readin with pandas. This file
contains some basic functions that help interpret profiling results.
"""


def linebreak_long_strings(string, chars_per_line=40):
    """Strings with filename are long, try to get them more readable in bar plots.

    Args:
        string: String to format.
        chars_per_line: Number of characters per line. Defaults to 40.

    Returns:
        Formatted string.
    """
    if len(string) < chars_per_line:
        return string

    string_len = len(string)
    new_string = string[:chars_per_line]
    cnt = chars_per_line
    while cnt <= len(string):
        if (string_len - cnt) < chars_per_line:
            new_string += "\n"
            new_string += string[cnt:]
            break
        else:
            new_string += "\n"
            new_string += string[cnt : cnt + chars_per_line]
            cnt += chars_per_line

    return new_string


def drop_filename(string):
    """Drop filename for shorter strings and easier viz.

    We do this because strings for code calls are long.

    Returns:
        String without filename.
    """
    return string.split(", file ")[0]


def sort_by_cumtime(df):
    new_df = df.sort_values("cumtime", ascending=False)
    return new_df


def sort_by_tottime(df):
    new_df = df.sort_values("tottime", ascending=False)
    return new_df


def get_data_from_df(df, sortby="cumtime"):
    df = df.sort_values(sortby, ascending=False)
    func_names = list(df["func"])
    times = list(df[sortby])

    return df, func_names, times


def bar_chart_cumtime(df, n_functions=None):
    df, func_names, cumtimes = get_data_from_df(df)
    if not n_functions:
        n_functions = len(func_names)

    fig, ax = plt.subplots(figsize=(20, 10))
    # NOTE: profiler dominates cumulative time because everything happens within profile
    # hence drop the first item so you can actually see the scale properly
    ax.bar(x=range(n_functions), height=cumtimes[1 : n_functions + 1])
    short_func_names = [drop_filename(func) for func in func_names]
    ax.set_xticks(range(n_functions))
    ax.set_xticklabels(short_func_names[1 : n_functions + 1], rotation=80)
    ax.set_ylabel("Cumulative Time (s)")
    ax.set_title(f"Time taken by top {n_functions}")

    plt.show()


def bar_chart_tottime(df, n_functions=None):
    df, func_names, tottimes = get_data_from_df(df, sortby="tottime")
    if not n_functions:
        n_functions = len(func_names)

    fig, ax = plt.subplots(figsize=(20, 10))
    # NOTE: profiler dominates cumulative time because everything happens within profile
    # hence drop the first item so you can actually see the scale properly
    ax.bar(x=range(n_functions), height=tottimes[n_functions])
    short_func_names = [drop_filename(func) for func in func_names]
    ax.set_xticks(range(n_functions))
    ax.set_xticklabels(short_func_names[n_functions], rotation=80)
    ax.set_ylabel("Total Time (s)")
    ax.set_title(f"Time taken by top {n_functions}")

    plt.show()


def print_top_k_functions(func_names, k=20):
    for i in range(k):
        print(func_names[i])
        print()


def get_total_time(df):
    total = df["cumtime"].sum()
    print(f"total time: {total} s")
    return total
