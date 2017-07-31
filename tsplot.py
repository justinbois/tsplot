import numpy as np
import pandas as pd

import numba

try:
    import tqdm
except:
    pass

import bokeh.models
import bokeh.palettes
import bokeh.plotting


@numba.jit(nopython=True)
def draw_bs_sample(data):
    return np.random.choice(data, size=len(data))


@numba.jit(nopython=True)
def draw_bs_reps_mean(data, size=1000):
    bs_reps = np.empty(size)
    for i in range(size):
        bs_reps[i] = np.mean(draw_bs_sample(data))
    return bs_reps


@numba.jit(nopython=True)
def draw_bs_reps_median(data, size=1000):
    bs_reps = np.empty(size)
    for i in range(size):
        bs_reps[i] = np.median(draw_bs_sample(data))
    return bs_reps


def bs_conf_int(data, ptiles, stat='mean', size=1000):
    if stat == 'mean':
        bs_reps = draw_bs_reps_mean(data, size=size)
    elif stat == 'median':
        bs_reps = draw_bs_reps_median(data, size=size)
    else:
        raise RuntimeError("`stat` must be either 'mean' or 'median'.")

    return np.percentile(bs_reps, ptiles)


def ts_conf_int(df, time, signal, ptiles, stat='mean', time_ind=None,
                size=1000):
    """
    Compute bootstrap confidence intervals on time series data.
    """
    if stat not in ['mean', 'median']:
        raise RuntimeError("`stat` must be either 'mean' or 'median'.")

    # Convenience strings to reference high and los confidence interval bounds
    low = 'low_' + signal
    high = 'high_' + signal

    # Set up output DataFrame
    df_out = pd.DataFrame(columns=[time, time_ind, signal, low, high])
    if time_ind is None:
        time_ind = time
        df_out[time] = df[time].unique()
        df_out[time_ind] = np.arange(len(df[time].unique()))
    else:
        df_out[time_ind] = df[time_ind].unique()
        t = [df.loc[df[time_ind]==i, time].iloc[0]
                            for i in df[time_ind].unique()]
        df_out[time] = t

    # Get bootstrap estimates
    try:
        iterator = tqdm.tqdm(df[time_ind].unique())
    except:
        iterator = df[time_ind].unique()

    for ind in iterator:
        data = df.loc[df[time_ind] == ind, signal].values
        conf_int = bs_conf_int(data, ptiles, stat=stat, size=size)
        if stat == 'mean':
            df_out.loc[df_out[time_ind]==ind, signal] = np.mean(data)
        elif stat == 'median':
            df_out.loc[df_out[time_ind]==ind, signal] = np.median(data)
        df_out.loc[df_out[time_ind]==ind, low] = conf_int[0]
        df_out.loc[df_out[time_ind]==ind, high] = conf_int[1]

    return df_out


def dark(df, time, light, buffer=0.0):
    """
    Compute start and end time for dark bars on plots.

    Parameters
    ----------
    df : pandas DataFrame
        Tidy data frame containing a column with time points and
        a column with Boolean values, False being "dark" regions
        on plot.
    time : string or other acceptable pandas index
        Column containing time points.
    light : string or other acceptable pandas index
        Column containing Booleans for where the plot background
        is light.
    buffer : float, default 0.0
        If the end of the time series is dark, how much buffer
        to add to the right side in the same units as the time
        column. If the time column is a datetime object, `buffer`
        is in units of seconds.

    Returns
    -------
    lefts : ndarray
        Time points for left side of dark bars
    rights : ndarray
        Time points for right side of dark bars
    """
    if df[time].dtype == '<M8[ns]':
        buffer *= 1e9

    t = df[time].values
    lefts = t[np.where(np.diff(df[light].astype(int)) == -1)[0] + 1]
    rights = t[np.where(np.diff(df[light].astype(int)) == 1)[0] + 1]

    if len(lefts) > len(rights):
        rights = np.concatenate((rights, (t[-1] + buffer,)))

    return lefts, rights


def shift_time_points(t, s, time_shift):
    """
    Shift time points along intervals.

    Parameters
    ----------
    t : ndarray
        Time points
    s : ndarray
        Signal
    time_shift : string, default 'left'
        One of {'left', 'right', 'center', 'interval'}
        left: do not perform a time shift
        right: Align time points to right edge of interval
        center: Align time points to the center of the interval
        interval: Plot the signal as a horizontal line segment
                  over the time interval

    Returns
    -------
    t_shift : ndarray
        Shifted time points
    s_shift : ndarray
        Appropriately adjusted signal.

    Notes
    -----
    .. Assumes that the input is aligned on the left of the time
       intervals.
    """
    if time_shift not in ['left', 'center', 'right', 'interval']:
        raise RuntimeError("`time_shift` must be one of {'left', 'center', 'right', 'interval'}.")

    if time_shift == 'interval':
        new_t = np.empty(2*len(t))
        new_t[0] = t[0]
        new_t[1:-1:2] = t[1:]
        new_t[2:-1:2] = t[1:]
        new_t[-1] = 2*t[-1] - t[-2]

        new_s = np.empty(2*len(s))
        new_s[::2] = s
        new_s[1::2] = s
    elif time_shift == 'center':
        new_t = np.copy(t)
        new_t[:-1] += np.diff(t) / 2
        new_t[-1] += (t[-1] - t[-2]) / 2
        new_s = s
    elif time_shift == 'right':
        new_t = np.empty_like(t)
        new_t[:-1] = t[1:]
        new_t[-1] = 2*t[-1] - t[-2]
        new_s = s
    elif time_shift == 'left':
        new_t = t
        new_s = s

    return new_t, new_s


def canvas(df=None, time=None, identifier=None, light=None, height=350,
           width=650, x_axis_label='time', y_axis_label=None, hover=True):
    """
    Make a Bokeh Figure instance for plotting time series.

    Parameters
    ----------
    df : pandas DataFrame
        Tidy DataFrame. Must have columns:
        - identifier: ID of each time series, if `light` is not None
        - time: time points for time series, if `light` is not None
        - light: Column of Booleans for if light is on, if `light`
            is not None
    time : string or None or any acceptable pandas index, default None
        The name of the column in `df` containing the time points.
        Ignored is `light` is None. Otherwise, `time` cannot be None.
    identifier : string or any acceptable pandas index, or None
        The name of the column in `df` containing the IDs. Used with
        the hover tool.
    light : string or None or any acceptable pandas index, default None
        Column containing Booleans for where the plot background
        is light. If None, no shaded bars are present on the figure.
    height : int, default 350
        Height of plot in pixels.
    width : int, default 650
        Width of plot in pixels.
    x_axis_label : string or None, default 'time'
        x-axis label.
    y_axis_label : string or None, default None
        y-axis label
    hover : bool, default True
        If True, have a hover tool for plots named 'hover'.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Bokeh figure ready for plotting time series.
    """

    # Create figure
    p = bokeh.plotting.figure(width=width, height=height,
                              x_axis_label=x_axis_label,
                              y_axis_label=y_axis_label,
                              tools='pan,box_zoom,wheel_zoom,reset,resize,save')

    if df is None:
        return p

    if light is not None:
        if time is None:
            raise RuntimeError('if `light` is not None, must supply `time`.')

        # Determine when nights start and end
        lefts, rights = dark(df[df[identifier]==df[identifier].unique().min()],
                             time, light)

        # Make shaded boxes
        dark_boxes = []
        for left, right in zip(lefts, rights):
            dark_boxes.append(
                    bokeh.models.BoxAnnotation(plot=p, left=left, right=right,
                                               fill_alpha=0.1, fill_color='gray'))
        p.renderers.extend(dark_boxes)

    # Add a HoverTool to highlight individuals
    if identifier is not None and hover:
        p.add_tools(bokeh.models.HoverTool(
                tooltips=[(identifier, '@'+identifier)], names=['hover']))

    return p


def time_series_plot(p, df, time, signal, identifier, time_ind=None,
                     summary_trace='mean', time_shift='left', alpha=0.75,
                     hover_color='#535353', colors=None, title=None,
                     legend=None):
    """
    Make a plot of multiple time series with a summary statistic.

    Parameters
    ----------
    p : bokeh.plotting.Figure instance
        Figure on which to make the plot, usually generated
        with the canvas() function.
    df : pandas DataFrame
        Tidy DataFrame minimally with columns:
        - time: The time points; should be the same for each ID
        - signal: The y-axis of the time series
        - identifier: ID of each time series
        Optionally:
        - time_ind: Indices of time points for use in computing
            summary statistics. Useful when the time points are
            floats.
        Note that if the DataFrame has a category column,
        this is ignored and all time serires are plotted.
    time : string or any acceptable pandas index
        The name of the column in `df` containing the time points
    signal : string or any acceptable pandas index
        The name of the column in `df` containing the y-values
    identifier : string or any acceptable pandas index
        The name of the column in `df` containing the IDs
    time_ind : string or any acceptable pandas index
        The name of the column in `df` containing the time indices
        to be used in computing summary statistics. These values
        are used to do a groupby. Default is the column given by
        `time`.
    summary_trace : string, float, or None, default 'mean'
        Which summary statistic to use to make summary trace. If a
        string, can one of 'mean', 'median', 'max', or 'min'. If
        None, no summary trace is generated. If a float between
        0 and 1, denotes which quantile to show.
    time_shift : string, default 'left'
        One of {'left', 'right', 'center', 'interval'}
        left: do not perform a time shift
        right: Align time points to right edge of interval
        center: Align time points to the center of the interval
        interval: Plot the signal as a horizontal line segment
                  over the time interval
    alpha : float, default 0.75
        alpha value for individual time traces
    hover_color : string, default '#535353'
        Hex value for color when hovering over a curve
    colors : list or tuple of length 2, default ['#a6cee3', '#1f78b4']
        colors[0]: hex value for color of all time series
        colors[1]: hex value for color of summary trace
    title : string or None, default None
        Title of plot.
    legend :  str or None, default None
        Legend text for summary line.

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Bokeh plot populated with time series.

    Notes
    -----
    .. Assumes that the signal, if binned, is aligned with the
       *left* of the time interval. I.e., if df[time] = [0, 1, 2],
       the values of df[signal] are assumed to be aggregated over
       time intervals 0 to 1, 1 to 2, and 2 to 3.
    """

    if time_shift not in ['left', 'center', 'right', 'interval']:
        raise RuntimeError("`time_shift` must be one of {'left', 'center', 'right', 'interval'}.")

    if colors is None:
        colors = bokeh.palettes.brewer['Paired'][3][:2]

    if time_ind is None:
        time_ind = time

    # Make the lines for display
    ml = []
    for individual in df[identifier].unique():
        t, s = df.loc[df[identifier]==individual, [time, signal]].values.T
        t, s = shift_time_points(t, s, time_shift)
        sub_df = pd.DataFrame({time: t, signal: s})
        source = bokeh.models.ColumnDataSource(sub_df)
        ml.append(p.line(x=time, y=signal, source=source, line_width=0.5,
                  alpha=alpha, color=colors[0], name='do_not_hover',
                  line_join='bevel'))

    # Plot summary trace
    if summary_trace is not None:
        # Get the time axis
        t = df.loc[df[identifier]==df[identifier].unique()[0], time].values

        # Perform summary statistic calculation
        if summary_trace == 'mean':
            y = df.groupby(time_ind)[signal].mean().values
        elif summary_trace == 'median':
            y = df.groupby(time_ind)[signal].median().values
        elif summary_trace == 'max':
            y = df.groupby(time_ind)[signal].max().values
        elif summary_trace == 'min':
            y = df.groupby(time_ind)[signal].min().values
        elif type(summary_trace) == float:
            if summary_trace > 0 and summary_trace < 1:
                y = df.groupby(time_ind)[signal].quantile(summary_trace).values
            else:
                raise RuntimeError('Invalid summary_trace value.')
        else:
            raise RuntimeError('Invalid summary_trace value.')

        t, y = shift_time_points(t, y, time_shift)
        summary_line = p.line(t, y, line_width=3, color=colors[1],
                              line_join='bevel', legend=legend)

    # Make lines for hover
    for individual in df[identifier].unique():
        t, s = df.loc[df[identifier]==individual, [time, signal]].values.T
        t, s = shift_time_points(t, s, time_shift)
        new_id = [individual] * len(t)
        sub_df = pd.DataFrame({time: t, signal: s, identifier: new_id})
        source = bokeh.models.ColumnDataSource(sub_df)
        p.line(x=time, y=signal, source=source, line_width=2, alpha=0,
               name='hover', line_join='bevel', hover_color=hover_color)

    # Label title
    if title is not None:
        p.title.text = title

    return p


def all_traces(df, time, signal, identifier, time_ind=None,
               light=None, summary_trace='mean', time_shift='left', alpha=0.75,
               hover_color='#535353', height=350, width=650,
               x_axis_label='time', y_axis_label=None, colors=None):
    """
    Make a plot of multiple time series with a summary statistic.

    Parameters
    ----------
    df : pandas DataFrame
        Tidy DataFrame minimally with columns:
        - time: The time points; should be the same for each ID
        - signal: The y-axis of the time series
        - identifier: ID of each time series
        Optionally:
        - time_ind: Indices of time points for use in computing
            summary statistics. Useful when the time points are
            floats.
        Note that if the DataFrame has a category column,
        this is ignored and all time serires are plotted.
    time : string or any acceptable pandas index
        The name of the column in `df` containing the time points
    signal : string or any acceptable pandas index
        The name of the column in `df` containing the y-values
    identifier : string or any acceptable pandas index
        The name of the column in `df` containing the IDs
    time_ind : string or any acceptable pandas index
        The name of the column in `df` containing the time indices
        to be used in computing summary statistics. These values
        are used to do a groupby. Default is the column given by
        `time`.
    light : string or None or any acceptable pandas index, default None
        Column containing Booleans for where the plot background
        is light. If None, no shaded bars are present on the figure.
    summary_trace : string, float, or None, default 'mean'
        Which summary statistic to use to make summary trace. If a
        string, can one of 'mean', 'median', 'max', or 'min'. If
        None, no summary trace is generated. If a float between
        0 and 1, denotes which quantile to show.
    time_shift : string, default 'left'
        One of {'left', 'right', 'center', 'interval'}
        left: do not perform a time shift
        right: Align time points to right edge of interval
        center: Align time points to the center of the interval
        interval: Plot the signal as a horizontal line segment
                  over the time interval
    alpha : float, default 0.75
        alpha value for individual time traces
    hover_color : string, default '#535353'
        Hex value for color when hovering over a curve
    height : int, default 350
        Height of plot in pixels.
    width : int, default 650
        Width of plot in pixels.
    x_axis_label : string or None, default 'time'
        x-axis label.
    y_axis_label : string or None, default None
        y-axis label
    colors : list or tuple of length 2, default ['#a6cee3', '#1f78b4']
        colors[0]: hex value for color of all time series
        colors[1]: hex value for color of summary trace

    Returns
    -------
    output : bokeh.plotting.Figure instance
        Bokeh plot populated with time series.

    Notes
    -----
    .. Assumes that the signal, if binned, is aligned with the
       *left* of the time interval. I.e., if df[time] = [0, 1, 2],
       the values of df[signal] are assumed to be aggregated over
       time intervals 0 to 1, 1 to 2, and 2 to 3.
    """

    p = canvas(df, time, identifier, light, height=height, width=width,
               x_axis_label=x_axis_label, y_axis_label=y_axis_label)

    time_series_plot(
        p, df, time, signal, identifier, time_ind=time_ind,
        summary_trace=summary_trace, time_shift=time_shift, alpha=alpha,
        hover_color=hover_color, colors=colors)

    return p


def grid(df, time, signal, category, identifier, cats=None, time_ind=None,
         light=None, summary_trace='mean', time_shift='left', alpha=0.75,
         hover_color='#535353', height=200, width=650,
         x_axis_label='time', y_axis_label=None, colors=None, show_title=True):
    """
    Generate a plot of time series separated by category.

    Parameters
    ----------
    df : pandas DataFrame
        Tidy DataFrame minimally with columns:
        - time: The time points; should be the same for each ID
        - signal: The y-axis of the time series
        - category: Categorization of each time series. I.e., each
            individual time trace belongs to a single category.
        - identifier: ID of each time series
        Optionally, can have columns
        - time_ind: Indices of time points for use in computing
            summary statistics. Useful when the time points are
            floats.
        - light: Column of Booleans for if light is on
    time : string or any acceptable pandas index
        The name of the column in `df` containing the time points
    signal : string or any acceptable pandas index
        The name of the column in `df` containing the y-values
    category : string or any acceptable pandas index
        The name of the column in `df` that is used to group time
        series into respective subplots.
    identifier : string or any acceptable pandas index
        The name of the column in `df` containing the IDs
    cats : list or tuple, default None
        List of categories to include in plot, in order. Each entry
        must be present in df['category']. If None, defaults to
        cats = df['category'].unique().
    time_ind : string or any acceptable pandas index
        The name of the column in `df` containing the time indices
        to be used in computing summary statistics. These values
        are used to do a groupby. Default is the column given by
        `time`.
    light : string or None or any acceptable pandas index, default None
        Column containing Booleans for where the plot background
        is light. If None, no shaded bars are present on the figure.
    summary_trace : string, float, or None, default 'mean'
        Which summary statistic to use to make summary trace. If a
        string, can one of 'mean', 'median', 'max', or 'min'. If
        None, no summary trace is generated. If a float between
        0 and 1, denotes which quantile to show.
    time_shift : string, default 'left'
        One of {'left', 'right', 'center', 'interval'}
        left: do not perform a time shift
        right: Align time points to right edge of interval
        center: Align time points to the center of the interval
        interval: Plot the signal as a horizontal line segment
                  over the time interval
    alpha : float, default 0.75
        alpha value for individual time traces
    hover_color : string, default '#535353'
        Hex value for color when hovering over a curve
    height : int, default 200
        Height of each subplot plot in pixels.
    width : int, default 650
        Width of each subplot in pixels.
    x_axis_label : string or None, default 'time'
        x-axis label.
    y_axis_label : string or None, default None
        y-axis label
    colors : dict, default None
        colors[cat] is a 2-list containg, for category `cat`:
            colors[cat][0]: hex value for color of all time series
            colors[cat][1]: hex value for color of summary trace
        If none, colors are generated using paired ColorBrewer colors,
        with a maximum of six categories.
    show_title : bool, default True
        If True, label subplots with with the category.

    Returns
    -------
    output : Bokleh grid plot
        Bokeh figure with subplots of all time series
    """
    # Get the categories
    if cats is None:
        cats = df[category].unique()
    elif np.isin(cats, df[category].unique()).all():
        raise RuntimeError('Specified `cats` not all present in df[category].')

    # Make colors if not supplied
    if colors is None:
        colors = get_colors(cats)

    # Create figures
    ps = [canvas(df, time, identifier, light, height=height, width=width,
                 x_axis_label=x_axis_label, y_axis_label=y_axis_label)
                        for _ in range(len(cats))]

    # Link ranges (enable linked panning/zooming)
    for i in range(1, len(cats)):
        ps[i].x_range = ps[0].x_range
        ps[i].y_range = ps[0].y_range

    # Populate glyphs
    title = None
    for p, cat in zip(ps, cats):
        sub_df = df.loc[df[category]==cat, :]
        if show_title:
            title = cat
        _ = time_series_plot(
                p, sub_df, time, signal, identifier, time_ind=time_ind, summary_trace=summary_trace, time_shift=time_shift, alpha=alpha,
                hover_color=hover_color, colors=colors[cat], title=title)

    return bokeh.layouts.gridplot([[ps[i]] for i in range(len(ps))])


def summary(df, time, signal, category, identifier, time_ind=None, light=None,
            summary_trace='mean', time_shift='left', confint=True,
            ptiles=(2.5, 97.5), n_bs_reps=1000, alpha=0.25, height=350,
            width=650, x_axis_label='time', y_axis_label=None, colors=None,
            legend=True):
    """
    Generate a set of plots of time series.

    Parameters
    ----------
        Tidy DataFrame minimally with columns:
        - time: The time points; should be the same for each ID
        - signal: The y-axis of the time series
        - category: Categorization of each time series. I.e., each
            individual time trace belongs to a single category.
        - identifier: ID of each time series
        Optionally, can have columns
        - time_ind: Indices of time points for use in computing
            summary statistics. Useful when the time points are
            floats.
        - light: Column of Booleans for if light is on
    time : string or any acceptable pandas index
        The name of the column in `df` containing the time points
    signal : string or any acceptable pandas index
        The name of the column in `df` containing the y-values
    category : string or any acceptable pandas index
        The name of the column in `df` that is used to group time
        series into respective subplots.
    identifier : string or any acceptable pandas index
        The name of the column in `df` containing the IDs
    time_ind : string or any acceptable pandas index
        The name of the column in `df` containing the time indices
        to be used in computing summary statistics. These values
        are used to do a groupby. Default is the column given by
        `time`.
    light : string or None or any acceptable pandas index, default None
        Column containing Booleans for where the plot background
        is light. If None, no shaded bars are present on the figure.
    summary_trace : string, default 'mean'
        Which summary statistic to use to make summary trace. Must
        be either 'mean' or 'median'.
    time_shift : string, default 'left'
        One of {'left', 'right', 'center', 'interval'}
        left: do not perform a time shift
        right: Align time points to right edge of interval
        center: Align time points to the center of the interval
        interval: Plot the signal as a horizontal line segment
                  over the time interval
    confint : bool, default True
        If True, also display confidence interval.
    ptiles : list or tuple of length two, default (2.5, 97.5)
        Percentiles for confidence intervals; ignored if
        `confint` is False.
    n_bs_reps : int, default 1000
        Number of bootstrap replicates to use in conf. int. Ignored if
        `confint` is False.
    alpha : float, default 0.25
        alpha value for confidence intervals
    height : int, default 350
        Height of plot in pixels.
    width : int, default 650
        Width of plot in pixels.
    x_axis_label : string or None, default 'time'
        x-axis label.
    y_axis_label : string or None, default None
        y-axis label
    colors : dict, default None
        colors[cat] is a 2-list containg, for category `cat`:
            colors[cat][0]: hex value for color of all time series
            colors[cat][1]: hex value for color of summary trace
        If none, colors are generated using paired ColorBrewer colors,
        with a maximum of six categories.
    legend : bool, default True
        If True, show legend.

    Returns
    -------
    output : Bokleh plot
        Bokeh figure with summary plots
    """
    # Get the categories
    cats = df[category].unique()

    # Make colors if not supplied
    if colors is None:
        colors = get_colors(cats)

    # Check input stat
    if summary_trace not in ['mean', 'median']:
        raise RuntimeError("`summary` trace must be either 'mean' or 'median'")

    # Convenient strings
    low = 'low_' + signal
    high = 'high_' + signal

    # Create figures
    p = canvas(df, time, identifier, light, height=height, width=width,
               x_axis_label=x_axis_label, y_axis_label=y_axis_label,
               hover=False)

    # Populate glyphs
    for cat in cats:
        sub_df = df.loc[df[category]==cat, :]
        if confint:
            print('Performing bootstrap estimates for {0:s}....'.format(cat))
            df_summ = ts_conf_int(
                    sub_df, time, signal, ptiles, stat=summary_trace,
                    time_ind=time_ind, size=n_bs_reps)

            # Extract and shift time points
            t, y_low = shift_time_points(df_summ[time].values,
                                         df_summ[low].values, time_shift)
            t, y_high = shift_time_points(df_summ[time].values,
                                          df_summ[high].values, time_shift)
            t, y = shift_time_points(df_summ[time].values,
                                     df_summ[signal].values, time_shift)

            # Plot confidence interval
            patch_t = np.concatenate((t, t[::-1]))
            patch_y = np.concatenate((y_low, y_high[::-1]))
            p.patch(patch_t, patch_y, color=colors[cat][0], fill_alpha=alpha,
                    line_join='bevel')
        else:
            if summary_trace == 'mean':
                y = sub_df.groupby(time_ind)[signal].mean().values
            elif summary_trace == 'median':
                y = df.groupby(time_ind)[signal].median().values
            t = sub_df.loc[sub_df[identifier]==sub_df[identifier].unique()[0],
                           time].values
            t, y = shift_time_points(t, y, time_shift)

        # Plot the summary line
        if legend:
            leg = cat
        else:
            leg = None
        summary_line = p.line(t, y, line_width=3, color=colors[cat][1],
                                line_join='bevel', legend=leg)

    return p


def get_colors(cats):
    """
    Makes a color dictionary for plots.

    Parameters
    ----------
    cats : list or tuple, maximum length of 6
        Categories to be used as keys for the color dictionary

    Returns
    -------
    colors : dict, default None
        colors[cat] is a 2-list containg, for category `cat`:
            colors[cat][0]: hex value for color of all time series
            colors[cat][1]: hex value for color of summary trace
        Colors are generated using paired ColorBrewer colors,
        with a maximum of six categories.
    """
    if len(cats) > 6:
        raise RuntimeError('Maxium of 6 categoriess allowed.')
    c = bokeh.palettes.brewer['Paired'][max(3, 2*len(cats))]
    return {g: (c[2*i], c[2*i+1]) for i, g in enumerate(cats)}
