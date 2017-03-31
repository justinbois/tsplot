import numpy as np
import pandas as pd

import tsplot

def test_dark():
    df = pd.DataFrame(
        {'time': np.arange(20, dtype=float),
         'light': [True]*10 + [False]*10})
    lefts, rights = tsplot.dark(df, 'time', 'light')
    assert lefts == np.array([10.0])
    assert rights == np.array([19.0])

    df = pd.DataFrame(
        {'time': np.arange(30, dtype=float),
         'light': [True]*10 + [False]*10 + [True]*10})
    lefts, rights = tsplot.dark(df, 'time', 'light')
    assert lefts == np.array([10.0])
    assert rights == np.array([20.0])

    df = pd.DataFrame(
        {'time': pd.date_range('03/30/2017 9:00:00', periods=30, freq='T'),
         'light': [True]*10 + [False]*10 + [True]*10})
    lefts, rights = tsplot.dark(df, 'time', 'light')
    assert lefts == np.array(['2017-03-30T09:10:00.000000000'],
                             dtype='datetime64[ns]')
    assert rights == np.array(['2017-03-30T09:20:00.000000000'],
                               dtype='datetime64[ns]')
