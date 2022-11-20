import numpy as np
import pandas as pd


def remove_outliers(df):
    # df = df[df.revenue != 0]
    # df = df[df.revenue >= 80]
    # df = df.drop(df[df.revenue <= df.revenue.quantile(.03)].index)
    # df = df.drop(df[df.revenue >= df.revenue.quantile(1 - .04)].index)
    upper = df.revenue.mean() + 3*df.revenue.std()
    lower = df.revenue.mean() - 3*df.revenue.std()
    df = df.drop(df[df.revenue > upper].index)
    df = df.drop(df[df.revenue < lower].index)
    return df


def fix_lat_lon(df):
    df['lat_processed'] = df.lat * 11.112
    df['lon_processed'] = df.lon * 6.4757
    return df


def fix_occurence(df, feature, replacing):
    threshold = 3
    df.loc[df[feature].value_counts()[df[feature]].values < threshold, feature] = replacing
    return df
