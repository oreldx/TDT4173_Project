import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder, StandardScaler


def fix_year(df):
    return df[df.year == 2016]


def create_population_repartition(df_grunnkrets_house_pers):
    tmp = df_grunnkrets_house_pers[[
            'grunnkrets_id',
            'couple_children_0_to_5_years',
            'couple_children_6_to_17_years',
            'couple_children_18_or_above',
            'couple_without_children',
            'single_parent_children_0_to_5_years',
            'single_parent_children_6_to_17_years',
            'single_parent_children_18_or_above',
            'singles'
        ]]
    return tmp.rename({column: 'hp_' + column for column in tmp.columns if column != 'grunnkrets_id'}, axis=1)


def create_income_repartition(df_grunnkrets_income_house):
    tmp = df_grunnkrets_income_house[[
            'grunnkrets_id',
            'all_households',
            'singles',
            'couple_without_children',
            'couple_with_children',
            'other_households',
            'single_parent_with_children'
            ]]
    return tmp.rename({column: 'ih_' + column for column in tmp.columns if column != 'grunnkrets_id'}, axis=1)


def create_grunnkret_geodata(df_grunnkrets_stripped):
    return df_grunnkrets_stripped[[
                            'grunnkrets_id',
                            'area_km2',
                            'municipality_name'
                            ]]


def create_hierarchy(df_plaace_hierarchy):
    return df_plaace_hierarchy[[
                            'plaace_hierarchy_id',
                            'lv1_desc',
                            'lv2_desc',
                            'lv3_desc'
                            ]]


def create_population_age(df_grunnkrets_age_dist):
    df_grunnkrets_population = df_grunnkrets_age_dist.loc[:, ['grunnkrets_id']]
    df_grunnkrets_population['total_population'] = df_grunnkrets_age_dist.iloc[:,2:92].sum(axis=1)
    df_grunnkrets_population['youngs'] = df_grunnkrets_age_dist.iloc[:, 2:20].sum(axis=1)
    df_grunnkrets_population['adults'] = df_grunnkrets_age_dist.iloc[:, 21:64].sum(axis=1)
    df_grunnkrets_population['seniors'] = df_grunnkrets_age_dist.iloc[:, 65:92].sum(axis=1)
    return df_grunnkrets_population


def find_closest_bus_stop(df, df_bus_stops):
    """
    Combine the training data with the bus stop data by finding :
    - the closest bus stop from the store (create a feature the minimal distance then)
    - the mean distance of every bus stop in 1km radius
    for each category of bus stop
    """
    df['lat_processed'] = df.lat_processed * 10
    df['lon_processed'] = df.lon_processed * 10

    categories = ['Mangler viktighetsnivå',
                  'Standard holdeplass',
                  'Lokalt knutepunkt',
                  'Nasjonalt knutepunkt',
                  'Regionalt knutepunkt',
                  'Annen viktig holdeplass']

    new_bs_features = pd.DataFrame(df.store_id)

    df_bus_tmp = df_bus_stops.loc[:, ['busstop_id']]
    df_bus_tmp[['lon_processed', 'lat_processed']] = df_bus_stops['geometry'].str.extract(
        r'(?P<lat>[0-9]*[.]?[0-9]+)\s(?P<lon>[0-9]*[.]?[0-9]+)', expand=True)
    df_bus_tmp['lon_processed'] = pd.to_numeric(df_bus_tmp['lon_processed']) * 6.4757 * 10  # value in km
    df_bus_tmp['lat_processed'] = pd.to_numeric(df_bus_tmp['lat_processed']) * 11.112 * 10  # value in km

    mat = cdist(df_bus_tmp[['lat_processed', 'lon_processed']], df[['lat_processed', 'lon_processed']], metric='euclidean')
    correlation_dist = pd.DataFrame(mat, index=df_bus_tmp['busstop_id'], columns=df['store_id'])
    new_bs_features = pd.merge(new_bs_features,
                               pd.DataFrame(correlation_dist.min(),
                                            columns=['BS_closest']),
                               on='store_id', how='left')
    new_bs_features = pd.merge(new_bs_features,
                               pd.DataFrame(correlation_dist[correlation_dist < 1].mean(),
                                            columns=['BS_mean_1km']),
                               on='store_id', how='left')
    new_bs_features = pd.merge(new_bs_features,
                               pd.DataFrame(correlation_dist[correlation_dist < 0.5].count(),
                                                columns=['number_BS_500m']), on='store_id')

    for category in categories:
        df_bus_tmp = df_bus_stops[df_bus_stops['importance_level'] == category].loc[:, ['busstop_id']]
        df_bus_tmp[['lon_processed', 'lat_processed']] = df_bus_stops['geometry'].str.extract(
                                                                r'(?P<lat>[0-9]*[.]?[0-9]+)\s(?P<lon>[0-9]*[.]?[0-9]+)',
                                                                expand=True)
        df_bus_tmp['lon_processed'] = pd.to_numeric(df_bus_tmp['lon_processed']) * 6.4757 * 10    # value in km
        df_bus_tmp['lat_processed'] = pd.to_numeric(df_bus_tmp['lat_processed']) * 11.112 * 10    # value in km

        mat = cdist(df_bus_tmp[['lat_processed', 'lat_processed']], df[['lat_processed', 'lon_processed']], metric='euclidean')
        correlation_dist = pd.DataFrame(mat, index=df_bus_tmp['busstop_id'], columns=df['store_id'])
        new_bs_features = pd.merge(new_bs_features,
                                   pd.DataFrame(correlation_dist.min(),
                                                columns=['BS_closest_' + category.lower().replace(' ', '_')]),
                                   on='store_id', how='left')
        new_bs_features = pd.merge(new_bs_features,
                                   pd.DataFrame(correlation_dist[correlation_dist < 1].mean(),
                                                columns=['BS_mean_1km_'+category.lower().replace(' ', '_')]),
                                   on='store_id', how='left')
    return new_bs_features.fillna(0)


def fix_municipalities(current_df):
    # Get the rows with missing municipality
    df_missing_mun = current_df[current_df["municipality_name"].isna()]
    # Create a copy of the current df and drop row where mun = NaN + Reset index
    current_df_copy = current_df.copy().dropna(subset=['municipality_name'])
    current_df_copy = current_df_copy.reset_index(drop=True)
    # For each missing municipality
    for index, row in df_missing_mun.iterrows():
        # Create a df with the the difference with the loc of the current store and all the others stores
        tmp_df = pd.concat([current_df_copy.loc[:, ["lat_processed"]] - row.lat_processed,
                            current_df_copy.loc[:, ["lon_processed"]] - row.lon_processed], axis=1)
        # Find the idx of the one with the smallest error (the closest from the other)
        idx = np.argmin(np.linalg.norm(tmp_df.to_numpy(), axis=1))
        # Retrieve the municipality of the closest one and input it in the missing one
        current_df.loc[index, "municipality_name"] = current_df_copy.loc[idx, "municipality_name"]
    return current_df


def keep_only_use_features(current_df, features):
    return current_df.loc[:, features]


def drop_non_use_features(current_df):
    return current_df.drop([
                        'year',
                        'store_id',
                        'store_name',
                        'address',
                        'sales_channel_name',
                        'chain_name',
                        'mall_name',
                        'plaace_hierarchy_id',
                        'lv1_desc',
                        'lv2_desc',
                        'lv3_desc',
                        'municipality_name',
                        # 'lat',
                        # 'lon',
                        # 'lat_processed',
                        # 'lon_processed',
                        # 'total_population',
                        # 'area_km2',
                        # 'BS_closest_standard_holdeplass',
                        # 'BS_mean_1km_lokalt_knutepunkt',
                        # 'BS_mean_1km_nasjonalt_knutepunkt',
                        # 'BS_mean_1km_nasjonalt_knutepunkt',
                        # 'BS_mean_1km_annen_viktig_holdeplass'
                        ], axis=1)


def label_uniformier(array_train, array_test):
    """
    Take the unique values from the train and test part to combine it in a single array.
    Useful to fit the label encoder and don't do a mess during the transform (previously fit_transform that was confusing)
    """
    label_encoder = LabelEncoder()
    labels = np.asarray(list(array_train.unique()) + list(set(array_test.unique()) - set(array_train.unique())))
    label_encoder.fit(labels)
    return label_encoder


def encode_feature(df_train, df_test, feature_name):
    le = label_uniformier(df_train[feature_name], df_test[feature_name])
    df_train['encoded_' + feature_name] = le.transform(df_train[feature_name])
    df_test['encoded_' + feature_name] = le.transform(df_test[feature_name])
    return df_train, df_test


def fill_nan_mean(current_df):
    return current_df.apply(lambda x: x.fillna(x.mean()), axis=0)


def store_secret_feature(df):
    tmp_df = df.loc[:, ['store_id', ]]
    tmp_df[['SI_p1', 'SI_p2', 'SI_p3']] = df['store_id'].str.extract(r'(?P<p1>[0-9]+)-+(?P<p2>[0-9]+)-+(?P<p3>[0-9]+)', expand=True)
    tmp_df['SI_all'] = tmp_df[['SI_p1', 'SI_p2', 'SI_p3']].stack().groupby(level=0).apply(''.join)
    tmp_df[['SI_p1', 'SI_p2', 'SI_p3']] = tmp_df[['SI_p1', 'SI_p2', 'SI_p3']].apply(pd.to_numeric)
    tmp_df['SI_all'] = tmp_df['SI_all'].astype('float')
    return tmp_df


def lat_lon_precisionless(df):
    tmp_df = df.loc[:, ['store_id', 'lat', 'lon']]
    tmp_df[['lat', 'lon']] = tmp_df[['lat', 'lon']]
    tmp_df[['lat_reduced', 'lon_reduced']] = tmp_df[['lat', 'lon']].astype('int')
    return tmp_df[['store_id', 'lat_reduced', 'lon_reduced']]


def scale_values(df_train, df_test, Y_train):
    scaler = StandardScaler()
    df_train[df_train.columns] = scaler.fit_transform(df_train)
    df_test[df_train.columns] = scaler.transform(df_test)

    scaler = StandardScaler()
    Y_train = np.log10(Y_train + 1)

    # Y_train = scaler.fit_transform(Y_train)

    return df_train, df_test, Y_train, scaler


def extract_revenue(df_train):
    return df_train.loc[:, ['revenue', ]], df_train.drop(['revenue'], axis=1)


def features_engineering(df_train, df_test):
    df_bus_stops = pd.read_csv("../data/busstops_norway.csv")
    df_grunnkrets_age_dist = pd.read_csv("../data/grunnkrets_age_distribution.csv")
    df_grunnkrets_house_pers = pd.read_csv("../data/grunnkrets_households_num_persons.csv")
    df_grunnkrets_income_house = pd.read_csv("../data/grunnkrets_income_households.csv")
    df_grunnkrets_stripped = pd.read_csv("../data/grunnkrets_norway_stripped.csv")
    df_plaace_hierarchy = pd.read_csv("../data/plaace_hierarchy.csv")

    df_grunnkrets_stripped = fix_year(df_grunnkrets_stripped)
    df_grunnkrets_age_dist = fix_year(df_grunnkrets_age_dist)
    df_grunnkrets_house_pers = fix_year(df_grunnkrets_house_pers)
    df_grunnkrets_income_house = fix_year(df_grunnkrets_income_house)

    df_train = pd.merge(df_train, store_secret_feature(df_train), how="left", on="store_id")
    df_test = pd.merge(df_test, store_secret_feature(df_test), how="left", on="store_id")

    df_train = pd.merge(df_train, lat_lon_precisionless(df_train), how="left", on="store_id")
    df_test = pd.merge(df_test, lat_lon_precisionless(df_test), how="left", on="store_id")
    df_train['latxlat'] = df_train['lat_reduced']*df_train['lon_reduced']
    df_test['latxlat'] = df_test['lat_reduced']*df_test['lon_reduced']

    df_train = pd.merge(df_train, create_population_repartition(df_grunnkrets_house_pers), how="left", on="grunnkrets_id")
    df_test = pd.merge(df_test, create_population_repartition(df_grunnkrets_house_pers), how="left", on="grunnkrets_id")

    df_train = pd.merge(df_train, create_income_repartition(df_grunnkrets_income_house), how="left",on="grunnkrets_id")
    df_test = pd.merge(df_test, create_income_repartition(df_grunnkrets_income_house), how="left", on="grunnkrets_id")

    df_train = pd.merge(df_train, create_hierarchy(df_plaace_hierarchy), how="left", on="plaace_hierarchy_id")
    df_test = pd.merge(df_test, create_hierarchy(df_plaace_hierarchy), how="left", on="plaace_hierarchy_id")

    df_train = pd.merge(df_train, create_grunnkret_geodata(df_grunnkrets_stripped), how="left", on="grunnkrets_id")
    df_test = pd.merge(df_test, create_grunnkret_geodata(df_grunnkrets_stripped), how="left", on="grunnkrets_id")

    df_train = pd.merge(df_train, create_population_age(df_grunnkrets_age_dist), how="left", on="grunnkrets_id")
    df_test = pd.merge(df_test, create_population_age(df_grunnkrets_age_dist), how="left", on="grunnkrets_id")

    df_train["population_density"] = df_train["total_population"] / df_train["area_km2"]
    df_test["population_density"] = df_test["total_population"] / df_test["area_km2"]

    df_train = pd.merge(df_train, find_closest_bus_stop(df_train, df_bus_stops), how="left", on="store_id")
    df_test = pd.merge(df_test, find_closest_bus_stop(df_test, df_bus_stops), how="left", on="store_id")

    df_train['mall_name'] = df_train['mall_name'].fillna('0')
    df_test['mall_name'] = df_test['mall_name'].fillna('0')
    df_train, df_test = encode_feature(df_train, df_test, 'mall_name')

    df_train['chain_name'] = df_train['chain_name'].fillna('0')
    df_test['chain_name'] = df_test['chain_name'].fillna('0')
    df_train, df_test = encode_feature(df_train, df_test, 'chain_name')

    df_train, df_test = encode_feature(df_train, df_test, 'sales_channel_name')

    df_train, df_test = encode_feature(df_train, df_test, 'lv3_desc')
    df_train, df_test = encode_feature(df_train, df_test, 'lv2_desc')
    df_train, df_test = encode_feature(df_train, df_test, 'lv1_desc')

    df_train = fix_municipalities(df_train)
    df_test = fix_municipalities(df_test)
    df_train, df_test = encode_feature(df_train, df_test, 'municipality_name')

    Y_train, df_train = extract_revenue(df_train)

    df_train = drop_non_use_features(df_train)
    df_test = drop_non_use_features(df_test)

    X_train = fill_nan_mean(df_train)
    X_test = fill_nan_mean(df_test)

    X_train, X_test, Y_train, scaler_revenue = scale_values(X_train, X_test, Y_train)

    return X_train, X_test, Y_train, scaler_revenue
