from src.features import (read_data, drop_columns, drop_nans, rename_cols, rename_col, drop_duplicates, drop_rows,
						  filter_rows, clean_address, clean_district, impute_district, clean_price, clean_sqm,
						  clean_state,
						  remove_outliers, create_price_sqm, create_heating, create_heating_type, create_heating_source,
						  create_energy_class, create_listing_date, create_elevator, create_disabled_access,
						  create_floor, create_garage_parking, create_external_parking, create_num_bathrooms,
						  create_num_rooms, string_parser, create_features_list, create_other_features, create_pipeline,
						  CustomEncoder
						  )
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer


def load_data(path, config):
	filenames = config['cleaning']['filenames']

	dfs = []
	for filename in filenames:
		dfs.append(read_data(path + filename))

	drop_cols_1 = config['cleaning']['drop_cols_1']
	keep_cols = config['cleaning']['keep_cols']
	drop_cols_2 = config['cleaning']['drop_cols_2']

	df1 = dfs[0].drop(columns=drop_cols_1)
	df2 = dfs[1][keep_cols]
	df3 = dfs[2].drop(columns=drop_cols_2)

	data = df1.join(df2).join(df3)
	return data


def clean_data(config):
	cleaning_pipeline = create_pipeline([
		rename_cols,
		drop_duplicates,
		drop_rows([1279, 4985, 9049]),
		clean_address,
		impute_district,  # Manual impute before splitting so no data leakage here.
		drop_nans(config['cleaning']['subset']),
		clean_district,
		filter_rows('Prezzo', 'Prezzo su richiesta'),
		clean_price,
		clean_sqm,
		clean_state,
		remove_outliers(['Prezzo_EUR', 'Superficie_m2']),
		create_price_sqm,
		create_heating,
		create_heating_type,
		create_heating_source,
		create_energy_class,
		create_listing_date,
		create_elevator,
		create_disabled_access,
		create_floor,
		create_garage_parking,
		create_external_parking,
		create_num_bathrooms,
		create_num_rooms,
		create_other_features(string_parser, create_features_list),
		drop_columns(config['cleaning']['drop_cols']),
		rename_col('Piani', 'Piano'),
		rename_col('Anno di costruzione', 'Anno_costruzione'),
		rename_col('Tipo proprietà', 'Tipo_proprietà'),
	])
	return cleaning_pipeline


# cat_features = None
# num_features = None
#
# cat_transformer = Pipeline([
# 	('label_encoding', CustomEncoder()),
# 	('imputing', IterativeImputer(initial_strategy='most_frequent', max_iter=10, random_state=0)),
# 	('oh_encoding', OneHotEncoder(handle_unknown='ignore'))
# ])
#
# num_transformer = Pipeline([
# 	('scaling', StandardScaler()),
# 	('imputing', IterativeImputer(initial_strategy='mean', max_iter=10, random_state=0))
# ])
#
# preprocessing_pipeline = ColumnTransformer([
# 	('categoricals', cat_transformer, cat_features),
# 	('numericals', num_transformer, num_features)
# ],
# 	remainder='passthrough'
# )
