from src.features import (read_data, drop_columns, drop_nans, rename_cols, rename_col, drop_duplicates, drop_rows,
						  filter_rows, clean_address, clean_district, impute_district, clean_price, clean_sqm,
						  clean_condition, remove_outliers_iqr, create_price_sqm, create_property_class,
						  create_property_type, create_contract_type, create_house_type, create_year_bins,
						  create_heating, create_heating_type, create_heating_source, create_energy_class,
						  create_listing_date, create_elevator, create_disabled_access, create_floor,
						  create_garage_parking, create_external_parking, create_num_bathrooms, create_num_rooms,
						  string_parser, create_parsed_features, create_windows, create_garden, create_furnished,
						  create_terrace, create_exposure, create_other_features, create_pipeline)


def load_raw_data(path, config):
	dfs = []
	for filename in config['cleaning']['filenames']:
		dfs.append(read_data(path + filename))

	df1 = dfs[0].drop(columns=config['cleaning']['drop_cols_1'])
	df2 = dfs[1][config['cleaning']['keep_cols']]
	df3 = dfs[2].drop(columns=config['cleaning']['drop_cols_2'])

	df = df1.join(df2).join(df3)
	return df


def clean_data(config):
	cleaning_pipeline = create_pipeline([
		rename_cols,
		drop_duplicates,
		drop_rows([1279, 4985, 9049]),
		clean_address,
		impute_district,  # Manual impute before splitting so no data leakage here
		drop_nans(config['cleaning']['subset']),
		clean_district,
		filter_rows('Prezzo', 'Prezzo su richiesta'),
		clean_price,
		clean_sqm,
		clean_condition,
		remove_outliers_iqr('Prezzo_EUR'),
		remove_outliers_iqr('Superficie_m2'),
		create_price_sqm,
		create_property_class,
		create_property_type,
		create_contract_type,
		create_house_type,
		create_year_bins,
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
		create_parsed_features(string_parser),
		create_windows,
		create_garden,
		create_furnished,
		create_terrace,
		create_exposure,
		create_other_features,
		drop_columns(config['cleaning']['drop_cols']),
		rename_col('Piani', 'Piano'),
	])
	return cleaning_pipeline
