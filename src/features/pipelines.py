from src.features import (read_data, drop_columns, drop_nans, rename_cols, rename_col, drop_duplicates, drop_rows,
						  filter_rows, clean_address, clean_district, impute_district, clean_price, clean_sqm,
						  clean_outliers, create_price_sqm, create_heating, create_heating_type, create_heating_source,
						  create_energy_class, create_listing_date, create_elevator, create_disabled_access,
						  create_floor, create_garage_parking, create_external_parking, create_num_bathrooms,
						  create_num_rooms, string_parser, create_features_list, create_other_features, data_split,
						  create_pipeline, CustomEncoder, ColumnSelector
						  )
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer


def load_data(path):
	filenames = ['/caratteristiche.xlsx', '/costi.xlsx', '/efficienza_energetica.xlsx']
	dfs = []
	for filename in filenames:
		dfs.append(read_data(path + filename))

	drop_cols_1 = ['unità', 'Data di inizio lavori e di consegna prevista', 'Dati catastali']
	keep_cols = ['prezzo', 'informazioni catastali', 'spese condominio']
	drop_cols_2 = ['numero immobili', 'offerta minima', 'rialzo minimo', 'Spesa prenota debito',
				   'Contributo non dovuto',
				   'Tipo vendita', 'data vendita']

	df1 = dfs[0].drop(columns=drop_cols_1)
	df2 = dfs[1][keep_cols]
	df3 = dfs[2].drop(columns=drop_cols_2)

	data = df1.join(df2).join(df3)
	return data


subset = ['Zona', 'Superficie']
drop_cols = ['Indirizzo', 'Prezzo', 'Superficie', 'Immobile garantito', 'Indice prest. energetica rinnovabile',
			 'Prestazione energetica del fabbricato', 'Certificazione energetica', 'Disponibilità', 'Contratto',
			 'Informazioni catastali', 'Spese condominio', 'Riscaldamento', 'Climatizzazione', 'Efficienza energetica',
			 'Riferimento e data annuncio', 'Piano', 'Totale piani edificio', 'Posti auto', 'Locali', 'Num_altri',
			 'Num_camere_letto', 'Num_locali', 'Altre caratteristiche', 'Altre_caratteristiche', 'Data_annuncio']

cleaning_pipeline = create_pipeline([
	rename_cols,
	drop_duplicates,
	drop_rows([1279, 4985, 9049]),
	clean_address,
	impute_district,  # Manual impute before splitting so no data leakage here.
	drop_nans(subset),
	clean_district,
	filter_rows('Prezzo', 'Prezzo su richiesta'),
	clean_price,
	clean_sqm,
	clean_outliers('Superficie_m2', 240018.0, 240.0),
	clean_outliers('Superficie_m2', 11350.0, 1135.0),
	clean_outliers('Superficie_m2', 6437.0, 64.0),
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
	drop_columns(drop_cols),
	rename_col('Piani', 'Piano'),
	rename_col('Anno di costruzione', 'Anno_costruzione'),
	rename_col('Tipo proprietà', 'Tipo_proprietà'),
])

cat_features = ['Tipologia', 'Zona', 'Stato', 'Tipo_proprietà', 'Riscaldamento_A_C', 'Tipo_riscaldamento',
				'Alimentazione_riscaldamento', 'Classe_energetica', 'Piano']
num_features = ['Prezzo_EUR', 'Superficie_m2', 'Num_bagni', 'Num_tot_locali', 'Anno_costruzione']

cat_transformer = Pipeline([
	('encoding', CustomEncoder()),
	('imputing', IterativeImputer(initial_strategy='most_frequent', max_iter=10, random_state=0))
])

num_transformer = Pipeline([
	('scaling', StandardScaler()),
	('imputing', IterativeImputer(initial_strategy='mean', max_iter=10, random_state=0))
])

preprocessing_pipeline = ColumnTransformer([
	('categoricals', cat_transformer, cat_features),
	('numericals', num_transformer, num_features)
],
	remainder='passthrough'
)
