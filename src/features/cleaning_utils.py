import numpy as np
import pandas as pd
import pickle
import yaml
from scipy import stats
from sklearn.model_selection import train_test_split


def parse_config(config_file):
	"""Parse the config file containing all the workflow parameters."""
	with open(config_file, "rb") as f:
		config = yaml.safe_load(f)
	return config


def read_data(filename):
	"""Read data and store it in a DataFrame."""
	return pd.read_excel(filename)


def drop_columns(cols):
	"""Drop columns in a DataFrame."""

	def dropper(data):
		return data.drop(columns=cols)

	return dropper


def drop_nans(subset):
	"""Drop rows of a DataFrame with NaN values in a given column and reset the index."""

	def dropper(data):
		return data.dropna(subset=subset).reset_index(drop=True)

	return dropper


def rename_cols(data):
	"""Rename columns of a DataFrame by capitalizing them."""
	return data.rename(columns=str.capitalize)


def rename_col(col, new_col):
	"""Rename specific column of a DataFrame."""

	def renamer(data):
		data = data.rename(columns={col: new_col})
		return data

	return renamer


def drop_duplicates(data):
	"""Drop duplicate rows in a DataFrame."""
	return data.drop_duplicates()


def drop_rows(rows):
	"""Drop rows in a DataFrame and reset the index."""

	def dropper(data):
		return data.drop(rows).reset_index(drop=True)

	return dropper


def filter_rows(col, value):
	"""Filter DataFrame by removing rows with unwanted values for a given column."""

	def filterer(data):
		return data.loc[data[col] != value]

	return filterer


def clean_address(data):
	"""Clean the address feature."""
	data['Indirizzo'] = data['Indirizzo'].str.replace('[', '').str.replace(']', '').str.replace('\'', '')
	return data


def impute_district(data):
	"""Manually impute values for the district feature."""
	data.loc[data['Indirizzo'] == 'Firenze, via vittorio emanuele orlando', 'Zona'] = 'Coverciano Bellariva'
	data.loc[data['Indirizzo'] == 'Firenze, via borgo la noce', 'Zona'] = 'Centro'
	data.loc[data['Indirizzo'] == 'Firenze, via Cigoli 31', 'Zona'] = 'L Isolotto'
	data.loc[data['Indirizzo'] == 'Firenze, via impruneta per mezzomonte', 'Zona'] = 'Bellosguardo Galluzzo'
	data.loc[data['Indirizzo'] == 'Firenze, via gioberti', 'Zona'] = 'Campo Di Marte Liberta'
	data.loc[data['Indirizzo'] == 'Firenze, via dei cioli 50', 'Zona'] = 'Settignano Rovezzano'
	data.loc[data['Indirizzo'] == 'Firenze, via spinucci 1', 'Zona'] = 'Serpiolle Careggi'
	data.loc[data['Indirizzo'] == 'Firenze, "via lungo laffrico 50"', 'Zona'] = 'Coverciano Bellariva'
	data.loc[data['Indirizzo'] == 'Firenze, via lippi', 'Zona'] = 'Legnaia Soffiano'
	data.loc[data['Indirizzo'] == 'Firenze, cairoli', 'Zona'] = 'Campo Di Marte Liberta'
	data.loc[data['Indirizzo'] == 'Firenze, via aretina', 'Zona'] = 'Coverciano Bellariva'
	data.loc[data['Indirizzo'] == 'Firenze, via Fra Bartolommeo  40', 'Zona'] = 'Campo Di Marte Liberta'
	data.loc[data['Indirizzo'] == 'Firenze, viale don minzoni 1', 'Zona'] = 'Campo Di Marte Liberta'
	data.loc[data['Indirizzo'] == 'Firenze, viale don minzoni  1', 'Zona'] = 'Campo Di Marte Liberta'
	data.loc[data['Indirizzo'] == 'Firenze, piazza beccaria', 'Zona'] = 'Campo Di Marte Liberta'
	data.loc[data['Indirizzo'] == 'Firenze, via san zanobi', 'Zona'] = 'Centro'
	data.loc[data['Indirizzo'] == 'Firenze, Piazzale Michelangelo', 'Zona'] = 'Michelangelo Porta Romana'
	data.loc[data['Indirizzo'] == 'Firenze, Via del Paradiso', 'Zona'] = 'Zona Firenze Sud'
	data.loc[data['Indirizzo'] == 'Firenze, via di Canonica', 'Zona'] = 'Centro'
	data.loc[data['Indirizzo'] == 'Firenze, Via di Canonica', 'Zona'] = 'Centro'
	data.loc[data['Indirizzo'] == 'Firenze, Via Frusa', 'Zona'] = 'Campo Di Marte Liberta'
	data.loc[data['Indirizzo'] == 'Firenze, Via Vespucci', 'Zona'] = 'Firenze Nord'
	data.loc[data['Indirizzo'] == 'Firenze, via baracca  148', 'Zona'] = 'Firenze Nord'
	data.loc[data['Indirizzo'] == 'Firenze, via dei Tavolini 1', 'Zona'] = 'Centro'
	data.loc[data['Indirizzo'] == 'Firenze, via Pisana 980', 'Zona'] = 'Ugnano Mantignano'
	data.loc[data['Indirizzo'] == 'Firenze, VIA SENESE', 'Zona'] = 'Bellosguardo Galluzzo'
	data.loc[data['Indirizzo'] == 'Firenze, "piazza dAzeglio"', 'Zona'] = 'Centro'
	data.loc[data['Indirizzo'] == 'Firenze, "Piazza dazeglio"', 'Zona'] = 'Centro'
	return data


def clean_district(data):
	"""Clean the district feature."""
	data['Zona'] = data['Zona'].str.replace('-', ' ').str.replace('/', '').str.title().replace(
		{'L Isolotto': 'L\'Isolotto'})
	return data


def clean_price(data):
	"""Clean the price feature."""
	data['Prezzo_EUR'] = data['Prezzo'].str.split('€').str[1].str.replace('.', '').astype('float')
	return data


def clean_sqm(data):
	"""Clean the square meters feature."""
	mask = data['Superficie'].str.contains(r'\|', na=False)
	data.loc[mask, 'Superficie_m2'] = (data.loc[mask, 'Superficie']
									   .str.extract(r'commerciale (\d+\.?\d*)')[0]
									   .str.replace('.', '')
									   )
	data.loc[~mask, 'Superficie_m2'] = (data.loc[~mask, 'Superficie']
										.str.extract(r'(\d+\.?\d*)')[0]
										.str.replace('.', '')
										)
	data['Superficie_m2'] = data['Superficie_m2'].astype('float')
	return data


def clean_condition(data):
	"""Clean the condition feature."""
	data['Stato'] = data['Stato'].str.replace(' ', '_')
	return data


def clean_outliers(col, current_value, new_value):
	"""Replace outlier values by new values."""

	def cleaner(data):
		data.loc[data[col] == current_value, col] = new_value
		return data

	return cleaner


def remove_outliers(cols):
	"""Remove all rows from a DataFrame that contain outliers in log-transformed selected columns by using a z
	statistic."""

	def remover(data):
		data = data.loc[(np.abs(stats.zscore(np.log(data[cols]))) < 3).all(axis=1)]
		return data

	return remover


def create_price_sqm(data):
	"""Create price per square meter feature."""
	data['Prezzo_per_m2'] = data['Prezzo_EUR'] / data['Superficie_m2']
	return data


def create_property(data):
	"""Create whole/naked property feature."""
	data['Proprietà_I_N'] = (data['Tipo proprietà']
							 .str.extract(r'(Intera proprietà|Nuda proprietà|Multiproprietà)', expand=False)
							 .str.lower())

	mask = data['Tipo proprietà'].notnull() & data['Proprietà_I_N'].isna()
	data.loc[mask, 'Proprietà_I_N'] = 'intera proprietà'
	return data


def create_property_type(data):
	"""Create property type feature."""
	data['Tipo_proprietà'] = (data['Tipo proprietà']
							  .str.split(',').str[-1]
							  .str.strip()
							  .str.lower()
							  .str.extract('(.*immobile.*)'))
	return data


def create_contract_type(data):
	"""Create contract type feature."""
	mask = data['Contratto'].str.match('.*a reddito.*')

	data.loc[mask, 'A_reddito'] = 'sì'
	data['A_reddito'] = data['A_reddito'].fillna('no')
	return data


def create_house_type(data):
	"""Create house type feature."""
	data['Tipologia'] = data['Tipologia'].str.lower()

	# Define masks
	mask1 = data['Tipologia'].str.match('.*appartamento.*')
	mask2 = data['Tipologia'].str.match('.*terratetto.*')
	mask3 = data['Tipologia'].str.match('(.*villa.*pluri.*)|(.*villa.*bifa.*)')
	mask4 = data['Tipologia'] == 'villa a schiera'

	# Apply masks
	data.loc[mask1, 'Tipologia'] = 'appartamento'
	data.loc[mask2, 'Tipologia'] = 'terratetto'
	data.loc[mask3, 'Tipologia'] = 'villa plurifamiliare'
	data.loc[mask4, 'Tipologia'] = 'villa unifamiliare'

	# Value_counts lower than or equal to 11 set to "other"
	mask5 = data['Tipologia'].value_counts() <= 11
	house_list = data['Tipologia'].value_counts().loc[mask5].index.values
	data.loc[data['Tipologia'].isin(house_list), 'Tipologia'] = 'altro'
	return data


def create_heating(data):
	"""Create centralized/autonomous heating feature."""
	data['Riscaldamento_A_C'] = data['Riscaldamento'].str.split(',').str[0].str.lower()

	# Impute by constant value which does not cause data leakage
	data['Riscaldamento_A_C'] = data['Riscaldamento_A_C'].fillna('centralizzato')
	return data


def create_heating_type(data):
	"""Create heating type feature."""
	data['Tipo_riscaldamento'] = data['Riscaldamento'].str.extract(r'(radiatori|aria|pavimento|stufa)')
	return data


def create_heating_source(data):
	"""Create heating source feature."""
	data['Alimentazione_riscaldamento'] = data['Riscaldamento'].str.extract(r'(metano|gas|gasolio|pompa di calore'
																			'|elettrica|fotovoltaico|pellet|gpl|solare)')
	return data


def create_energy_class(data):
	"""Create energy class feature."""
	data['Classe_energetica'] = data['Efficienza energetica'].str.extract(r'([A-G]\d?)')
	return data


def create_listing_date(data):
	"""Create listing date feature."""
	data['Data_annuncio'] = (data['Riferimento e data annuncio']
							 .str.split('-')
							 .str[-1]
							 .str.strip()
							 .astype('datetime64[D]')
							 )
	return data


def create_elevator(data):
	"""Create elevator feature."""
	mask = data['Piano'].str.match(r'.*ascensore.*').fillna(False)
	data.loc[mask, 'Ascensore'] = 'sì'

	# Impute by constant value which does not cause data leakage
	data['Ascensore'] = data['Ascensore'].fillna('no')
	return data


def create_disabled_access(data):
	"""Create disabled access feature."""
	mask = data['Piano'].str.match(r'.*accesso disabili.*').fillna(False)
	data.loc[mask, 'Accesso_disabili'] = 'sì'

	# Impute by constant value which does not cause data leakage
	data['Accesso_disabili'] = data['Accesso_disabili'].fillna('no')
	return data


def create_floor(data):  # TODO: there has got to be a better way to create this feature
	"""Create floor feature."""
	mask1 = data['Piano'].str.match(r'.*\d+°.*').fillna(False)
	mask2 = data['Piano'].str.match(r'.*[pP]iano terra.*').fillna(False)
	mask3 = data['Piano'].str.match(r'[uU]ltimo.*').fillna(False)
	mask4 = data['Piano'].str.match(r'[pP]iano rialzato.*').fillna(False)
	mask5 = data['Piano'].str.match(r'.*[sS]eminterrato.*').fillna(False)
	mask6 = data['Piano'].str.match(r'.*[aA]mmezzato.*').fillna(False)
	mask7 = data['Piano'].str.match(r'.*[iI]nterrato.*').fillna(False)
	mask8 = data['Piano'].str.match(r'(.*da.*|.*più livelli.*)').fillna(False)
	mask9 = data['Piano'].str.match(r'.*Oltre il decimo piano.*').fillna(False)

	data.loc[mask1, 'Piani'] = data.loc[mask1, 'Piano'].str.extract(r'(\d+°)').values
	data.loc[mask2, 'Piani'] = 'terra'
	data.loc[mask3, 'Piani'] = 'ultimo'
	data.loc[mask4, 'Piani'] = 'rialzato'
	data.loc[mask5, 'Piani'] = 'seminterrato'
	data.loc[mask6, 'Piani'] = 'ammezzato'
	data.loc[mask7, 'Piani'] = 'interrato'
	data.loc[mask8, 'Piani'] = 'più livelli'
	data.loc[mask9, 'Piani'] = 'oltre il decimo'
	return data


def create_garage_parking(data):
	"""Create garage parking feature."""
	data['Posti_garage'] = data['Posti auto'].str.extract(r'(\d).*garage\/box').astype('float')

	# Impute by constant value which does not cause data leakage
	data['Posti_garage'] = data['Posti_garage'].fillna(0)
	return data


def create_external_parking(data):
	"""Create external parking feature."""
	data['Posti_esterni'] = data['Posti auto'].str.extract(r'(\d+).*esterno').astype('float')

	# Impute by constant value which does not cause data leakage
	data['Posti_esterni'] = data['Posti_esterni'].fillna(0)
	return data


def create_num_bathrooms(data):
	"""Create number of bathrooms feature."""
	data['Num_bagni'] = data['Locali'].str.extract(r'(\d\+?) bagn\w')
	data.loc[data['Num_bagni'] == '3+', 'Num_bagni'] = 4  # Set 3+ toilets to 4
	data['Num_bagni'] = data['Num_bagni'].astype('float')
	return data


def create_num_rooms(data):
	"""Create number of rooms feature."""
	# All types of rooms
	data['Num_altri'] = data['Locali'].str.extract(r'(\d+\+?) altr\w').astype('float')
	data['Num_altri'] = data['Num_altri'].fillna(0)  # Set NaNs to 0 to be able to sum

	data['Num_camere_letto'] = data['Locali'].str.extract(r'(\d+\+?) camer\w da letto').astype('float')
	data['Num_camere_letto'] = data['Num_camere_letto'].fillna(0)  # Set NaNs to 0 to be able to sum

	data['Num_locali'] = data['Locali'].str.extract(r'(\d+\+?) local\w').astype('float')
	data['Num_locali'] = data['Num_locali'].fillna(0)  # Set NaNs to 0 to be able to sum

	# Total number of rooms
	data['Num_tot_locali'] = data['Num_locali'] + data['Num_camere_letto'] + data['Num_altri']

	# Set values of "0" to "np.nan" for imputation later
	data.loc[data['Num_tot_locali'] == 0, 'Num_tot_locali'] = np.nan
	return data


def string_parser(row):
	"""Parse string values in other features column to extract all the features and store them in a list."""
	if pd.notnull(row['Altre caratteristiche']):
		string_list = row['Altre caratteristiche'].split('\n')
		row['Altre_caratteristiche'] = ([string.strip().replace(' ', '_')
										 for string in string_list if string.strip() != ''])
	else:
		row['Altre_caratteristiche'] = []
	return row['Altre_caratteristiche']


def create_features_list(data):
	"""Create list of all possible features present in "Altre caratteristiche" column."""
	features_list = []
	for idx, series in data.iterrows():
		for feature in series.loc['Altre_caratteristiche']:
			if feature not in features_list:
				features_list.append(feature)
		else:
			continue
	return features_list


def create_other_features(parser, list_creator):
	"""Create columns for each other feature extracted."""

	def creator(data):
		# Parse features
		data['Altre_caratteristiche'] = data['Altre caratteristiche'].copy()
		data['Altre_caratteristiche'] = data.apply(parser, axis=1)

		# Create features list
		features_list = list_creator(data)

		# Create one-hot encoded column for each extracted feature
		for feature in features_list:
			mask = data['Altre_caratteristiche'].apply(lambda x: feature in x)
			data.loc[mask, feature] = 'sì'
			data[feature] = data[feature].fillna('no')
		return data

	return creator


def data_split(test_size=0.2):
	"""Split the dataset into a training and a test set."""

	def splitter(data):
		return train_test_split(data, test_size=test_size, random_state=0)

	return splitter


def save_data(data, path, name):
	"""Save model as pickle file."""
	with open(path + name, 'wb') as f:
		pickle.dump(data, f, protocol=4)


def create_pipeline(list_functions):
	"""Pipeline function for data cleaning steps."""

	def pipeline(data):
		out = data
		for function in list_functions:
			out = function(out)
		return out

	return pipeline
