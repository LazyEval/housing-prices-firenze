import numpy as np
import pandas as pd
import yaml
from scipy import stats

rng = np.random.RandomState(0)


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
	data['Zona'] = (data['Zona']
					.str.replace('-', ' ')
					.str.replace('/', '')
					.str.title()
					.replace({'L Isolotto': 'L\'Isolotto'}))
	return data


def clean_price(data):
	"""Clean the price feature."""
	data['Prezzo'] = data['Prezzo'].str.split('€').str[1].str.replace('.', '').astype('float')
	return data


def clean_sqm(data):
	"""Clean the square meters feature."""
	mask = data['Superficie'].str.contains('\|', na=False)

	data['Superficie'] = (data['Superficie']
						  .mask(mask, data['Superficie']
								.str.extract('commerciale (\d+\.?\d*)', expand=False)
								.str.replace('.', ''))
						  .where(mask, data['Superficie']
								 .str.extract('(\d+\.?\d*)', expand=False)
								 .str.replace('.', ''))
						  .astype('float64'))
	return data


def clean_condition(data):
	"""Clean the condition feature."""
	data['Stato'] = data['Stato'].str.replace(' / ', '/').str.lower()
	return data


def clean_outliers(col, current_value, new_value):
	"""Replace outlier values by new values."""

	def cleaner(data):
		data.loc[data[col] == current_value, col] = new_value
		return data

	return cleaner


def remove_outliers_iqr(cols, bounds=[.25, .75], k=1.5):
	"""Remove all rows from a DataFrame that contain outliers based on the iqr of a set of columns."""

	def remover(data):
		q = data[cols].quantile(bounds)
		iqr = q.iloc[1] - q.iloc[0]
		mask = (data[cols] >= q.iloc[0] - k * iqr) & (data[cols] <= q.iloc[1] + k * iqr)
		data = data[mask.all(axis=1)]
		return data

	return remover


def remove_outliers_zscore(cols, z=3):
	"""Remove all rows from a DataFrame that contain outliers based on the z-score of the variables."""

	def remover(data):
		data = data.loc[(np.abs(stats.zscore(np.log(data[cols]))) < z).all(axis=1)]
		return data

	return remover


def filter_data(col, min_value, max_value):
	"""Filter DataFrame by min and max value for a specific column."""

	def filter(data):
		data = data.loc[(data[col] > min_value) & (data[col] < max_value)]
		return data

	return filter


def create_price_sqm(data):
	"""Create price per square meter feature."""
	data['Prezzo_per_m2'] = data['Prezzo'] / data['Superficie']
	return data


def create_property_class(data):
	"""Create property type feature."""
	data['Classe_immobile'] = (data['Tipo proprietà']
							   .str.lower()
							   .str.extract('(economica|media|signorile|lusso)', expand=False))
	return data


def create_property_type(data):
	"""Create whole/naked property feature."""
	data['Tipo_proprietà'] = (data['Tipo proprietà']
							  .str.lower()
							  .str.extract('(intera proprietà|nuda proprietà|multiproprietà)', expand=False))

	mask = data['Contratto'].str.contains('a reddito')
	data['Tipo_proprietà'] = data['Tipo_proprietà'].mask(mask, 'a reddito')

	# Assume the remaining property types that are not missing but also not specified are "intera proprietà"
	mask = data['Tipo proprietà'].notnull() & data['Tipo_proprietà'].isna()
	data['Tipo_proprietà'] = data['Tipo_proprietà'].mask(mask, 'intera proprietà')
	return data


def create_house_type(data):
	"""Create house type feature."""
	data['Tipologia_casa'] = data['Tipologia'].str.lower().replace({'appartamento in villa': 'appartamento',
																	'terratetto unifamiliare': 'terratetto',
																	'terratetto plurifamiliare': 'terratetto',
																	'villa bifamiliare': 'villa plurifamiliare',
																	'villa a schiera': 'villa unifamiliare'})

	# Value_counts lower than or equal to 15 set to "other"
	others_list = data['Tipologia'].value_counts().loc[data['Tipologia'].value_counts() < 15].index.values
	mask = data['Tipologia'].isin(others_list)

	data['Tipologia_casa'] = data['Tipologia_casa'].mask(mask, 'altro')
	return data


def create_year_bins(data):
	"""Create binned feature of year of construction."""
	data['Anno_costruzione_bins'] = pd.cut(data['Anno di costruzione'], [0, 1850, 1950, 2000, 2021])
	return data


def create_heating(data):
	"""Create centralized/autonomous heating feature."""
	data['Riscaldamento_A_C'] = (data['Riscaldamento']
								 .str.lower()
								 .str.extract('(centralizzato|autonomo)')
								 .fillna('centralizzato'))
	return data


def create_heating_type(data):
	"""Create heating type feature."""
	data['Tipo_riscaldamento'] = data['Riscaldamento'].str.extract('(radiatori|aria|pavimento|stufa)')
	return data


def create_heating_source(data):
	"""Create heating source feature."""
	data['Alimentazione_riscaldamento'] = data['Riscaldamento'].str.extract('(metano|gas|gasolio|pompa di calore'
																			'|elettrica|fotovoltaico|pellet|gpl|solare)')
	return data


def create_air_conditioning(data):
	"""Create air conditioning feature by extracting relevant information."""
	data['Climatizzazione'] = (data['Climatizzazione']
							   .str.lower()
							   .str.extract('(predisposizione|autonomo|centralizzato)')
							   .fillna('non presente'))
	return data


def create_energy_efficiency(data):
	"""Create energy efficiency feature by grouping the energy classes into three groups."""
	mask1 = data['Efficienza energetica'].str.contains('A\d?', na=False)
	mask2 = data['Efficienza energetica'].str.contains('[B-D]', na=False)
	mask3 = data['Efficienza energetica'].str.contains('[E-G]', na=False)

	conditions = [mask1, mask2, mask3]
	choices = ['alta (A, A+, A1-A4)', 'media (B, C, D)', 'bassa (E, F, G)']

	data['Efficienza_energetica'] = pd.Series(np.select(conditions, choices, np.nan)).replace({'nan': np.nan})
	return data


def create_listing_date(data):
	"""Create listing date feature."""
	data['Data_annuncio'] = (data['Riferimento e data annuncio']
							 .str.split('-')
							 .str[-1]
							 .str.strip()
							 .astype('datetime64[D]'))
	return data


def create_elevator(data):
	"""Create elevator feature."""
	mask = data['Piano'].str.contains('ascensore', na=False)
	data['Ascensore'] = np.where(mask, 'sì', 'no')
	return data


def create_disabled_access(data):
	"""Create disabled access feature."""
	mask = data['Piano'].str.contains('accesso disabili', na=False)
	data['Accesso_disabili'] = np.where(mask, 'sì', 'no')
	return data


def create_floor(data):
	"""Create floor feature."""
	mask1 = data['Piano'].str.lower().str.contains('\d+°|oltre il decimo piano|su più livelli', na=False)
	mask2 = data['Piano'].str.lower().str.contains('seminterrato|interrato|ammezzato', na=False)
	mask3 = data['Piano'].str.lower().str.contains('terra|piano rialzato', na=False)
	mask4 = data['Piano'].str.lower().str.contains('ultimo', na=False)
	mask5 = (data['Totale piani edificio'].str.extract('(\d+)', expand=False) ==
			 data['Piano'].str.extract('(\d+)', expand=False))

	data['Piano'] = (data['Piano']
					 .mask(mask1, 'intermedio')
					 .mask(mask2, 'interrato')
					 .mask(mask3, 'terra')
					 .mask(mask4 | mask5, 'ultimo'))
	return data


def create_garage_parking(data):
	"""Create garage parking feature."""
	data['Posti_garage'] = data['Posti auto'].str.extract('(\d).*garage\/box').astype('float').fillna(0)
	return data


def create_external_parking(data):
	"""Create external parking feature."""
	data['Posti_esterni'] = data['Posti auto'].str.extract('(\d+).*esterno').astype('float').fillna(0)
	return data


def create_num_bathrooms(data):
	"""Create number of bathrooms feature."""
	data['Num_bagni'] = data['Locali'].str.extract(r'(\d\+?) bagn\w')
	data['Num_bagni'] = data['Num_bagni'].mask(data['Num_bagni'] == '3+', 4)  # Set 3+ toilets to 4
	return data


def create_num_rooms(data):
	"""Create number of rooms feature."""
	# All types of rooms
	data['Num_altri'] = data['Locali'].str.extract('(\d+\+?) altr\w').astype('float64').fillna(0)
	data['Num_camere_letto'] = data['Locali'].str.extract('(\d+\+?) camer\w da letto').astype('float64').fillna(0)
	data['Num_locali'] = data['Locali'].str.extract(r'(\d+\+?) local\w').astype('float64').fillna(0)

	# Total number of rooms
	data['Num_tot_locali'] = ((data['Num_altri'] + data['Num_camere_letto'] + data['Num_locali'])
							  .mask(data['Locali'].isna(), np.nan))
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


def create_parsed_features(parser):
	"""Create feature from parsed attributes."""

	def creator(data):
		data['Altre_caratteristiche'] = data['Altre caratteristiche'].copy()
		data['Altre_caratteristiche'] = data.apply(parser, axis=1)
		return data

	return creator


def create_windows(data):
	"""Create windows feature."""
	data['Infissi'] = (data['Altre_caratteristiche']
					   .apply(lambda x: str([y for y in x if 'Infissi' in y]))
					   .str.extract('(doppio|triplo)', expand=False)
					   .fillna('singolo'))
	return data


def create_garden(data):
	"""Create garden feature."""
	data['Giardino'] = (data['Altre_caratteristiche']
						.apply(lambda x: str([y for y in x if 'Giardino' in y]))
						.str.extract('(comune|privato)', expand=False)
						.fillna('no'))
	return data


def create_furnished(data):
	"""Create furnished feature."""
	data['Arredato'] = data['Altre_caratteristiche'].apply(lambda x: str([y for y in x if 'Arredat' in y]))
	data['Arredato'] = data['Arredato'].replace({'[\'Parzialmente Arredato\']': 'parzialmente',
												 '[\'Solo Cucina Arredata\']': 'parzialmente',
												 '[\'Arredato\']': 'totalmente',
												 '[]': 'no'})
	return data


def create_terrace(data):
	"""Create terrace features."""
	data['Terrazza'] = (data['Altre_caratteristiche']
						.apply(lambda x: str([y for y in x if 'Terrazza' in y or 'Balcone' in y])))
	data.loc[~(data['Terrazza'] == '[]'), 'Terrazza'] = 'sì'
	data.loc[data['Terrazza'] == '[]', 'Terrazza'] = 'no'
	return data


def create_exposure(data):
	"""Create exposure feature."""
	data['Esposizione'] = (data['Altre_caratteristiche']
						   .apply(lambda x: str([y for y in x if 'Esposizione' in y]))
						   .str.extract('(doppia|esterna|interna)', expand=False)
						   .fillna('esterna'))
	return data


def create_other_features(data):
	"""Create columns for each other feature extracted."""
	# Features list
	features_list = ['Fibra ottica', 'Cancello elettrico', 'Cantina', 'Impianto di allarme', 'Mansarda', 'Taverna',
					 'Cablato', 'Idromassaggio', 'Piscina']

	# Create column for each extracted feature
	for feature in features_list:
		mask = data['Altre_caratteristiche'].apply(lambda x: feature in x)
		data[feature] = np.where(mask, 'sì', 'no')
	return data


def create_pipeline(list_functions):
	"""Pipeline function for data cleaning steps."""

	def pipeline(data):
		out = data
		for function in list_functions:
			out = function(out)
		return out

	return pipeline
