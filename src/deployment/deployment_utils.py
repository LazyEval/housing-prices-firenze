import streamlit as st
import numpy as np
import pandas as pd
from src.models import Model


def user_input_features():
	"""Define and user inputs as a DataFrame."""
	# Header
	st.sidebar.header('Input your data here.')

	# Square meters
	square_meters = st.sidebar.number_input('Square meters', 30, 1000, 65)

	# District
	values = ['Bellosguardo Galluzzo', 'Campo Di Marte Liberta', 'Centro', 'Coverciano Bellariva', 'Firenze Nord',
			  'L\'Isolotto', 'Legnaia Soffiano', 'Leopoldo Porta Al Prato', 'Michelangelo Porta Romana', 'Oltrarno',
			  'Serpiolle Careggi', 'Settignano Rovezzano', 'Ugnano Mantignano', 'Zona Bolognese Le Cure',
			  'Zona Firenze Sud']
	default_ix = values.index('Centro')
	district = st.sidebar.selectbox('District', values, index=default_ix)

	# Number of bathrooms
	num_bathrooms = st.sidebar.slider('Number of bathrooms', 1, 5, 1)

	# Number of rooms
	num_rooms = st.sidebar.slider('Number of rooms', 1, 20, 3)

	# Type of house
	values = ['appartamento', 'terratetto', 'villa unifamiliare', 'open space', 'loft', 'attico', 'altro']
	default_ix = values.index('appartamento')
	house_type = st.sidebar.selectbox('Type of house', values, index=default_ix)

	# Property class
	values = ['economica', 'media', 'signorile', 'lusso']
	default_ix = values.index('media')
	property_class = st.sidebar.selectbox('Property class', values, index=default_ix)

	# Type of property
	values = ['intera proprietà', 'nuda proprietà', 'a reddito']
	default_ix = values.index('intera proprietà')
	property_type = st.sidebar.selectbox('Property type', values, index=default_ix)

	# Year of construction
	values = ['(0.0, 1850.0]', '(1850.0, 1950.0]', '(1950.0, 2000.0]', '(2000.0, 2021.0]']
	default_ix = values.index('(1950.0, 2000.0]')
	year_of_construction = st.sidebar.selectbox('Year of construction', values, index=default_ix)

	# Condition
	values = ['ottimo/ristrutturato', 'buono/abitabile', 'da ristrutturare', 'nuovo/in costruzione']
	default_ix = values.index('ottimo/ristrutturato')
	condition = st.sidebar.selectbox('State', values, index=default_ix)

	# Heating A/C
	values = ['autonomo', 'centralizzato']
	default_ix = values.index('centralizzato')
	heating_A_C = st.sidebar.selectbox('Heating A/C', values, index=default_ix)

	# Type of heating
	values = ['radiatori', 'aria', 'stufa', 'pavimento']
	default_ix = values.index('radiatori')
	heating_type = st.sidebar.selectbox('Type of heating', values, index=default_ix)

	# Heating source
	values = ['metano', 'gas', 'pompa di calore', 'elettrica', 'fotovoltaico', 'pellet', 'gpl', 'solare']
	default_ix = values.index('metano')
	heating_source = st.sidebar.selectbox('Heating source', values, index=default_ix)

	# Air conditioning
	values = ['non presente', 'autonomo', 'centralizzato', 'predisposizione']
	default_ix = values.index('non presente')
	air_conditioning = st.sidebar.selectbox('Air conditioning', values, index=default_ix)

	# Energy efficiency
	values = ['bassa (E, F, G)', 'media (B, C, D)', 'alta (A, A+, A1-A4)']
	default_ix = values.index('bassa (E, F, G)')
	energy_efficiency = st.sidebar.selectbox('Energy efficiency', values, index=default_ix)

	# Elevator
	values = ['sì', 'no']
	default_ix = values.index('no')
	elevator = st.sidebar.selectbox('Elevator', values, index=default_ix)

	# Disabled access
	values = ['sì', 'no']
	default_ix = values.index('no')
	disabled_access = st.sidebar.selectbox('Disabled access', values, index=default_ix)

	# Floor
	values = ['terra', 'intermedio', 'ultimo', 'interrato']
	default_ix = values.index('intermedio')
	floor = st.sidebar.selectbox('Floor', values, index=default_ix)

	# Garage parking
	garage_parking = st.sidebar.slider('Garage parking', 0, 10, 0)

	# External parking
	external_parking = st.sidebar.slider('External parking', 0, 10, 0)

	# Windows
	values = ['singolo', 'doppio', 'triplo']
	default_ix = values.index('singolo')
	windows = st.sidebar.selectbox('Window type', values, index=default_ix)

	# Garden
	values = ['non presente', 'privato', 'comune']
	default_ix = values.index('non presente')
	garden = st.sidebar.selectbox('Garden', values, index=default_ix)

	# Furnished
	values = ['no', 'parzialmente', 'totalmente']
	default_ix = values.index('no')
	furnished = st.sidebar.selectbox('Furnished', values, index=default_ix)

	# Terrace
	values = ['sì', 'no']
	default_ix = values.index('no')
	terrace = st.sidebar.selectbox('Terrace', values, index=default_ix)

	# Exposition
	values = ['esterna', 'interna', 'doppia']
	default_ix = values.index('esterna')
	exposition = st.sidebar.selectbox('Exposition', values, index=default_ix)

	# Fiber optic
	values = ['sì', 'no']
	default_ix = values.index('no')
	fiber_optic = st.sidebar.selectbox('Fiber optic', values, index=default_ix)

	# Electrical gate
	values = ['sì', 'no']
	default_ix = values.index('no')
	gate = st.sidebar.selectbox('Electrical gate', values, index=default_ix)

	# Cellar
	values = ['sì', 'no']
	default_ix = values.index('no')
	cellar = st.sidebar.selectbox('Cellar', values, index=default_ix)

	# Alarm system
	values = ['sì', 'no']
	default_ix = values.index('no')
	alarm = st.sidebar.selectbox('Alarm system', values, index=default_ix)

	# Attic
	values = ['sì', 'no']
	default_ix = values.index('no')
	attic = st.sidebar.selectbox('Attic', values, index=default_ix)

	# Tavern
	values = ['sì', 'no']
	default_ix = values.index('no')
	tavern = st.sidebar.selectbox('Tavern', values, index=default_ix)

	# Cabled
	values = ['sì', 'no']
	default_ix = values.index('no')
	cabled = st.sidebar.selectbox('Cabled', values, index=default_ix)

	# Hydromassage
	values = ['sì', 'no']
	default_ix = values.index('no')
	hydromassage = st.sidebar.selectbox('Hydromassage', values, index=default_ix)

	# Pool
	values = ['sì', 'no']
	default_ix = values.index('no')
	pool = st.sidebar.selectbox('Pool', values, index=default_ix)

	# Dictionary with user inputs
	data = {
		'Superficie': square_meters,
		'Piano': floor,
		'Zona': district,
		'Stato': condition,
		'Climatizzazione': air_conditioning,
		'Classe_immobile': property_class,
		'Tipo_proprietà': property_type,
		'Tipologia_casa': house_type,
		'Anno_costruzione_bins': year_of_construction,
		'Riscaldamento_A_C': heating_A_C,
		'Tipo_riscaldamento': heating_type,
		'Alimentazione_riscaldamento': heating_source,
		'Efficienza_energetica': energy_efficiency,
		'Ascensore': elevator,
		'Accesso_disabili': disabled_access,
		'Posti_garage': garage_parking,
		'Posti_esterni': external_parking,
		'Num_bagni': num_bathrooms,
		'Num_tot_locali': num_rooms,
		'Infissi': windows,
		'Giardino': garden,
		'Arredato': furnished,
		'Terrazza': terrace,
		'Esposizione': exposition,
		'Fibra ottica': fiber_optic,
		'Cancello elettrico': gate,
		'Cantina': cellar,
		'Impianto di allarme': alarm,
		'Mansarda': attic,
		'Taverna': tavern,
		'Cablato': cabled,
		'Idromassaggio': hydromassage,
		'Piscina': pool
	}
	return pd.DataFrame(data, index=[0])


def predict(model_filepath, config, input_data):
	"""Return prediction from user input."""
	# Load model
	model = Model.load(model_filepath + config['predicting']['model_name'])

	# Predict
	prediction = int(np.round(model.predict(input_data), -3)[0])
	return prediction
