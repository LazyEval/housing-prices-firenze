import streamlit as st
import numpy as np
import pandas as pd
from src.models import Model


def user_input_features():
	square_meters = st.sidebar.number_input('Square meters', 30, 1000, 65)
	district = st.sidebar.selectbox('District', ('Bellosguardo Galluzzo', 'Coverciano Bellariva', 'Firenze Nord',
												 'Campo Di Marte Liberta', 'Legnaia Soffiano', 'Centro', 'Oltrarno',
												 'Zona Firenze Sud', 'Leopoldo Porta Al Prato', 'Serpiolle Careggi',
												 "L'Isolotto", 'Settignano Rovezzano', 'Zona Bolognese Le Cure',
												 'Michelangelo Porta Romana', 'Ugnano Mantignano'))
	num_bathrooms = st.sidebar.slider('Number of bathrooms', 0, 5, 1)
	num_rooms = st.sidebar.slider('Number of rooms', 0, 20, 3)
	house_type = st.sidebar.selectbox('Type of house', ('appartamento', 'terratetto', 'villa unifamiliare',
														'open space', 'loft', 'attico', 'altro'))
	property_class = st.sidebar.selectbox('Property class', ('economica', 'media', 'signorile', 'lusso'))
	property_type = st.sidebar.selectbox('Property type', ('intera proprietà', 'nuda proprietà', 'multiproprietà'))
	income_property = st.sidebar.selectbox('Income property', ('sì', 'no'))
	year_of_construction = st.sidebar.selectbox('Year of construction', (pd.Interval(left=0, right=1850),
																		 pd.Interval(left=1850, right=1950),
																		 pd.Interval(left=1950, right=2000),
																		 pd.Interval(left=2000, right=2021)))
	state = st.sidebar.selectbox('State', ('ottimo/ristrutturato', 'buono/abitabile', 'da ristrutturare',
										   'nuovo/in costruzione'))
	heating_A_C = st.sidebar.selectbox('Heating A/C', ('Autonomo', 'Centralizzato'))
	heating_type = st.sidebar.selectbox('Type of heating', ('radiatori', 'aria', 'stufa', 'pavimento'))
	heating_source = st.sidebar.selectbox('Heating source', ('metano', 'gas', 'pompa di calore', 'elettrica',
															 'fotovoltaico', 'pellet', 'gpl', 'solare'))
	energy_class = st.sidebar.selectbox('Energy class', ('G', 'F', 'E', 'D', 'C', 'B', 'A', 'A1', 'A2', 'A3', 'A4'))
	elevator = st.sidebar.selectbox('Elevator', ('sì', 'no'))
	disabled_access = st.sidebar.selectbox('Disabled access', ('sì', 'no'))
	floor = st.sidebar.selectbox('Floor', ('terra', '1°', '2°', '3°', '4°', '5°', '6°', '7°', '8°', '9°', '10°',
										   'ultimo', 'più livelli', 'rialzato', 'interrato', 'ammezzato'))
	garage_parking = st.sidebar.slider('Garage parking', 0, 10, 0)
	external_parking = st.sidebar.slider('External parking', 0, 10, 0)

	data = {
		'Tipologia': house_type,
		'Superficie_m2': square_meters,
		'Zona': district,
		'Classe_immobile': property_class,
		'Tipo_proprietà': property_type,
		'A_reddito': income_property,
		'Stato': state,
		'Anno_costruzione_bins': year_of_construction,
		'Riscaldamento_A_C': heating_A_C,
		'Tipo_riscaldamento': heating_type,
		'Alimentazione_riscaldamento': heating_source,
		'Classe_energetica': energy_class,
		'Ascensore': elevator,
		'Accesso_disabili': disabled_access,
		'Piano': floor,
		'Posti_garage': garage_parking,
		'Posti_esterni': external_parking,
		'Num_bagni': num_bathrooms,
		'Num_tot_locali': num_rooms
	}
	return pd.DataFrame(data, index=[0])


def predict(model_filepath, config, input_data):
	# Load model
	model = Model.load(model_filepath + config['predicting']['model_name'])

	# Predict
	prediction = int(np.round(model.predict(input_data), -3)[0])
	return prediction
