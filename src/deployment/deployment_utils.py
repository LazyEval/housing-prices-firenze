import streamlit as st
import numpy as np
import pandas as pd
from src.models import Model


def user_input_features():
	square_meters = st.sidebar.slider('Square meters', 30, 1000)
	district = st.sidebar.selectbox('District', ('Bellosguardo Galluzzo', 'Coverciano Bellariva', 'Firenze Nord',
												 'Campo Di Marte Liberta', 'Legnaia Soffiano', 'Centro', 'Oltrarno',
												 'Zona Firenze Sud', 'Leopoldo Porta Al Prato', 'Serpiolle Careggi',
												 "L'Isolotto", 'Settignano Rovezzano', 'Zona Bolognese Le Cure',
												 'Michelangelo Porta Romana', 'Ugnano Mantignano'))
	num_bathrooms = st.sidebar.slider('Number of bathrooms', 0, 10, 1)
	num_rooms = st.sidebar.slider('Number of rooms', 0, 20, 3)
	type_of_house = st.sidebar.selectbox('Type of house', ('Appartamento', 'Terratetto unifamiliare',
														   'Villa unifamiliare', 'Terratetto plurifamiliare', 'Rustico',
														   'Villa a schiera', 'Attico', 'Open space',
														   'Villa bifamiliare', 'Loft', 'Appartamento in villa',
														   'Casa colonica', 'Mansarda', 'Casale',
														   'Villa plurifamiliare', 'Bed & Breakfast', 'Ufficio'))
	type_of_property = st.sidebar.selectbox('Type of property', ('Intera proprietà, classe immobile media',
																 'Intera proprietà, classe immobile signorile',
																 'Intera proprietà, classe immobile economica',
																 'Intera proprietà',
																 'Nuda proprietà, classe immobile media',
																 'Intera proprietà, immobile di lusso',
																 'Classe immobile economica', 'Classe immobile media',
																 'Classe immobile signorile',
																 'Nuda proprietà, classe immobile economica',
																 'Immobile di lusso',
																 'Nuda proprietà, classe immobile signorile',
																 'Nuda proprietà', 'Nuda proprietà, immobile di lusso'))
	year_of_construction = st.sidebar.slider('Year of construction', 1300, 2020)
	state = st.sidebar.selectbox('State', ('Ottimo_/_Ristrutturato', 'Buono_/_Abitabile', 'Da_ristrutturare',
										   'Nuovo_/_In_costruzione'))
	heating_A_C = st.sidebar.selectbox('Heating A/C', ('Autonomo', 'Centralizzato'))
	heating_type = st.sidebar.selectbox('Type of heating', ('radiatori', 'aria', 'stufa', 'pavimento'))
	heating_source = st.sidebar.selectbox('Heating source', ('metano', 'gas', 'pompa di calore', 'elettrica',
															 'fotovoltaico', 'pellet', 'gpl', 'solare'))
	energy_class = st.sidebar.selectbox('Energy class', ('F', 'G', 'E', 'D', 'C', 'B', 'A', 'A1', 'A2', 'A3', 'A4'))
	elevator = st.sidebar.selectbox('Elevator', ('sì', 'no'))
	disabled_access = st.sidebar.selectbox('Disabled access', ('sì', 'no'))
	floor = st.sidebar.selectbox('Floor', ('terra', '1°', '2°', '3°', '4°', '5°', '6°', '7°', '8°', '9°', '10°',
										   'ultimo', 'più livelli', 'rialzato', 'interrato', 'ammezzato'))
	garage_parking = st.sidebar.slider('Garage parking', 0, 10, 0)
	external_parking = st.sidebar.slider('External parking', 0, 10, 0)

	data = {
		'Tipologia': type_of_house,
		'Tipo_proprietà': type_of_property,
		'Zona': district,
		'Anno_costruzione': year_of_construction,
		'Stato': state,
		'Superficie_m2': square_meters,
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
	prediction = int(np.round(np.expm1(model.predict(input_data)), -3)[0])
	return prediction
