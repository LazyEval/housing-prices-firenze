# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import streamlit as st
from src.features import parse_config
from src.deployment import user_input_features, predict


def main(model_filepath='models/', config_file='config.yml'):
	""" Loads user input and uses stored model to predict price of the house."""
	logger = logging.getLogger(__name__)
	logger.info('Loading user input and predicting.')

	# Parse config file
	config = parse_config(config_file)

	# Front-end page setup
	st.set_page_config(page_title='Housing price estimator for Florence, Italy.', layout='centered')

	# Load user input
	X = user_input_features()

	# Write to front-end
	st.subheader('Created by Matteo Latinov')

	st.markdown('This estimator was created with the purpose of obtaining an approximate list price for a house in'
				' Florence, Italy based on a set of features or attributes of the home. The model was built with data'
				' collected from *immobiliari.it* in October 2020.  \n  \n Feel free to try it out and let me know what'
				' you think!')

	if st.button('Estimate the value of your house'):
		prediction = predict(model_filepath, config, X)
		st.markdown('Estimated price for this house:')
		st.success('{:,.0f} EUR'.format(prediction))


if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	# not used in this stub but often useful for finding various files
	project_dir = Path(__file__).resolve().parents[2]

	main()
