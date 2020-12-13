# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.features import parse_config, clean_data
from src.models import preprocessing_pipeline, Model
from src.deployment import user_input_features

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('config_file', type=str, default='config.yml')
def main(input_filepath, model_filepath, config_file):
	""" Runs data loading and cleaning and pre-processing scripts and saves data in ../processed."""
	logger = logging.getLogger(__name__)
	logger.info('Loading training data set, setting up pipeline, tuning and training model.')

	# Parse config file
	config = parse_config(config_file)

	# Load data
	X = user_input_features()
	#X_test = pd.read_csv(input_filepath + '/X_test.csv')
	print(X)

	# Load model
	model = Model.load(model_filepath + config['predicting']['model_name'])

	# Make prediction
	print(np.round(np.expm1(model.predict(X)), -3))


if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	# not used in this stub but often useful for finding various files
	project_dir = Path(__file__).resolve().parents[2]

	# find .env automagically by walking up directories until it's found, then
	# load up the .env entries as environment variables
	load_dotenv(find_dotenv())

	main()
