# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR
from src.features import parse_config, rng
from src.models import preprocessing_pipeline, Model


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('config_file', type=str, default='config.yml')
def main(input_filepath, output_filepath, config_file):
	""" Runs data loading and cleaning and pre-processing scripts and saves data in ../processed."""
	logger = logging.getLogger(__name__)
	logger.info('Loading training data set, setting up pipeline, tuning, training and evaluating final model.')

	# Parse config file
	config = parse_config(config_file)

	# Load training data
	X_train = pd.read_csv(input_filepath + '/X_train.csv')
	y_train = pd.read_csv(input_filepath + '/y_train.csv').values.ravel()

	# Pre-processing and modeling pipeline
	cat_features = X_train.select_dtypes(exclude='float64').columns
	num_features = X_train.select_dtypes(include='float64').columns

	pipe = Pipeline([
		('preprocessing', preprocessing_pipeline(cat_features, num_features)),
		('model', TransformedTargetRegressor(regressor=SVR(), func=np.log1p, inverse_func=np.expm1))
	])

	# Tune or select model
	kf = KFold(config['modeling']['num_folds'], shuffle=True, random_state=rng).get_n_splits(X_train.values)

	model = Model(model=pipe)

	# Train model
	model.train(X_train, y_train)

	# Save model
	model.save(output_filepath + model.name + '.pkl')


if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	# not used in this stub but often useful for finding various files
	project_dir = Path(__file__).resolve().parents[2]

	# find .env automagically by walking up directories until it's found, then
	# load up the .env entries as environment variables
	load_dotenv(find_dotenv())

	main()
