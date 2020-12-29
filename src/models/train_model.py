# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from src.features import parse_config
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

	# Load data
	X_train = pd.read_csv(input_filepath + '/X_train.csv')
	y_train = pd.read_csv(input_filepath + '/y_train.csv').values
	X_test = pd.read_csv(input_filepath + '/X_test.csv')
	y_test = pd.read_csv(input_filepath + '/y_test.csv').values

	# Pre-processing and modeling pipeline
	cat_features = X_train.select_dtypes(include='object').columns
	num_features = X_train.select_dtypes(exclude='object').columns

	pipe = Pipeline([
		('preprocessing', preprocessing_pipeline(cat_features, num_features)),
		('model', RandomForestRegressor())
	])

	# Tune model
	kf = KFold(config['modeling']['num_folds'], shuffle=True,
			   random_state=config['seeding']['seed']).get_n_splits(X_train.values)
	param_grid = {
		'model__n_estimators': [5, 20, 50, 100],
		'model__min_samples_split': [2, 8, 10],
		'model__max_features': [2, 5, 10]
	}

	model = Model.tune(pipe, X_train, y_train.ravel(), param_grid, cv=kf)

	# Train model
	model.train(X_train, y_train.ravel())

	# Evaluate model
	train_preds = model.predict(X_train)
	test_preds = model.predict(X_test)

	print('Training set RSME for {} model: {}'.format(model.name, np.sqrt(mean_squared_error(y_train, train_preds))))
	print('Test set RSME for {} model: {}'.format(model.name, np.sqrt(mean_squared_error(y_test, test_preds))))

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
