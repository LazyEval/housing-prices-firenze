# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.features import parse_config
from src.models import preprocessing_pipeline, Model

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
@click.argument('output_filepath', type=click.Path())
@click.argument('config_file', type=str, default='config.yml')
def main(input_filepath, output_filepath, config_file):
	""" Runs data loading and cleaning and pre-processing scripts and saves data in ../processed."""
	logger = logging.getLogger(__name__)
	logger.info('Loading training data set, setting up pipeline, tuning and training model.')

	# Parse config file
	config = parse_config(config_file)

	# Load data
	X_train = pd.read_csv(input_filepath + '/X_train.csv')
	y_train = np.log1p(pd.read_csv(input_filepath + '/y_train.csv').values)

	# Pre-processing and modeling pipeline
	cat_features = X_train.select_dtypes(include='object').columns
	num_features = X_train.select_dtypes(exclude='object').columns

	pipe = Pipeline([
		('preprocessing', preprocessing_pipeline(cat_features, num_features)),
		('model', SVR())
	])

	# Model
	kf = KFold(config['modeling']['num_folds'], shuffle=True, random_state=42).get_n_splits(X_train.values)
	param_grid = {
		'model__C': 10. ** np.arange(-3, 3),
		'model__gamma': 10. ** np.arange(-3, 3),
	}

	model = Model.tune(pipe, X_train, y_train.ravel(), param_grid, cv=kf)
	print('Mean cross-validation score for {}:'.format(model.name))
	print(np.round(np.sqrt(-model.cv_score(X_train, y_train.ravel(), config['modeling']['scoring'], kf)), 3))

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
