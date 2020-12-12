# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from src.features import parse_config, load_data, save_data, clean_data


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('config_file', type=str, default='config.yml')
def main(input_filepath, output_filepath, config_file):
	""" Runs data loading and cleaning and pre-processing scripts and saves data in ../processed."""
	logger = logging.getLogger(__name__)
	logger.info('Loading collected data, cleaning it, pre-processing it and saving it.')

	# Parse config file
	config = parse_config(config_file)

	# Load data
	df = load_data(input_filepath, config)

	# Clean and save data for EDA
	df_clean = clean_data(config)(df)
	df_clean.to_csv(output_filepath + '/data_clean.csv', index=False)

	# Split data
	train, test = train_test_split(df_clean, test_size=0.2, random_state=0)

	# Pre-process datasets
	train_processed = preprocessing_pipeline.fit_transform(train)
	test_processed = preprocessing_pipeline.transform(test)

	# Save datasets for modeling
	save_data(train_processed, output_filepath, '/train')
	save_data(test_processed, output_filepath, '/test')


if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	# not used in this stub but often useful for finding various files
	project_dir = Path(__file__).resolve().parents[2]

	# find .env automagically by walking up directories until it's found, then
	# load up the .env entries as environment variables
	load_dotenv(find_dotenv())

	main()
