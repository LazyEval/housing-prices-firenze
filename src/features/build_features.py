# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from src.features import parse_config, load_raw_data, clean_data, rng


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('config_file', type=str, default='config.yml')
def main(input_filepath, output_filepath, config_file):
    """ Runs data loading and cleaning and pre-processing scripts and
    saves data in ../processed."""
    logger = logging.getLogger(__name__)
    logger.info('Loading collected data, cleaning it, splitting it and'
                'saving it for pre-processing and modeling.')

    # Parse config file
    config = parse_config(config_file)

    # Load data
    df = load_raw_data(input_filepath, config)

    # Clean and save data for EDA
    df_clean = clean_data(config)(df)
    df_clean.to_csv(output_filepath + '/data_clean.csv', index=False)

    # Select features for pre-processing and modeling
    target = config['features']['target']

    X = df_clean.drop(columns=config['features']['drop_cols'])
    y = df_clean[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=rng)

    # Save X and y sets for modeling
    datasets = [X_train, X_test, y_train, y_test]
    filenames = ['X_train', 'X_test', 'y_train', 'y_test']
    for dataset, name in zip(datasets, filenames):
        dataset.to_csv(output_filepath + '/{}.csv'.format(name), index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's
    # found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
