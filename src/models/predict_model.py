# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from src.features import parse_config
from src.visualization import plot_predictions
from src.models import Model
from sklearn.metrics import mean_squared_error


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
@click.argument('config_file', type=str, default='config.yml')
def main(input_filepath, model_filepath, output_filepath, config_file):
    """Runs data loading and cleaning and pre-processing scripts and
    saves data in ../processed."""
    logger = logging.getLogger(__name__)
    logger.info('Loading training set, test set and model and predicting.')

    # Parse config file
    config = parse_config(config_file)

    # Load data
    X_train = pd.read_csv(input_filepath + '/X_train.csv')
    y_train = pd.read_csv(input_filepath + '/y_train.csv').values.ravel()

    X_test = pd.read_csv(input_filepath + '/X_test.csv')
    y_test = pd.read_csv(input_filepath + '/y_test.csv').values.ravel()

    # Load model
    model = Model.load(model_filepath + config['predicting']['model_name'])

    # Make predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Evaluate model
    train_score = np.sqrt(mean_squared_error(y_train, train_pred))
    test_score = np.sqrt(mean_squared_error(y_test, test_pred))

    # Plot predictions
    scores = (
        (r'$RMSE={:,.0f}$' + ' EUR').format(train_score),
        (r'$RMSE={:,.0f}$' + ' EUR').format(test_score),
    )
    pred_plots = plot_predictions(scores, train_pred, test_pred, y_train,
                                  y_test)
    pred_plots.savefig(output_filepath + '/pred_plots.png')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's
    # found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
