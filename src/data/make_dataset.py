# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.data import WebScraper


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data collecting scripts to collect data and save it in
    ../interim."""
    logger = logging.getLogger(__name__)
    logger.info('Collecting and saving data.')

    # Define scraping variables
    website = 'https://www.immobiliare.it/vendita-case/firenze/'
    n_pages = 366

    # Collect data
    scraper = WebScraper(raw_dir=input_filepath,
                         interim_dir=output_filepath,
                         website=website,
                         n_pages=n_pages)
    dataframes = scraper.get_data()

    # Save data
    filenames = ['caratteristiche', 'costi', 'efficienza_energetica']
    for df, name in zip(dataframes, filenames):
        scraper.save_data(df, name)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's
    # found, then load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
