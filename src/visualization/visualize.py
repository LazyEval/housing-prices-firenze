# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from src.visualization import (histogram, boxplot, scatterplot, hist_per_district, scatter_per_district,
							   ordered_barchart)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
	""" Loads cleaned data and creates visualizations that are then stored in reports/figures."""
	logger = logging.getLogger(__name__)
	logger.info('Loading cleaned data and creating visualizations.')

	# Load data
	df = pd.read_csv(input_filepath+'/data_clean.csv')

	# Variables
	continuous_vars = ['Prezzo_EUR', 'Superficie_m2']

	# Histograms
	histograms = histogram(df, continuous_vars, transformation=np.log)
	histograms.savefig(output_filepath+'/histograms.png')

	# Boxplots
	boxplots = boxplot(df, continuous_vars, transformation=np.log)
	boxplots.savefig(output_filepath + '/boxplots.png')

	# Scatterplot
	scatter = scatterplot(df, 'Superficie_m2', 'Prezzo_EUR', hue_data=df['Zona'], transformation=np.log)
	scatter.savefig(output_filepath + '/scatter.png')

	# Facetgrid histograms
	facetgrid_histograms = hist_per_district(df, 'Zona', None, 'Prezzo_EUR', transformation=np.log)
	facetgrid_histograms.savefig(output_filepath + '/facetgrid_histograms.png')

	# Facetgrid scatterplots
	facetgrid_scatters = scatter_per_district(df, 'Zona', None, transformation=None)
	facetgrid_scatters.savefig(output_filepath + '/facetgrid_scatters.png')

	# Ordered barchart
	barchart = ordered_barchart(df)
	barchart.savefig(output_filepath + '/barchart.png')


if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	# not used in this stub but often useful for finding various files
	project_dir = Path(__file__).resolve().parents[2]

	# find .env automagically by walking up directories until it's found, then
	# load up the .env entries as environment variables
	load_dotenv(find_dotenv())

	main()
