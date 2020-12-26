import matplotlib.pyplot as plt
import seaborn as sns


def histogram(data, columns, transformation=None):
	"""Create histograms for a list of continuous variables."""
	fig = plt.figure(figsize=(16, 8))
	sns.set_style('whitegrid')
	for i, col in enumerate(columns, 1):
		plt.subplot(1, 2, i)
		if transformation:
			sns.histplot(transformation(data[col]), kde=False, binwidth=0.2)
		else:
			sns.histplot(data[col], kde=False)
		plt.title(col)
	plt.tight_layout()
	return fig


def boxplot(data, columns, transformation=None):
	"""Create boxplots for a list of continuous variables."""
	fig = plt.figure(figsize=(16, 8))
	for i, col in enumerate(columns, 1):
		plt.subplot(1, 2, i)
		if transformation:
			plt.boxplot(transformation(data[col]), whis=None)
			plt.ylabel("log({})".format(col), size=12)
		else:
			plt.boxplot(data[col], whis=None)
			plt.ylabel("{}".format(col), size=12)
		plt.title(col)
	plt.tight_layout()
	return fig


def scatterplot(data, x_col, y_col, hue_data, transformation=None):
	"""Create scatter plot for two continuous variables."""
	fig = plt.figure(figsize=(16, 16))
	sns.set_style('whitegrid')
	if transformation:
		sns.scatterplot(x=transformation(data[x_col]), y=transformation(data[y_col]), hue=hue_data)
		plt.xlabel('log({})'.format(x_col), size=14)
		plt.ylabel('log({})'.format(y_col), size=14)
		plt.title('Log price vs. log square meters', size=14, weight='bold')
	else:
		sns.scatterplot(x=data[x_col], y=data[y_col], hue=hue_data)
		plt.xlabel('{}'.format(x_col), size=14)
		plt.ylabel('{}'.format(y_col), size=14)
		plt.title('Price vs. square meters', size=14, weight='bold')
	return fig


def hist_per_district(data, col, row, feature, transformation=None):
	"""Create a facetgrid with histograms."""
	g = sns.FacetGrid(data, col=col, row=row, col_wrap=3, sharex=False, sharey=False, height=5)

	if transformation:
		g.map_dataframe(sns.histplot, x=transformation(data[feature]),  binwidth=0.2)
	else:
		g.map_dataframe(sns.histplot, x=feature, binwidth=0.2)
	return g


def scatter_per_district(data, col, row, transformation=None):
	"""Create a facetgrid with scatter plots."""
	g = sns.FacetGrid(data, col=col, row=row, col_wrap=3, sharex=False, sharey=False, height=5)
	if transformation:
		g.map_dataframe(sns.scatterplot, x=transformation(data['Superficie_m2']), y=transformation(data['Prezzo_EUR']))
		g.set_axis_labels("log(price)", "log(square meters)")
	else:
		g.map_dataframe(sns.scatterplot, x='Superficie_m2', y='Prezzo_EUR')
		g.set_axis_labels("Square meters", "Price")
	g.fig.tight_layout()
	return g


def ordered_barchart(data):
	"""Create a barchart with ordered values on the y axis."""
	gb_ordered = data.groupby('Zona').agg({'Prezzo_per_m2': 'mean'}).sort_values('Prezzo_per_m2', ascending=False)

	fig = plt.figure(figsize=(12, 10))
	sns.barplot(x=gb_ordered['Prezzo_per_m2'], y=gb_ordered.index, ci=None)
	plt.xlabel("Average price/m2", size=12)
	plt.ylabel("District", size=12)
	plt.title('Average price/m2 per district', weight='bold')
	plt.tight_layout()
	return fig


def correlation_plot(data, corr_cols):
	"""Create a correlation plot for selected features of a DataFrame."""
	fig = plt.figure(figsize=(8, 6))
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	sns.heatmap(data[corr_cols].corr(), linewidth=.5, cbar_kws={'shrink': .8}, cmap=cmap)
	plt.title('Correlation between numerical features', weight='bold')
	plt.tight_layout()
	return fig


def plot_predictions(train_values, cv_values, true_values):
	"""Create scatter plots of the training set and cross-validation values vs. the true values."""
	fig, ax = plt.subplots(figsize=(16, 8))

	ax1 = plt.subplot(1, 2, 1)
	ax1.scatter(true_values, train_values, edgecolors=(0, 0, 0))
	ax1.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--', lw=4)
	ax1.set_xlabel('Measured')
	ax1.set_ylabel('Predicted')
	ax1.set_title('Training set results')

	ax2 = plt.subplot(1, 2, 2)
	ax2.scatter(true_values, cv_values, edgecolors=(0, 0, 0))
	ax2.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--', lw=4)
	ax2.set_xlabel('Measured')
	ax2.set_ylabel('Predicted')
	ax2.set_title('Cross-validation results')
	return fig


def plot_model_comparison(results, names):
	"""Create boxplots of the model results."""
	fig = plt.figure(figsize=(10, 10))
	fig.title('Algorithm Comparison')
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	return fig
