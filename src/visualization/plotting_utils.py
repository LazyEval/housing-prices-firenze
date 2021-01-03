import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def histogram(data, columns, log=False):
	"""Create histograms for a list of continuous variables."""
	fig = plt.figure(figsize=(16, 8))
	sns.set_style('whitegrid')

	for i, col in enumerate(columns, 1):
		plt.subplot(1, len(columns), i)
		if log:
			sns.histplot(np.log(data[col]), kde=False, binwidth=0.2)
		else:
			sns.histplot(data[col], kde=False, binwidth=0.2)
		plt.title(col)
	plt.tight_layout()
	return fig


def boxplot(data, columns, log=False):
	"""Create boxplots for a list of continuous variables."""
	fig = plt.figure(figsize=(16, 8))

	for i, col in enumerate(columns, 1):
		plt.subplot(1, len(columns), i)
		if log:
			plt.boxplot(np.log(data[col]), whis=None)
			plt.ylabel("log({})".format(col), size=12)
		else:
			plt.boxplot(data[col], whis=None)
			plt.ylabel("{}".format(col), size=12)
		plt.title(col)
	plt.tight_layout()
	return fig


def create_hue(data):
	gb_ordered = data.groupby('Zona').agg({'Prezzo_per_m2': 'mean'}).sort_values('Prezzo_per_m2', ascending=False)

	mask1 = data['Zona'].isin(gb_ordered[gb_ordered['Prezzo_per_m2'] < 3500].index)
	mask2 = data['Zona'].isin(gb_ordered[gb_ordered['Prezzo_per_m2'] >= 4500].index)
	mask3 = (data['Zona'].isin(gb_ordered[gb_ordered['Prezzo_per_m2'] < 4500].index)) & \
			(data['Zona'].isin(gb_ordered[gb_ordered['Prezzo_per_m2'] >= 3500].index))

	conditions = [mask1, mask2, mask3]
	choices = ['< 3500 EUR/m2', '> 4500 EUR/m2', '[3500, 4500) EUR/m2']
	data['hue'] = np.select(conditions, choices)
	return data


def scatterplot(data, x_col, y_col, hue_data, log=False):
	"""Create scatter plot for two continuous variables."""
	fig = plt.figure(figsize=(16, 16))
	sns.set_style('whitegrid')

	if log:
		sns.scatterplot(x=np.log(data[x_col]), y=np.log(data[y_col]), hue=hue_data)
		plt.xlabel('log({})'.format(x_col), size=14)
		plt.ylabel('log({})'.format(y_col), size=14)
		plt.title('Log price vs. log square meters', size=16)
	else:
		sns.scatterplot(x=data[x_col], y=data[y_col], hue=hue_data)
		plt.xlabel('{}'.format(x_col), size=14)
		plt.ylabel('{}'.format(y_col), size=14)
		plt.title('Price vs. square meters', size=16)
	plt.legend(title='Grouped districts', title_fontsize=14, fontsize=14)
	plt.tight_layout()
	return fig


def hist_per_district(data, col, row, feature, log=False):
	"""Create a facetgrid with histograms."""
	g = sns.FacetGrid(data, col=col, row=row, col_wrap=3, sharex=False, sharey=False, height=5)

	if log:
		g.map_dataframe(sns.histplot, x=np.log(data[feature]), binwidth=0.2)
	else:
		g.map_dataframe(sns.histplot, x=feature, binwidth=0.2)
	g.fig.tight_layout()
	return g


def scatter_per_district(data, col, row, log=False):
	"""Create a facetgrid with scatter plots."""
	g = sns.FacetGrid(data, col=col, row=row, col_wrap=3, sharex=False, sharey=False, height=5)

	if log:
		g.map_dataframe(sns.scatterplot, x=np.log(data['Superficie']), y=np.log(data['Prezzo']))
		g.set_axis_labels("log(price)", "log(square meters)")
	else:
		g.map_dataframe(sns.scatterplot, x='Superficie', y='Prezzo')
		g.set_axis_labels("Square meters", "Price")
	g.fig.tight_layout()
	return g


def ordered_barchart(data):
	"""Create a barchart with ordered values on the y axis."""
	gb_ordered = data.groupby('Zona').agg({'Prezzo_per_m2': 'mean'}).sort_values('Prezzo_per_m2', ascending=False)

	fig = plt.figure(figsize=(12, 10))
	sns.barplot(x=gb_ordered['Prezzo_per_m2'], y=gb_ordered.index, ci=None, color="lightblue")
	plt.xlabel("Average price/m2", size=12)
	plt.ylabel("District", size=12)
	plt.title('Average price/m2 per district', size=14)
	plt.tight_layout()
	return fig


def correlation_plot(data, corr_cols):
	"""Create a correlation plot for selected features of a DataFrame."""
	corr = data[corr_cols].corr()
	mask = np.triu(np.ones_like(corr, dtype=bool))

	fig = plt.figure(figsize=(8, 6))
	cmap = sns.diverging_palette(230, 20, as_cmap=True)
	sns.heatmap(corr, mask=mask, linewidth=.5, cbar_kws={'shrink': .8}, cmap=cmap, annot=True, square=True)
	plt.title('Correlation between numerical features', size=14)
	plt.xticks(rotation=45)
	plt.tight_layout()
	return fig


def plot_predictions(scores, train_values, test_values, train_labels, test_labels=None):
	"""Create scatter plots of the training set and cross-validation values vs. the true values."""
	fig, ax = plt.subplots(figsize=(16, 8))

	ax1 = plt.subplot(1, 2, 1)
	ax1.scatter(train_labels, train_values, edgecolors=(0, 0, 0))
	ax1.plot([train_labels.min(), train_labels.max()], [train_labels.min(), train_labels.max()], 'r--', lw=4)
	extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0)
	ax1.legend([extra], [scores[0]], loc='best', fontsize=14)
	ax1.set_xlabel('Measured')
	ax1.set_ylabel('Predicted')
	ax1.set_title('Training set results', size=14)

	ax2 = plt.subplot(1, 2, 2)
	if test_labels:
		ax2.scatter(test_labels, test_values, edgecolors=(0, 0, 0))
		ax2.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'r--', lw=4)
		ax2.set_title('Test set results', size=14)
	else:
		ax2.scatter(train_labels, test_values, edgecolors=(0, 0, 0))
		ax2.plot([train_labels.min(), train_labels.max()], [train_labels.min(), train_labels.max()], 'r--', lw=4)
		ax2.set_title('Cross-validation results', size=14)
	extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0)
	ax2.legend([extra], [scores[1]], loc='best', fontsize=14)
	ax2.set_xlabel('Measured')
	ax2.set_ylabel('Predicted')

	plt.tight_layout()
	return fig


def plot_model_comparison(results, names):
	"""Create boxplots of the model results."""
	fig = plt.figure(figsize=(10, 10))
	fig.title('Comparison of algorithms', size=14)
	ax = fig.add_subplot(111)
	plt.boxplot(results)
	ax.set_xticklabels(names)
	return fig
