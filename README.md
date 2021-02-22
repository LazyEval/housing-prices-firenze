Housing price estimator: project overview
--
- Created a tool to estimate the price of a home (RMSE ~ 60K EUR) in Firenze, Italy based on data from October 2020.
- Scraped over 9000 house listings from *immobiliare.it* using Python.
- Built a data cleaning pipeline and extracted features such as district, total number of rooms, heating, and energy class.
- Performed EDA to better understand the data and the influence of each of the features on price.
- Built a pre-processing pipeline including various regression models to select the best model.
- Deployed the model using streamlit: [Housing price estimator](https://share.streamlit.io/lazyeval/housing-prices-firenze/src/deployment/deploy_model.py).
- Used Docker to containerize the project.

Problems encountered
--
The main problems encountered in this project were the following:
- The target variable that was collected from the website represents the list price as opposed to the sale price.
- Modeling results were not very satisfactory, especially at first:
    - it was found that there were a lot of outliers in the data
    - most of these were due to bad data inserted on the website with price/m2 ratios which did not make sense
    - to solve this, an in-depth look at the outliers was made before removing a large part of them
- A lot of data was missing, and some of it important, such as the year of construction.

Code and resources
--
**Python version**: 3.8

**Packages**: python-dotenv, requests, setuptools, beautifulsoup4, lxml, numpy, pandas, scikit-learn, matplotlib, seaborn, scipy, pyyaml, joblib, streamlit

**Running the app using Docker**: clone the repository and type `make run_container` in the terminal from the root directory of the project.

Web scraping
--
Scraped 9000 house listings on *immobiliare.it* and got the following data:
- Price
- Square meters
- District
- Year of construction
- Type of property
- Category
- Condition
- Heating
- Air condition
- Energy efficiency
- Number of rooms
- Floors
- Other features

Data cleaning
--
After scraping the data from the web, I created a data cleaning pipeline where I did the following:
- Dropped uninformative columns such as the listings ID or the type of contract
- Dropped rows corresponding to homes that are not located in Firenze
- Manually imputed missing districts
- Drop rows with missing values in price and square meters
- Extracted price, square meters, type of property, condition of the house
- Feature engineered price/m<sup>2</sup>, heating, air conditioning, energy efficiency, elevator, disabled access, floor, parking, number of bathrooms, number of rooms, other features (created a new column for each)
- Removed outliers in the price, square meters and price/m<sup>2</sup> features.

Exploratory data analysis
--
I looked at the distributions, boxplots and scatter plots of the data to get a better understanding of what I was working with. Below are some highlights from this analysis:

<img src="https://github.com/LazyEval/housing-prices-firenze/blob/master/reports/figures/barchart.png" width="800">
<img src="https://github.com/LazyEval/housing-prices-firenze/blob/master/reports/figures/scatter.png" width="800">
<img src="https://github.com/LazyEval/housing-prices-firenze/blob/master/reports/figures/corr_plot.png" width="600">

For more details on the exploratory data analysis, please refer to my [EDA notebook](https://github.com/LazyEval/housing-prices-firenze/blob/master/notebooks/2.0-LazyEval-EDA.ipynb).

Model building
--
I created a few pre-processing pipelines for various models, including different transformations or imputations on the data. The data was split into training and test (20 %) sets so as to avoid data leakage, and cross-validation on the pipelines was used to evaluate the model performance on unseen data and for tuning. Model selection was based on the performance during cross-validation and the test set was only used to evaluate the performance of the final model.

The chosen metric is the Root Mean Squared Error (RMSE) so as to penalize larger errors in prediction.

The following models were tried:
- **Ridge regression** — baseline model.
- **Support vector regressor** — the relationship between the predictors and the target variable does not seem to be strictly linear, so I chose a non-linear SVR which yielded better results.
- **Ensemble models** — I tried a series of more complex models to see how they would fit the data and generalize. Most of these models would tend to overfit the data.

Ridge regression and the random forest regressor illustrate feature importance:

<img src="https://github.com/LazyEval/housing-prices-firenze/blob/master/reports/figures/linear_coefs.png" width="400"> | <img src="https://github.com/LazyEval/housing-prices-firenze/blob/master/reports/figures/feature_importance.png" width="400">

Model performance
--
The figure below compares the performance of the various algorithms **using cross-validation**:

<img src="https://github.com/LazyEval/housing-prices-firenze/blob/master/reports/figures/algo_comparison.png" width="700">

The SVR, random forest regressor and extra tree regressor models performed best and I therefore chose to select the SVR model. The figure below plots the predictions of the final estimator both on the training and on the test set:
<img src="https://github.com/LazyEval/housing-prices-firenze/blob/master/reports/figures/pred_plots.png" width="1200">

The RMSE of the final model used on the test set is 60'000 EUR.

For more details on my model selection process be sure to take a look at my [modeling notebook](https://github.com/LazyEval/housing-prices-firenze/blob/master/notebooks/3.0-LazyEval-modeling.ipynb).

Model deployment
--
The model was deployed using the streamlit library. The script that is run by streamlit was integrated into the project workflow so that predictions are made instantly. The API endpoint takes in a request with a set of values (that correspond to features of the house) inserted by the user and returns an estimated price.

Improvements
--
- Sale prices should be collected as opposed to list prices.
- A comparison could be made between this estimator and the one on *immobiliare.it* by making predictions on a number of houses.
- Data validation and testing should be implemented.

Project organization
--
The project organization, **based on drivendata's data science cookiecutter template**, is as follows:

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
