Housing price estimator: Project overview
--
- Created a tool to estimate the price of a home in Firenze, Italy based on data from October 2020.
- Scraped over 9000 house listings from Immobiliari using Python.
- Built a data cleaning pipeline and extracted features such as district, total number of rooms, heating, and energy class.
- Performed EDA to better understand the data and the influence of each of the features on price.
- Built a pre-processing pipeline including various regression models to select the best model.
- TODO: deploy the model using streamlit.
- TODO: implement testing.
- TODO: containerization with docker.

Code and resources
--
**Python version**: 3.8  
**Packages**: python-dotenv, requests, setuptools, beautifulsoup4, lxml, numpy, pandas, scikit-learn, matplotlib, seaborn, scipy  
**Requirements**: `pip install -r requirements.txt`

Web scraping
--
Scraped 9000 house listings on immobiliari.it and got the following data:
- Price
- Square meters
- District
- Type of property
- Category
- State of the house
- Heating
- Energy class
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
- Extracted price, square meters, state of the house
- Feature engineered price/m<sup>2</sup>, heating, energy class, elevator, disabled access, floor, parking, number of bathrooms, number of rooms, other features (created a new column for each)

Exploratory data analysis
--
I looked at the distributions, boxplots and scatter plots of the data to get a better understanding of what I was working with. Below are some highlights from this analysis:

<img src="https://github.com/LazyEval/housing-prices-firenze/blob/master/reports/figures/barchart.png" width="600">

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
