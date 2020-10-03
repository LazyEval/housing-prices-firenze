from bs4 import BeautifulSoup
from requests import get
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import regex as re
import pickle
sns.set()

# Initial URL and number of pages to scrape
path_raw = '/home/matteo@COPPET/Documents/data_science/projects/housing_prices_firenze/data/raw/'
immobiliare = 'https://www.immobiliare.it/vendita-case/firenze/'
n_pages = 365
average_price = 3.457 # Average price per square meter in Sept 2020

# Get full list of URLs to scrape
with open(path_raw+'urls.txt', 'w') as f:
    urls = []
    for i in range(1, n_pages + 1):
        if i == 1:
            url = immobiliare
        else:
            url = immobiliare + '?pag=' + str(i)

        response = get(url)
        html_soup = BeautifulSoup(response.text, 'html.parser')
        a = html_soup.find_all(href=re.compile("/annunci/"))

        for item in a:
            href = item.get('href')
            urls.append(href)
            f.write("%s\n" % href)
        print('Loop ' + str(i) + ' completed.')

with open(path_raw+'urls.txt', 'r') as f:
    urls = [line.strip() for line in f]


# Get all possible entry titles
all_titles = [[], [], []]
c = 0
for url in urls:
    response = get(url)
    html_soup = BeautifulSoup(response.text, 'lxml')
    tables = html_soup.find_all(class_="im-features__list")

    for i, table in enumerate(tables[:3]):
        titles = table.find_all(class_='im-features__title')
        for title in titles:
            if title.text not in all_titles[i]:
                all_titles[i].append(title.text)
    if c % 500 == 0:
        print('Loop ' + str(c) + ' completed.')
    c += 1

# Write titles to file
with open(path_raw+'titles.txt', 'w') as f:
    for tables in all_titles:
        for title in tables:
            f.write("%s\n" % title)

# Open file into list of lists
with open(path_raw+'titles.txt', 'r') as f:
    all_titles = [[], [], []]
    for i in range(3):
        for j, line in enumerate(f):
            if j <= 14:
                all_titles[0].append(line.strip())
            elif j <= 27:
                all_titles[1].append(line.strip())
            else:
                all_titles[2].append(line.strip())

caratteristiche = {key: list() for key in all_titles[0]}
caratteristiche['indirizzo'] = []
caratteristiche['zona'] = []

costi = {key: list() for key in all_titles[1]}
efficienza_energetica = {key: list() for key in all_titles[2]}

dicts = [caratteristiche, costi, efficienza_energetica]

loop = 0
for url in urls:
    try:
        response = get(url)
        html_soup = BeautifulSoup(response.text, 'lxml')

        # Get area
        area = html_soup.find('div', class_="im-relatedLink__container").find_all('a')
        caratteristiche['zona'].append(area[-1]['href'][63:])

        # Get address
        addresses = html_soup.find_all(class_="im-location")
        addresses_text = list(set([address.text for address in addresses]))
        caratteristiche['indirizzo'].append(addresses_text)

        # Get tables
        tables = html_soup.find_all(class_="im-features__list")

        for i, table in enumerate(tables[:3]):

            # Get entries (left) and values (right) for first 3 tables
            titles = table.find_all(class_='im-features__title')
            values = table.find_all(class_='im-features__value')
            titles_text = [title.text for title in titles]
            c = 0
            for key in dicts[i].keys():
                if key in titles_text:
                    dicts[i][key].append(values[c].text.strip())
                    c += 1
                elif key not in ['indirizzo', 'zona']:
                    dicts[i][key].append('n/a')
        print('Loop ' + str(loop) + ' completed.')
        loop += 1
    except:
        print('Loop ' + str(loop) + ' failed.')
        loop += 1


caratteristiche = pd.DataFrame.from_dict(dicts[0], orient='index').transpose()
costi = pd.DataFrame.from_dict(dicts[1], orient='index').transpose()
efficienza_energetica = pd.DataFrame.from_dict(dicts[2], orient='index').transpose()

caratteristiche.to_excel(path_raw+'caratteristiche.xlsx', index=False)
costi.to_excel(path_raw+'costi.xlsx', index=False)
efficienza_energetica.to_excel(path_raw+'efficienza_energetica.xlsx', index=False)
