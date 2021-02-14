from pathlib import Path
from requests import get
from bs4 import BeautifulSoup
import re
import pandas as pd


class WebScraper:
    def __init__(self, raw_dir, interim_dir, website, n_pages):
        self.raw_dir = raw_dir
        self.interim_dir = interim_dir
        self.website = website
        self.n_pages = n_pages

    def _get_urls(self):
        """Get full list of URLs to scrape."""
        urls = []

        print('Getting URLs from website...')
        for i in range(1, self.n_pages + 1):
            if i == 1:
                url = self.website
            else:
                url = self.website + '?pag=' + str(i)

            response = get(url)
            html_soup = BeautifulSoup(response.text, 'html.parser')

            a = html_soup.find_all(href=re.compile("/annunci/"))
            for item in a:
                href = item.get('href')
                urls.append(href)

            print('Loop ' + str(i) + ' completed.')
        return urls

    def save_urls(self, urls):
        """Save URLs to .txt file."""
        with open(self.raw_dir+'urls.txt', 'w') as f:
            for url in urls:
                f.write("%s\n" % url)

    def load_urls(self):
        """Load URLs from .txt file."""
        with open(self.raw_dir+'urls.txt', 'r') as f:
            urls = [line.strip() for line in f]
        return urls

    @staticmethod
    def _get_titles(urls):
        """Get all possible entry titles."""
        table_list = [[], [], []]
        c = 0
        print('Getting all possible titles...')
        for url in urls:
            response = get(url)
            html_soup = BeautifulSoup(response.text, 'lxml')

            tables = html_soup.find_all(class_="im-features__list")
            for i, table in enumerate(tables[:3]):
                titles = table.find_all(class_='im-features__title')
                for title in titles:
                    if title.text not in table_list[i]:
                        table_list[i].append(title.text)
            if c % 500 == 0:
                print('Loop ' + str(c) + ' completed.')
            c += 1
        return table_list

    def save_titles(self, table_list):
        """Write titles to .txt file."""
        with open(self.raw_dir+'titles.txt', 'w') as f:
            for table in table_list:
                for title in table:
                    f.write("%s\n" % title)

    def load_titles(self):
        """Load titles from .txt file."""
        with open(self.raw_dir+'titles.txt', 'r') as f:
            all_titles = [[], [], []]
            for i in range(3):
                for j, line in enumerate(f):
                    if j <= 14:
                        all_titles[0].append(line.strip())
                    elif j <= 27:
                        all_titles[1].append(line.strip())
                    else:
                        all_titles[2].append(line.strip())
        return all_titles

    @staticmethod
    def _get_dicts(table_list):
        """Create dictionaries from all possible titles."""
        caratteristiche = {key: list() for key in table_list[0]}
        caratteristiche['indirizzo'] = []
        caratteristiche['zona'] = []

        costi = {key: list() for key in table_list[1]}

        efficienza_energetica = {key: list() for key in table_list[2]}

        dicts = [caratteristiche, costi, efficienza_energetica]
        return dicts

    def get_data(self):
        """Scrape the data and store it in dictionaries."""
        urls = self._get_urls()
        table_list = self._get_titles(urls)
        dicts = self._get_dicts(table_list)

        print('Getting data from URLs...')
        loop = 0
        for url in urls:
            try:
                response = get(url)
                html_soup = BeautifulSoup(response.text, 'lxml')

                # Get area
                area = (html_soup
                        .find('div', class_="im-relatedLink__container")
                        .find_all('a'))
                dicts[0]['zona'].append(area[-1]['href'][63:])

                # Get address
                addresses = html_soup.find_all(class_="im-location")
                addresses_text = list(set([address.text for address in
                                           addresses]))
                dicts[0]['indirizzo'].append(addresses_text)

                # Get tables
                tables = html_soup.find_all(class_="im-features__list")

                # Loop through the tables
                for i, table in enumerate(tables[:3]):
                    # Get entries (left) and values (right) for first 3 tables
                    titles = table.find_all(class_='im-features__title')
                    values = table.find_all(class_='im-features__value')

                    titles_text = [title.text for title in titles]
                    values_text = [value.text.strip() for value in values]
                    entries = dict(zip(titles_text, values_text))

                    c = 0
                    for key in dicts[i].keys():
                        if key in titles_text:
                            dicts[i][key].append(entries[key])
                            c += 1
                        elif key not in ['indirizzo', 'zona']:
                            dicts[i][key].append('n/a')

                print('Loop ' + str(loop) + ' completed.')
                loop += 1
            except AttributeError:
                print('Loop ' + str(loop) + ' failed.')
                loop += 1

        # Convert dictionaries to dataframes and save them
        dfs = []
        for dict_ in dicts:
            dfs.append(self._to_dataframe(dict_))
        return dfs

    @staticmethod
    def _to_dataframe(dictionary):
        """Transform dictionaries to pandas DataFrames."""
        df = pd.DataFrame.from_dict(dictionary, orient='index').transpose()
        return df

    def save_data(self, data, name):
        """Save DataFrame to csv files."""
        data.to_csv(Path(self.interim_dir+'/{}.csv'.format(name)), index=False)
