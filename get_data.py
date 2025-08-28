# import time
# from bs4 import BeautifulSoup
import pandas as pd
import requests

all_dfs = []

# Get data from year: 2020-2023
start = 278
end = 996

for page in range(start, end + 1):
    print(f"Fetching page {page}")

    try:
        url = f"https://erp.pgcb.gov.bd/w/generations/view_generations?page={page}"
        res = requests.get(url, verify=False)
        res.raise_for_status()
        html = res.text

        tables = pd.read_html(html)

        page_df = tables[0]

        all_dfs.append(page_df)
    except requests.exceptions.RequestException as e:
        print(e)

full_df = pd.concat(all_dfs, ignore_index=True)
full_df.to_csv("electricity_data.csv", index=False)
