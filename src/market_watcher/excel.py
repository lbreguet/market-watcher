# src/market_watcher/excel.py
from os import listdir, mkdir, path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import LineChart, Reference

from .client import get_markets

def build_df(market):
    df = pd.DataFrame(market['contracts'])
    df['timeStamp'] = market['timeStamp']
    df['shortName'] = df['shortName'].astype(str)
    return df.drop(columns=['id', 'name', 'status', 'dateEnd', 'image', 'displayOrder'])


def to_xl():
    if not path.exists("markets"):
        mkdir("markets")

    for market in get_markets():
        market['shortName'] = (market['shortName']
                            .replace("?", "_").replace("/", "-")
                            .replace("'", "").replace("\"", ""))
        filepath = 'markets/' + market['shortName'] + '.xlsx'
        df = build_df(market)

        if path.exists(filepath):
            df = pd.concat([df, pd.read_excel(filepath, dtype={'shortNames': str})])
            df = df.sort_values(by=['shortName', 'timeStamp'])

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)


def create_charts():
    to_xl()

    for excel in listdir('markets'):
        if excel.startswith('~$') or not excel.endswith('.xlsx'):
            continue

        filepath = 'markets/' + excel
        df = pd.read_excel(filepath, dtype={'shortName': str})
        workbook = load_workbook(filepath)
        worksheet = workbook.worksheets[0]

        counts = df.groupby('shortName', sort=True).size()

        start_row = 2  # first data row (row 1 is the header)
        for idx, (name, count) in enumerate(counts.items()):
            end_row = start_row + count - 1

            chart = LineChart()
            chart.title = str(name)
            chart.y_axis.title = 'price'
            chart.x_axis.title = 'time stamp'

            data = Reference(worksheet, min_col=2, max_col=7,
                            min_row=start_row, max_row=end_row)
            chart.add_data(data)

            worksheet.add_chart(chart, f'L{2 + idx * 20}')
            start_row = end_row + 1

        workbook.save(filepath)