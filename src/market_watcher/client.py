# src/market_watcher/client.py
import json
import requests


def get_markets():
    r = requests.get('https://www.predictit.org/api/marketdata/all')
    return json.loads(r.content.decode())['markets']