import requests, os
from dotenv import load_dotenv
load_dotenv()
token = os.getenv('AQICN_TOKEN')

stations = [
    '@A471607',
    '@A471608', 
    '@A471609',
    '@A471610',
    '@A471611',
    '@A471612',
    '@A471613',
    '@A471614',
    '@A471615',
]

for s in stations:
    r = requests.get(f'https://api.waqi.info/feed/{s}/?token={token}')
    d = r.json()
    if d['status'] == 'ok' and d['data']['aqi'] != '-':
        name = d['data']['city']['name']
        aqi = d['data']['aqi']
        time = d['data']['time'].get('s', 'unknown')
        print(f"{s}: {name} | AQI={aqi} | Time={time}")
    else:
        print(f"{s}: not found or no data")