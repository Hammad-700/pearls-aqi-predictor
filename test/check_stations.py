import requests, os
from dotenv import load_dotenv
load_dotenv()
token = os.getenv('AQICN_TOKEN')

stations = [
    'berlin',
    'berlin-wedding',
    'berlin-neukölln', 
    'berlin-tempelhof',
    '@A228',
    '@A229',
    '@A230',
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