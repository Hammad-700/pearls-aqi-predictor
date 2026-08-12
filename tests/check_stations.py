import requests, os
from dotenv import load_dotenv
load_dotenv()
token = os.getenv('AQICN_TOKEN')
stations = [
    'lahore', 
    'lahore/gulberg',
    'lahore/township',
    'lahore/pac-valves',
    '@A471607',
    '@11411',
    '@11412',
    '@11413',
]
for s in stations:
    r = requests.get(f'https://api.waqi.info/feed/{s}/?token={token}')
    d = r.json()
    if d['status'] == 'ok':
        name = d['data']['city']['name']
        aqi = d['data']['aqi']
        time = d['data']['time']['s']
        print(f"{s}: {name} | AQI={aqi} | Time={time}")
    else:
        print(f"{s}: not found")