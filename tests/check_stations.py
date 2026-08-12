import requests, os
from dotenv import load_dotenv
load_dotenv()
token = os.getenv('AQICN_TOKEN')
stations = ['lahore', 'lahore-lda', 'lahore-pk', '@7536', '@7537']
for s in stations:
    r = requests.get(f'https://api.waqi.info/feed/{s}/?token={token}')
    d = r.json()
    if d['status'] == 'ok':
        print(f"{s}: AQI={d['data']['aqi']} Time={d['data']['time']['s']}")
    else:
        print(f"{s}: not found")