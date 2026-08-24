import pandas as pd

from features.engineer_features import build_gold_features, clean_to_silver
from features.fetch_aqi import station_matches_lahore_pakistan


def test_clean_to_silver_preserves_weather_values():
    bronze_row = {
        "city": "lahore",
        "timestamp": "2026-08-24T09:00:00Z",
        "raw_data": {
            "aqi": 104,
            "iaqi": {
                "pm25": {"v": 104},
                "t": {"v": 35.2},
                "h": {"v": 52},
            },
        },
    }

    silver = clean_to_silver(bronze_row)

    assert silver["temperature"] == 35.2
    assert silver["humidity"] == 52.0
    assert silver["aqi"] == 104


def test_station_validation_rejects_non_lahore_coordinates():
    valid = {"data": {"city": {"geo": [31.5481, 74.3440], "location": "Lahore, Pakistan"}}}
    invalid = {"data": {"city": {"geo": [44.0582, -121.3153], "location": "Bend, Oregon, USA"}}}

    assert station_matches_lahore_pakistan(valid)
    assert not station_matches_lahore_pakistan(invalid)


def test_gold_features_use_latest_weather_and_lag():
    rows = [
        {
            "city": "lahore",
            "timestamp": "2026-08-24T08:00:00Z",
            "aqi": 100,
            "temperature": 34.0,
            "humidity": 55.0,
        },
        {
            "city": "lahore",
            "timestamp": "2026-08-24T09:00:00Z",
            "aqi": 104,
            "temperature": 35.2,
            "humidity": 52.0,
        },
    ]

    gold = build_gold_features(rows)

    assert gold["aqi_lag_1h"] == 100
    assert gold["temperature"] == 35.2
    assert gold["humidity"] == 52.0
    assert pd.Timestamp(gold["timestamp"]).tzinfo is not None
