-- Run this in the Supabase SQL Editor before running the feature pipeline.
ALTER TABLE aqi_bronze_raw
    ADD COLUMN IF NOT EXISTS weather jsonb;

ALTER TABLE aqi_silver_cleaned
    ADD COLUMN IF NOT EXISTS wind_speed double precision,
    ADD COLUMN IF NOT EXISTS wind_direction double precision,
    ADD COLUMN IF NOT EXISTS precipitation double precision,
    ADD COLUMN IF NOT EXISTS pressure double precision,
    ADD COLUMN IF NOT EXISTS pm25_raw double precision,
    ADD COLUMN IF NOT EXISTS pm10_raw double precision,
    ADD COLUMN IF NOT EXISTS no2_raw double precision,
    ADD COLUMN IF NOT EXISTS o3_raw double precision;

ALTER TABLE aqi_gold_features
    ADD COLUMN IF NOT EXISTS wind_speed double precision,
    ADD COLUMN IF NOT EXISTS wind_direction double precision,
    ADD COLUMN IF NOT EXISTS precipitation double precision,
    ADD COLUMN IF NOT EXISTS pressure double precision,
    ADD COLUMN IF NOT EXISTS pm25_raw double precision,
    ADD COLUMN IF NOT EXISTS pm10_raw double precision,
    ADD COLUMN IF NOT EXISTS no2_raw double precision,
    ADD COLUMN IF NOT EXISTS o3_raw double precision;
