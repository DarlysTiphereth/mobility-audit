import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Generate GPS tracking data
num_buses = 500
bus_ids = [f'Bus_{i+1}' for i in range(num_buses)]

records = []
dates = pd.date_range(start='2024-03-18', end='2024-03-22', freq='H')

for bus_id in bus_ids:
    for date in dates:
        records.append({
            'bus_id': bus_id,
            'timestamp': date,
            'latitude': random.uniform(-9.6000, -9.6000 + 0.05), # Simulated lat
            'longitude': random.uniform(-35.7000, -35.7000 + 0.05), # Simulated long
        })

gps_df = pd.DataFrame(records)
gps_df.to_csv('gps_frota.csv', index=False)


# Generate demographic data
neighborhoods = ['Centro', 'Benedito Bentes', 'Jatiúca', 'Ponta Verde', 'Tabuleiro do Martins', 'Cruz das Almas',
                 'Gustavo Paiva', 'São Jorge', 'Mangabeiras', 'Gruta de Lourdes', 'Farol', 'Água Branca',
                 'Jacintinho', 'Canaã', 'Santo Eduardo']

censo_records = []
for neighborhood in neighborhoods:
    censo_records.append({
        'neighborhood': neighborhood,
        'population': random.randint(3000, 20000),
        'average_income': round(random.uniform(500, 2500), 2),
        'housing_units': random.randint(1000, 5000)
    })

censo_df = pd.DataFrame(censo_records)
censo_df.to_csv('censo_social.csv', index=False)


# Generate street segments data
malha_records = []
for i in range(150):
    malha_records.append({
        'segment_id': f'Segment_{i+1}',
        'length_km': round(random.uniform(0.5, 5.0), 2),
        'neighborhood': random.choice(neighborhoods)
    })

malha_df = pd.DataFrame(malha_records)
malha_df.to_csv('malha_viaria.csv', index=False)


# Generate cellular demand sensors data
sensor_records = []
for bus_id in bus_ids[:100]:  # Only 100 sensors
    for date in dates:
        sensor_records.append({
            'sensor_id': bus_id,
            'timestamp': date,
            'demand': random.randint(0, 100)
        })

sensor_df = pd.DataFrame(sensor_records)
sensor_df.to_csv('sensores_celulares.csv', index=False)


# Generate bus stop infrastructure data
pontos_records = []
for i in range(200):
    pontos_records.append({
        'stop_id': f'Stop_{i+1}',
        'neighborhood': random.choice(neighborhoods),
        'latitude': random.uniform(-9.6000, -9.6000 + 0.05),
        'longitude': random.uniform(-35.7000, -35.7000 + 0.05)
    })

pontos_df = pd.DataFrame(pontos_records)
pontos_df.to_csv('pontos_onibus.csv', index=False)