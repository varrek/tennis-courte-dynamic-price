import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import os

class TennisDataGenerator:
    def __init__(self, num_records=600):
        self.fake = Faker()
        self.num_records = num_records
        
    def generate_data(self):
        data = []
        
        # Define possible values for categorical features
        surface_types = ['Hard', 'Clay', 'Grass', 'Carpet']
        court_types = ['Indoor', 'Outdoor']
        match_types = ['Singles', 'Doubles', 'Training']
        court_quality = ['Standard', 'Premium', 'Elite']
        
        start_date = datetime.now() - timedelta(days=365)
        
        for _ in range(self.num_records):
            booking_date = self.fake.date_time_between(start_date=start_date)
            
            # Generate feature values
            record = {
                'booking_date': booking_date,
                'booking_time': booking_date.strftime('%H:%M'),
                'duration': np.random.choice([1, 1.5, 2, 2.5, 3]),
                'court_surface': np.random.choice(surface_types),
                'court_type': np.random.choice(court_types),
                'court_lighting': bool(np.random.choice([0, 1])),
                'num_players': np.random.choice([2, 4]),
                'match_type': np.random.choice(match_types),
                'equipment_rental': bool(np.random.choice([0, 1])),
                'coaching_requested': bool(np.random.choice([0, 1])),
                'ball_machine': bool(np.random.choice([0, 1])),
                'refreshments': bool(np.random.choice([0, 1])),
                'court_quality': np.random.choice(court_quality),
                'day_of_week': booking_date.strftime('%A'),
                'season': self._get_season(booking_date),
                'booking_lead_time': np.random.randint(0, 30),
                'historical_demand': np.random.uniform(0, 1),
                'temperature': np.random.normal(20, 5),
                'precipitation_chance': np.random.uniform(0, 1),
                'special_requests': bool(np.random.choice([0, 1]))
            }
            
            # Calculate price based on features
            price = self._calculate_price(record)
            record['price'] = price
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def _get_season(self, date):
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    def _calculate_price(self, record):
        # Base price
        base_price = 30
        
        # Adjustments based on features
        if record['court_quality'] == 'Premium':
            base_price *= 1.5
        elif record['court_quality'] == 'Elite':
            base_price *= 2
            
        if record['court_type'] == 'Indoor':
            base_price *= 1.2
            
        if record['coaching_requested']:
            base_price += 40
            
        if record['ball_machine']:
            base_price += 15
            
        if record['equipment_rental']:
            base_price += 10
            
        # Add some random variation
        price = base_price * np.random.uniform(0.9, 1.1)
        
        return round(price, 2)

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    generator = TennisDataGenerator(600)
    df = generator.generate_data()
    df.to_csv('data/tennis_bookings.csv', index=False)
    print("Dataset generated successfully!") 