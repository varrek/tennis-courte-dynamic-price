import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataProcessor:
    def __init__(self):
        self.preprocessor = None
        
    def create_preprocessor(self):
        # Define numeric and categorical columns
        numeric_features = [
            'duration', 'num_players', 'booking_lead_time',
            'historical_demand', 'temperature', 'precipitation_chance'
        ]
        
        categorical_features = [
            'court_surface', 'court_type', 'match_type',
            'court_quality', 'day_of_week', 'season'
        ]
        
        boolean_features = [
            'court_lighting', 'equipment_rental', 'coaching_requested',
            'ball_machine', 'refreshments', 'special_requests'
        ]
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        # Combine all transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('bool', 'passthrough', boolean_features)
            ])
        
        return self.preprocessor
    
    def process_time_features(self, df):
        # Convert booking_time to hour and minute features
        df['booking_hour'] = pd.to_datetime(df['booking_time'], format='%H:%M').dt.hour
        df['booking_minute'] = pd.to_datetime(df['booking_time'], format='%H:%M').dt.minute
        
        # Drop original time column
        df = df.drop('booking_time', axis=1)
        
        return df
    
    def prepare_data(self, df, is_training=True):
        # Process datetime features
        df = self.process_time_features(df)
        
        if is_training:
            # Create and fit preprocessor
            self.create_preprocessor()
            X = self.preprocessor.fit_transform(df.drop('price', axis=1))
            y = df['price']
            return X, y
        else:
            # Use existing preprocessor
            X = self.preprocessor.transform(df)
            return X 