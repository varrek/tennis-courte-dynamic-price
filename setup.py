import os
from src.data_generator import TennisDataGenerator
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer

def setup():
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Generate dataset if it doesn't exist
    if not os.path.exists('data/tennis_bookings.csv'):
        generator = TennisDataGenerator(600)
        df = generator.generate_data()
        df.to_csv('data/tennis_bookings.csv', index=False)
        print("Dataset generated successfully!")
    
    # Train model if it doesn't exist
    if not os.path.exists('models/trained_model.joblib'):
        from train_model import train_initial_model
        train_initial_model()
        print("Model trained successfully!")

if __name__ == "__main__":
    setup() 