import pandas as pd
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer

def train_initial_model():
    # Load data
    df = pd.read_csv('data/tennis_bookings.csv')
    
    # Initialize processor and prepare data
    processor = DataProcessor()
    X, y = processor.prepare_data(df, is_training=True)
    
    # Train and save model
    trainer = ModelTrainer()
    trainer.train(X, y, processor.preprocessor)
    trainer.save_model()
    
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_initial_model() 