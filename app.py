import streamlit as st
import pandas as pd
import os
from src.data_generator import TennisDataGenerator
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.text_processor import TextProcessor

def initialize_session_state():
    if 'model' not in st.session_state:
        st.session_state.model = ModelTrainer()
        preprocessor = st.session_state.model.load_model()
    
    if 'processor' not in st.session_state:
        st.session_state.processor = DataProcessor()
        st.session_state.processor.preprocessor = preprocessor
    
    if 'text_processor' not in st.session_state:
        st.session_state.text_processor = TextProcessor(st.secrets["OPENAI_API_KEY"])

def main():
    st.title("Tennis Court Price Predictor")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for navigation
    page = st.sidebar.selectbox(
        "Navigate to",
        ["Price Prediction", "View Dataset", "About"]
    )
    
    if page == "Price Prediction":
        show_prediction_page()
    elif page == "View Dataset":
        show_dataset_page()
    else:
        show_about_page()

def show_prediction_page():
    st.header("Predict Tennis Court Booking Price")
    
    # Two tabs for structured and unstructured input
    tab1, tab2 = st.tabs(["Structured Input", "Natural Language Input"])
    
    with tab1:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                date = st.date_input("Booking Date")
                time = st.time_input("Booking Time")
                duration = st.select_slider(
                    "Duration (hours)",
                    options=[1, 1.5, 2, 2.5, 3]
                )
                court_surface = st.selectbox(
                    "Court Surface",
                    ["Hard", "Clay", "Grass", "Carpet"]
                )
                court_type = st.selectbox(
                    "Court Type",
                    ["Indoor", "Outdoor"]
                )
            
            with col2:
                court_quality = st.selectbox(
                    "Court Quality",
                    ["Standard", "Premium", "Elite"]
                )
                match_type = st.selectbox(
                    "Match Type",
                    ["Singles", "Doubles", "Training"]
                )
                num_players = st.selectbox(
                    "Number of Players",
                    [2, 4]
                )
                
                features = st.multiselect(
                    "Additional Features",
                    ["Court Lighting", "Equipment Rental", "Coaching",
                     "Ball Machine", "Refreshments"]
                )
            
            submit_button = st.form_submit_button("Predict Price")
            
            if submit_button:
                # Process inputs and make prediction
                input_data = create_input_dataframe(
                    date, time, duration, court_surface, court_type,
                    court_quality, match_type, num_players, features
                )
                
                show_prediction_results(input_data)
    
    with tab2:
        # Add example prompts
        st.subheader("Example Prompts")
        examples = {
            "Basic Singles": "I want to book a hard court for 2 hours of singles play",
            "Indoor Training": "Need an indoor court for 1.5 hours of training with a coach",
            "Premium Doubles": "Looking for a premium court for doubles match, 2 hours outdoor",
            "Elite Practice": "Book an elite indoor court with ball machine for 2 hours",
            "Clay Court Training": "3 hour training session on clay court with coaching",
            "Quick Practice": "1 hour practice on standard outdoor court",
            "Tournament Prep": "Elite indoor court for 2.5 hours with coaching and ball machine",
            "Casual Game": "Standard grass court for 1.5 hours doubles game"
        }
        
        # Create columns for example buttons
        cols = st.columns(2)
        selected_example = None
        
        # Distribute examples across columns
        for i, (label, text) in enumerate(examples.items()):
            if cols[i % 2].button(label):
                selected_example = text
        
        # Text input area with example if selected
        text_input = st.text_area(
            "Describe your booking requirements",
            value=selected_example if selected_example else "Example: I want to book an indoor hard court for 2 hours of singles training with a coach.",
            height=100
        )
        
        if st.button("Process Text"):
            # Process natural language input
            processed_data = st.session_state.text_processor.process_text(text_input)
            st.json(processed_data)
            
            # Convert the JSON string to dictionary if it's not already
            if isinstance(processed_data, str):
                import json
                features = json.loads(processed_data)
            else:
                features = processed_data
                
            # Create a DataFrame with default values
            input_data = pd.DataFrame({
                'booking_date': [pd.Timestamp.now()],
                'booking_time': [pd.Timestamp.now().strftime('%H:%M')],
                'duration': [features.get('duration', 1)],
                'court_surface': [features.get('court_surface', 'Hard')],
                'court_type': [features.get('court_type', 'Indoor')],
                'court_lighting': [False],
                'num_players': [features.get('num_players', 2)],
                'match_type': [features.get('match_type', 'Singles')],
                'equipment_rental': [False],
                'coaching_requested': [features.get('coaching_requested', False)],
                'ball_machine': [features.get('ball_machine', False)],
                'refreshments': [False],
                'court_quality': [features.get('court_quality', 'Standard')],
                'day_of_week': [pd.Timestamp.now().strftime('%A')],
                'season': [get_season(pd.Timestamp.now())],
                'booking_lead_time': [0],
                'historical_demand': [0.5],
                'temperature': [20],
                'precipitation_chance': [0],
                'special_requests': [False]
            })
            
            # Make prediction
            show_prediction_results(input_data)

def show_dataset_page():
    st.header("Historical Booking Data")
    
    # Load and display dataset
    df = pd.read_csv('data/tennis_bookings.csv')
    st.dataframe(df)
    
    # Display basic statistics
    st.subheader("Dataset Statistics")
    st.write(df.describe())

def show_about_page():
    st.header("About Tennis Court Price Predictor")
    st.write("""
    This application predicts tennis court booking prices based on various factors
    using machine learning. The model is trained on historical booking data and
    takes into account factors such as:
    
    - Court type and surface
    - Booking duration and time
    - Additional services (coaching, equipment rental, etc.)
    - Historical demand and weather conditions
    
    The prediction model uses Random Forest Regression and provides explanations
    for its predictions using SHAP values.
    """)

def create_input_dataframe(date, time, duration, surface, court_type,
                          quality, match_type, num_players, features):
    # Create a single-row dataframe with all features
    data = {
        'booking_date': [date],
        'booking_time': [time.strftime("%H:%M")],
        'duration': [duration],
        'court_surface': [surface],
        'court_type': [court_type],
        'court_lighting': ['Court Lighting' in features],
        'num_players': [num_players],
        'match_type': [match_type],
        'equipment_rental': ['Equipment Rental' in features],
        'coaching_requested': ['Coaching' in features],
        'ball_machine': ['Ball Machine' in features],
        'refreshments': ['Refreshments' in features],
        'court_quality': [quality],
        'day_of_week': [date.strftime("%A")],
        'season': get_season(date),
        'booking_lead_time': [(date - pd.Timestamp.now().date()).days],
        'historical_demand': [0.5],  # Could be calculated based on historical data
        'temperature': [20],  # Could be fetched from weather API
        'precipitation_chance': [0],  # Could be fetched from weather API
        'special_requests': [False]
    }
    
    return pd.DataFrame(data)

def get_season(date):
    month = date.month
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def show_prediction_results(input_data):
    # Preprocess input data
    X = st.session_state.processor.prepare_data(input_data, is_training=False)
    
    # Make prediction
    prediction = st.session_state.model.predict(X)[0]
    
    # Get feature importance
    shap_values = st.session_state.model.explain_prediction(X)
    
    # Display results
    st.subheader("Prediction Results")
    st.metric("Predicted Price", f"${prediction:.2f}")
    
    # Display explanation
    st.subheader("Price Explanation")
    
    # Base price explanation
    st.write("Base price components:")
    
    # Create detailed explanation
    explanation = []
    
    # Court quality impact
    court_quality = input_data['court_quality'].iloc[0]
    if court_quality == 'Premium':
        explanation.append("- Premium court: 50% increase to base price")
    elif court_quality == 'Elite':
        explanation.append("- Elite court: 100% increase to base price")
    
    # Court type impact
    if input_data['court_type'].iloc[0] == 'Indoor':
        explanation.append("- Indoor court: 20% increase to base price")
    
    # Duration impact
    duration = input_data['duration'].iloc[0]
    explanation.append(f"- Duration: {duration} hours (${30 * duration:.2f} base rate)")
    
    # Additional services
    if input_data['coaching_requested'].iloc[0]:
        explanation.append("- Coaching service: +$40.00")
    if input_data['ball_machine'].iloc[0]:
        explanation.append("- Ball machine: +$15.00")
    if input_data['equipment_rental'].iloc[0]:
        explanation.append("- Equipment rental: +$10.00")
    if input_data['court_lighting'].iloc[0]:
        explanation.append("- Court lighting: +$5.00")
    
    # Display all explanation components
    for exp in explanation:
        st.write(exp)
    
    # Additional factors
    st.write("\nOther factors affecting the price:")
    st.write(f"- Day of week: {input_data['day_of_week'].iloc[0]}")
    st.write(f"- Season: {input_data['season'].iloc[0]}")
    st.write(f"- Number of players: {input_data['num_players'].iloc[0]}")
    st.write(f"- Match type: {input_data['match_type'].iloc[0]}")
    
    # Feature importance visualization
    st.subheader("Feature Importance")
    
    # Create a DataFrame for feature importance
    feature_importance = pd.DataFrame({
        'feature': st.session_state.processor.preprocessor.get_feature_names_out(),
        'importance': abs(shap_values[0])
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
    
    # Plot feature importance
    st.bar_chart(feature_importance.set_index('feature'))
    
    st.write("""
    The chart above shows the top 10 most influential features in determining the price.
    Larger bars indicate stronger impact on the final price.
    """)

if __name__ == "__main__":
    main() 