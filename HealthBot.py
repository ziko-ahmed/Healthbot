import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta
import re
import requests
from transformers import pipeline


# Set page configuration
st.set_page_config(
    page_title="Healthcare Planner & Wellness Advisor",
    page_icon="🩺",
    layout="wide"
)

# Define navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Health Metrics", "Dashboard", "Fitness Tracker", "Nutrition Planner", "Medication Tracker", "Health Advisor Chat"])

# Units selection
if 'units' not in st.session_state:
    st.session_state.units = 'Metric'

# Unit conversion functions
def convert_weight(weight, to_unit='kg'):
    if to_unit == 'kg':
        return weight
    elif to_unit == 'lbs':
        return weight * 2.20462
    return weight

def convert_height(height, to_unit='cm'):
    if to_unit == 'cm':
        return height
    elif to_unit == 'in':
        return height * 0.393701
    return height

# Format metrics for display
def format_weight(weight, unit='kg'):
    if unit == 'kg':
        return f"{weight:.1f} kg"
    elif unit == 'lbs':
        return f"{weight:.1f} lbs"
    return f"{weight:.1f}"

def format_height(height, unit='cm'):
    if unit == 'cm':
        return f"{height:.1f} cm"
    elif unit == 'in':
        return f"{height:.1f} in"
    return f"{height:.1f}"

# Units selector in sidebar
st.sidebar.title("Units Settings")
units_option = st.sidebar.selectbox(
    "Select Units",
    ("Metric", "Imperial"),
    index=0 if st.session_state.units == 'Metric' else 1
)

if units_option != st.session_state.units:
    st.session_state.units = units_option
    st.experimental_rerun()

# Global variables to store user data
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'age': 0,
        'gender': '',
        'height': 0.0,
        'weight': 0.0,
        'weight_history': {},
        'activity_level': '',
        'health_goals': [],
        'chronic_conditions': [],
        'daily_steps': 0,
        'sleep_hours': 0.0,
        'water_intake': 0.0,
        'exercise_minutes': {},
        'nutrition': {
            'calories': 0,
            'protein': 0,
            'carbs': 0,
            'fats': 0
        }
    }

# Chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize the LLM for health advice
@st.cache_resource
def load_llm_model():
    try:
        # Use a smaller model that can run locally
        return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    except Exception as e:
        st.error(f"Error loading LLM model: {e}")
        return None

# Months list for reference
months = [
    "January", "February", "March", "April", "May", "June", "July", "August", 
    "September", "October", "November", "December"
]

# Days of the week
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

#####################################
# Health Metrics Functions
#####################################

def calculate_bmi(weight, height):
    """Calculate BMI based on weight (kg) and height (cm)."""
    if height == 0:
        return 0  # Return 0 instead of causing division by zero
    
    height_m = height / 100  # Convert cm to meters
    bmi = weight / (height_m * height_m)
    return round(bmi, 1)

def interpret_bmi(bmi):
    """Interpret BMI value."""
    if bmi < 18.5:
        return "Underweight", "orange"
    elif bmi < 25:
        return "Normal weight", "green"
    elif bmi < 30:
        return "Overweight", "orange"
    else:
        return "Obese", "red"

def calculate_bmr(weight, height, age, gender):
    """Calculate Basal Metabolic Rate."""
    if gender.lower() == 'male':
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
    
    return round(bmr)

def calculate_tdee(bmr, activity_level):
    """Calculate Total Daily Energy Expenditure."""
    activity_multipliers = {
        'sedentary': 1.2,
        'lightly_active': 1.375,
        'moderately_active': 1.55,
        'very_active': 1.725,
        'extremely_active': 1.9
    }
    
    multiplier = activity_multipliers.get(activity_level.lower().replace(' ', '_'), 1.2)
    tdee = bmr * multiplier
    
    return round(tdee)

def calculate_water_intake(weight, activity_level):
    """Calculate recommended daily water intake in liters."""
    # Base recommendation: 30ml per kg of body weight
    base_intake = weight * 0.03
    
    # Adjust based on activity level
    activity_multipliers = {
        'sedentary': 1.0,
        'lightly_active': 1.1,
        'moderately_active': 1.2,
        'very_active': 1.3,
        'extremely_active': 1.4
    }
    
    multiplier = activity_multipliers.get(activity_level.lower().replace(' ', '_'), 1.0)
    water_intake = base_intake * multiplier
    
    return round(water_intake, 1)

def calculate_macro_nutrients(tdee, goal):
    """Calculate recommended macronutrients based on TDEE and goal."""
    macros = {}
    
    if goal == 'lose_weight':
        calorie_target = tdee * 0.85  # 15% deficit
        macros['protein'] = round((calorie_target * 0.40) / 4)  # 40% protein
        macros['carbs'] = round((calorie_target * 0.30) / 4)    # 30% carbs
        macros['fats'] = round((calorie_target * 0.30) / 9)     # 30% fats
    elif goal == 'gain_muscle':
        calorie_target = tdee * 1.1  # 10% surplus
        macros['protein'] = round((calorie_target * 0.35) / 4)  # 35% protein
        macros['carbs'] = round((calorie_target * 0.40) / 4)    # 40% carbs
        macros['fats'] = round((calorie_target * 0.25) / 9)     # 25% fats
    else:  # maintain
        calorie_target = tdee
        macros['protein'] = round((calorie_target * 0.30) / 4)  # 30% protein
        macros['carbs'] = round((calorie_target * 0.40) / 4)    # 40% carbs
        macros['fats'] = round((calorie_target * 0.30) / 9)     # 30% fats
    
    macros['calories'] = round(calorie_target)
    
    return macros

def plot_weight_trend():
    """Generate a weight trend graph."""
    weight_history = st.session_state.user_data['weight_history']
    
    if not weight_history:
        return None
    
    # Sort by date
    dates = sorted(weight_history.keys())
    weights = [weight_history[date] for date in dates]
    
    # Convert to display units if necessary
    if st.session_state.units == 'Imperial':
        weights = [convert_weight(w, 'lbs') for w in weights]
    
    fig = go.Figure()
    
    # Weight line
    fig.add_trace(go.Scatter(
        x=dates, y=weights, mode='lines+markers',
        name='Weight',
        line=dict(color='blue', width=2)
    ))
    
    # Calculate trend line
    if len(dates) > 1:
        x_numeric = np.arange(len(dates))
        z = np.polyfit(x_numeric, weights, 1)
        p = np.poly1d(z)
        trend = p(x_numeric)
        
        fig.add_trace(go.Scatter(
            x=dates, y=trend,
            mode='lines', name='Trend',
            line=dict(color='red', width=1, dash='dash')
        ))
    
    weight_unit = "lbs" if st.session_state.units == 'Imperial' else "kg"
    
    fig.update_layout(
        title="Weight Trend Over Time",
        xaxis_title="Date",
        yaxis_title=f"Weight ({weight_unit})",
        hovermode="x unified"
    )
    
    return fig

def plot_exercise_distribution():
    """Generate a pie chart for exercise distribution."""
    exercise_minutes = st.session_state.user_data['exercise_minutes']
    
    if not exercise_minutes:
        return None
    
    # Prepare data
    labels = list(exercise_minutes.keys())
    values = list(exercise_minutes.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.3,
        textinfo="label+percent", hoverinfo="label+value"
    )])
    
    fig.update_layout(title_text="Exercise Type Distribution (minutes)")
    return fig

def plot_nutrition_breakdown():
    """Generate a bar chart for nutrition breakdown."""
    nutrition = st.session_state.user_data['nutrition']
    
    if not nutrition or all(v == 0 for v in nutrition.values()):
        return None
    
    # Prepare data
    labels = ['Calories', 'Protein (g)', 'Carbs (g)', 'Fats (g)']
    values = [nutrition['calories'], nutrition['protein'], nutrition['carbs'], nutrition['fats']]
    
    # Calculate recommended values
    if st.session_state.user_data['weight'] > 0 and st.session_state.user_data['height'] > 0:
        age = st.session_state.user_data['age']
        gender = st.session_state.user_data['gender']
        weight = st.session_state.user_data['weight']
        height = st.session_state.user_data['height']
        activity = st.session_state.user_data['activity_level']
        
        # Default to 'maintain' if no goal is set
        goal = 'maintain'
        if 'Lose weight' in st.session_state.user_data['health_goals']:
            goal = 'lose_weight'
        elif 'Gain muscle' in st.session_state.user_data['health_goals']:
            goal = 'gain_muscle'
        
        # Calculate recommended macros
        bmr = calculate_bmr(weight, height, age, gender)
        tdee = calculate_tdee(bmr, activity)
        recommended = calculate_macro_nutrients(tdee, goal)
        
        rec_values = [recommended['calories'], recommended['protein'], recommended['carbs'], recommended['fats']]
    else:
        rec_values = [0, 0, 0, 0]
    
    fig = go.Figure()
    
    # Actual intake
    fig.add_trace(go.Bar(
        x=labels, y=values, name='Actual Intake',
        marker_color='blue'
    ))
    
    # Recommended intake
    fig.add_trace(go.Bar(
        x=labels, y=rec_values, name='Recommended',
        marker_color='green'
    ))
    
    fig.update_layout(
        title="Nutrition Breakdown",
        barmode='group',
        xaxis_title="Nutrient",
        yaxis_title="Amount"
    )
    
    return fig

def estimate_body_fat_percentage(gender, age, bmi):
    """Estimate body fat percentage based on gender, age and BMI."""
    # This is a simplified estimate using the Deurenberg formula
    if gender.lower() == 'male':
        body_fat = (1.20 * bmi) + (0.23 * age) - 16.2
    else:  # female
        body_fat = (1.20 * bmi) + (0.23 * age) - 5.4
    
    return max(0, min(body_fat, 60))  # Clamp between 0% and 60%

def interpret_body_fat(gender, body_fat):
    """Interpret body fat percentage based on gender."""
    if gender.lower() == 'male':
        if body_fat < 6:
            return "Essential fat", "orange"
        elif body_fat < 14:
            return "Athletic", "green"
        elif body_fat < 18:
            return "Fitness", "green"
        elif body_fat < 25:
            return "Average", "blue"
        else:
            return "Obese", "red"
    else:  # female
        if body_fat < 14:
            return "Essential fat", "orange"
        elif body_fat < 21:
            return "Athletic", "green"
        elif body_fat < 25:
            return "Fitness", "green"
        elif body_fat < 32:
            return "Average", "blue"
        else:
            return "Obese", "red"

def calculate_ideal_weight(height, gender, frame_size='medium'):
    """Calculate ideal weight range based on height, gender and frame size."""
    # Height in cm, returns weight in kg
    height_m = height / 100
    
    # Base calculations using the Hamwi formula
    if gender.lower() == 'male':
        base_weight = 48 + 2.7 * (height - 152.4) / 2.54
    else:  # female
        base_weight = 45.5 + 2.2 * (height - 152.4) / 2.54
    
    # Adjust for frame size
    if frame_size == 'small':
        min_weight = base_weight * 0.9
        max_weight = base_weight * 0.95
    elif frame_size == 'large':
        min_weight = base_weight * 1.05
        max_weight = base_weight * 1.1
    else:  # medium
        min_weight = base_weight * 0.95
        max_weight = base_weight * 1.05
    
    return round(min_weight, 1), round(max_weight, 1)

def generate_health_report(user_data):
    """Generate a comprehensive health report with visualizations and recommendations."""
    # Extract user data
    age = user_data['age']
    gender = user_data['gender']
    height = user_data['height']
    weight = user_data['weight']
    activity_level = user_data['activity_level']
    health_goals = user_data['health_goals']
    
    # Calculate metrics
    bmi = calculate_bmi(weight, height)
    bmi_category, _ = interpret_bmi(bmi)
    bmr = calculate_bmr(weight, height, age, gender)
    tdee = calculate_tdee(bmr, activity_level)
    body_fat = estimate_body_fat_percentage(gender, age, bmi)
    body_fat_category, _ = interpret_body_fat(gender, body_fat)
    
    # Generate report
    report = f"""
    # Health Assessment Report
    
    ## Basic Information
    - Age: {age} years
    - Gender: {gender}
    - Height: {height:.1f} cm
    - Weight: {weight:.1f} kg
    - Activity Level: {activity_level}
    
    ## Key Health Metrics
    - BMI: {bmi:.1f} ({bmi_category})
    - Estimated Body Fat: {body_fat:.1f}% ({body_fat_category})
    - Basal Metabolic Rate: {bmr} calories/day
    - Total Daily Energy Expenditure: {tdee} calories/day
    
    ## Health Goals
    {', '.join(health_goals) if health_goals else 'No specific goals set'}
    
    ## Recommendations
    """
    
    # Add recommendations based on metrics
    if bmi_category == "Overweight" or bmi_category == "Obese":
        report += """
    ### Weight Management
    - Consider a moderate calorie deficit of 300-500 calories per day
    - Focus on increasing protein intake to preserve muscle mass
    - Aim for 150-300 minutes of moderate-intensity exercise per week
    - Incorporate strength training 2-3 times per week
    """
    
    if body_fat_category == "Obese":
        report += """
    ### Body Composition Improvement
    - Prioritize resistance training to build lean muscle mass
    - Consider high-intensity interval training (HIIT) for efficient fat burning
    - Monitor protein intake to support muscle maintenance
    - Focus on whole foods and limit processed foods
    """
    
    if "Improve cardiovascular health" in health_goals:
        report += """
    ### Cardiovascular Health
    - Aim for 150 minutes of moderate-intensity aerobic activity per week
    - Include activities like brisk walking, swimming, or cycling
    - Monitor heart rate during exercise (target: 50-70% of max heart rate for moderate intensity)
    - Consider adding 2-3 sessions of HIIT per week for improved cardiovascular benefits
    """
    
    if "Better sleep" in health_goals:
        report += """
    ### Sleep Optimization
    - Maintain a consistent sleep schedule (even on weekends)
    - Create a restful environment (dark, quiet, cool)
    - Avoid screens 1 hour before bedtime
    - Limit caffeine after noon and alcohol before bed
    - Consider relaxation techniques like meditation before sleep
    """
    
    # Add general recommendations
    report += """
    ### General Wellness
    - Stay hydrated (aim for 8-10 cups of water per day)
    - Consume a balanced diet rich in fruits, vegetables, lean proteins, and whole grains
    - Practice stress management techniques
    - Get regular health check-ups and screenings
    """
    
    return report

#####################################
# Medication Tracking Functions
#####################################
def calculate_next_dose(last_dose_time, frequency_hours):
    """Calculate the next dose time based on last dose and frequency."""
    next_dose = last_dose_time + timedelta(hours=frequency_hours)
    return next_dose

def check_medication_interactions(medications_list):
    """Check for potential interactions between medications."""
    # This is a simplified example - in a real application, this would 
    # connect to a medication database API
    common_interactions = {
        ('Aspirin', 'Warfarin'): 'Increased bleeding risk',
        ('Ibuprofen', 'Aspirin'): 'Increased risk of GI bleeding',
        ('Lisinopril', 'Potassium supplements'): 'Risk of hyperkalemia',
        ('Simvastatin', 'Grapefruit juice'): 'Increased statin concentration',
        ('Warfarin', 'Vitamin K'): 'Decreased anticoagulant effect'
    }
    
    interactions = []
    for i, med1 in enumerate(medications_list):
        for med2 in medications_list[i+1:]:
            if (med1, med2) in common_interactions:
                interactions.append(f"{med1} + {med2}: {common_interactions[(med1, med2)]}")
            elif (med2, med1) in common_interactions:
                interactions.append(f"{med2} + {med1}: {common_interactions[(med2, med1)]}")
    
    return interactions

#####################################
# Health Advisor Chat with Open Source LLM
#####################################

def get_health_advice_llm(query):
    """Generate health advice using open source LLM."""
    try:
        # Load the model
        model = load_llm_model()
        if model is None:
            return "I'm sorry, the language model could not be loaded. Please try again later."
        
        # Add context about the user's health situation if available
        context = ""
        if st.session_state.user_data['age'] > 0:
            age = st.session_state.user_data['age']
            gender = st.session_state.user_data['gender']
            weight = st.session_state.user_data['weight']
            height = st.session_state.user_data['height']
            
            if st.session_state.units == 'Imperial':
                # Convert to Imperial for display
                weight_lbs = convert_weight(weight, 'lbs')
                height_in = convert_height(height, 'in')
                context = f"With age: {age}, gender: {gender}, weight: {weight_lbs:.1f} lbs, height: {height_in:.1f} in: "
            else:
                context = f"With age: {age}, gender: {gender}, weight: {weight:.1f} kg, height: {height:.1f} cm: "

        # Create a prompt with health advisor context
        prompt = f"""
        <|system|>
        You are a knowledgeable healthcare assistant. Provide clear, accurate, and helpful advice on health and wellness topics. 
        Keep responses concise but informative. Don't diagnose medical conditions or prescribe treatments. Always recommend consulting a doctor for specific medical issues.
        </|system|>
        
        <|user|>
        {context}{query}
        </|user|>
        
        <|assistant|>
        """
        
        # Generate response
        response = model(prompt, max_length=400, temperature=0.7, num_return_sequences=1)
        
        # Extract the response text
        generated_text = response[0]['generated_text']
        
        # Extract only the assistant's response
        assistant_response = generated_text.split("<|assistant|>")[-1].strip()
        
        # Clean up the response if needed
        if "<|" in assistant_response:
            assistant_response = assistant_response.split("<|")[0].strip()
            
        return assistant_response
        
    except Exception as e:
        st.error(f"Error with LLM service: {e}")
        return "I'm sorry, I couldn't process your request. Please try again or ask a different question."
    
def simple_symptom_checker(symptoms):
    """Provide basic information about common symptoms."""
    # This is a simplified example - in a real application, you would use a medical API
    common_symptoms = {
        "headache": {
            "possible_causes": ["Stress", "Dehydration", "Eye strain", "Migraine", "Sinus infection"],
            "self_care": ["Rest in a quiet, dark room", "Stay hydrated", "Apply a cold or warm compress", "Over-the-counter pain relievers if appropriate"],
            "when_to_see_doctor": ["Severe or sudden headache", "Headache with fever, stiff neck, confusion", "Headache after head injury", "Headache that worsens despite treatment"]
        },
        "fever": {
            "possible_causes": ["Infection (viral or bacterial)", "Inflammatory conditions", "Medication reactions"],
            "self_care": ["Rest", "Stay hydrated", "Dress in light clothing", "Over-the-counter fever reducers if appropriate"],
            "when_to_see_doctor": ["Temperature above 103°F (39.4°C)", "Fever lasting more than 3 days", "Fever with severe headache or rash", "Difficulty breathing"]
        },
        "cough": {
            "possible_causes": ["Common cold", "Allergies", "Asthma", "Reflux", "Respiratory infection"],
            "self_care": ["Stay hydrated", "Use a humidifier", "Honey for soothing (if not for young children)", "Avoid irritants"],
            "when_to_see_doctor": ["Cough lasting more than 3 weeks", "Coughing up blood", "Shortness of breath", "Fever above 100.4°F (38°C)"]
        },
        "fatigue": {
            "possible_causes": ["Lack of sleep", "Poor diet", "Stress", "Anemia", "Depression", "Underlying medical condition"],
            "self_care": ["Improve sleep habits", "Regular physical activity", "Balanced diet", "Stress management"],
            "when_to_see_doctor": ["Extreme fatigue that doesn't improve with rest", "Fatigue with unexplained weight loss", "Fatigue with other symptoms like dizziness or shortness of breath"]
        },
        "nausea": {
            "possible_causes": ["Food poisoning", "Motion sickness", "Medication side effects", "Viral infections", "Pregnancy"],
            "self_care": ["Small, frequent meals", "Clear liquids", "Avoid strong odors", "Ginger or peppermint tea"],
            "when_to_see_doctor": ["Severe or persistent nausea", "Signs of dehydration", "Nausea with severe abdominal pain", "Nausea after head injury"]
        }
    }
    
    # Look for symptoms in the input
    found_symptoms = []
    for symptom in common_symptoms:
        if symptom in symptoms.lower():
            found_symptoms.append(symptom)
    
    if not found_symptoms:
        return "I couldn't identify specific symptoms. Please be more specific about what you're experiencing."
    
    # Create response
    response = "Here's some information about your symptoms:\n\n"
    
    for symptom in found_symptoms:
        info = common_symptoms[symptom]
        response += f"### {symptom.capitalize()}\n\n"
        response += f"**Possible causes:** {', '.join(info['possible_causes'])}\n\n"
        response += f"**Self-care measures:**\n"
        for measure in info['self_care']:
            response += f"- {measure}\n"
        response += f"\n**When to see a doctor:**\n"
        for warning in info['when_to_see_doctor']:
            response += f"- {warning}\n"
        response += "\n"
    
    response += "\n**Disclaimer:** This information is for educational purposes only and is not a substitute for professional medical advice."
    
    return response

#####################################
# Fitness Tracking Functions
#####################################

def calculate_steps_target(age, activity_level):
    """Calculate recommended daily steps based on age and activity level."""
    base_steps = 10000  # Default recommendation
    
    # Adjust based on age
    if age < 18:
        base_steps = 12000
    elif age > 65:
        base_steps = 8000
    
    # Adjust based on activity level
    activity_multipliers = {
        'sedentary': 0.6,
        'lightly_active': 0.8,
        'moderately_active': 1.0,
        'very_active': 1.2,
        'extremely_active': 1.4
    }
    
    multiplier = activity_multipliers.get(activity_level.lower().replace(' ', '_'), 1.0)
    target_steps = base_steps * multiplier
    
    return round(target_steps)

def calculate_exercise_minutes_target(age, health_goals):
    """Calculate recommended weekly exercise minutes based on age and goals."""
    # Base recommendation: 150 minutes of moderate activity per week
    base_minutes = 150
    
    # Adjust based on age
    if age < 18:
        base_minutes = 180
    elif age > 65:
        base_minutes = 120
    
    # Adjust based on goals
    if 'Lose weight' in health_goals:
        base_minutes *= 1.3
    elif 'Gain muscle' in health_goals:
        base_minutes *= 1.2
    elif 'Improve cardiovascular health' in health_goals:
        base_minutes *= 1.4
    
    return round(base_minutes)

def calculate_sleep_target(age):
    """Calculate recommended sleep hours based on age."""
    if age < 6:
        return "10-14 hours"
    elif age < 13:
        return "9-11 hours"
    elif age < 18:
        return "8-10 hours"
    elif age < 65:
        return "7-9 hours"
    else:
        return "7-8 hours"

def plot_daily_activity():
    """Generate a bar chart for daily activity breakdown."""
    # Sample data structure for daily activity
    daily_steps = st.session_state.user_data['daily_steps']
    sleep_hours = st.session_state.user_data['sleep_hours']
    
    if daily_steps == 0 and sleep_hours == 0:
        return None
    
    # Create a stacked bar chart for daily activity
    fig = go.Figure()
    
    # Sleep hours
    fig.add_trace(go.Bar(
        x=["Daily Activity"], y=[sleep_hours],
        name='Sleep (hours)',
        marker_color='purple'
    ))
    
    # Estimate active hours based on steps (rough approximation)
    active_hours = daily_steps / 1000  # ~1000 steps per active hour
    active_hours = min(active_hours, 16)  # Cap at 16 hours
    
    fig.add_trace(go.Bar(
        x=["Daily Activity"], y=[active_hours],
        name='Active (hours)',
        marker_color='green'
    ))
    
    # Estimate sedentary hours (24 - sleep - active)
    sedentary_hours = 24 - sleep_hours - active_hours
    
    fig.add_trace(go.Bar(
        x=["Daily Activity"], y=[sedentary_hours],
        name='Sedentary (hours)',
        marker_color='orange'
    ))
    
    fig.update_layout(
        title="Daily Activity Breakdown",
        barmode='stack',
        xaxis_title="",
        yaxis_title="Hours",
        yaxis=dict(range=[0, 24])
    )
    
    return fig

#####################################
# Page Content
#####################################

# Dashboard Page
if page == "Dashboard":
    st.title("📊 Health Dashboard")
    
    # Check if we have basic user data
    if st.session_state.user_data['age'] == 0 or not st.session_state.user_data['gender']:
        st.warning("Please complete your profile in the Health Metrics section to view your dashboard.")
        if st.button("Go to Health Metrics"):
            st.experimental_rerun()
    else:
        # Get user data
        age = st.session_state.user_data['age']
        gender = st.session_state.user_data['gender']
        height = st.session_state.user_data['height']
        weight = st.session_state.user_data['weight']
        activity_level = st.session_state.user_data['activity_level']
        daily_steps = st.session_state.user_data['daily_steps']
        sleep_hours = st.session_state.user_data['sleep_hours']
        water_intake = st.session_state.user_data['water_intake']
        
        # Calculate key metrics
        bmi = calculate_bmi(weight, height)
        bmi_category, _ = interpret_bmi(bmi)
        bmr = calculate_bmr(weight, height, age, gender)
        tdee = calculate_tdee(bmr, activity_level)
        
        # Display overview metrics
        st.subheader("Health Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("BMI", f"{bmi:.1f}", f"{bmi_category}")
        with col2:
            st.metric("Daily Steps", f"{daily_steps:,}", "Goal: 10,000")
        with col3:
            st.metric("Sleep", f"{sleep_hours} hrs", "Goal: 7-9 hrs")
        with col4:
            st.metric("Water", f"{water_intake} L", "Goal: 2-3 L")
        
        # Create tabs for different dashboard sections
        tab1, tab2, tab3, tab4 = st.tabs(["Fitness", "Nutrition", "Weight", "Wellness Score"])
        
        with tab1:
            st.subheader("Fitness Metrics")
            
            # Activity chart
            activity_chart = plot_daily_activity()
            if activity_chart:
                st.plotly_chart(activity_chart, use_container_width=True)
            
            # Exercise distribution
            exercise_chart = plot_exercise_distribution()
            if exercise_chart:
                st.plotly_chart(exercise_chart, use_container_width=True)
            else:
                st.info("No exercise data available. Track your exercise in the Fitness Tracker.")
        
        with tab2:
            st.subheader("Nutrition Overview")
            
            # Nutrition chart
            nutrition_chart = plot_nutrition_breakdown()
            if nutrition_chart:
                st.plotly_chart(nutrition_chart, use_container_width=True)
            else:
                st.info("No nutrition data available. Track your nutrition in the Nutrition Planner.")
            
            # Display TDEE and recommendations
            st.metric("Daily Energy Expenditure", f"{tdee} calories")
            
            # Macronutrient recommendations
            st.subheader("Recommended Macronutrient Breakdown")
            goal = 'maintain'
            if 'Lose weight' in st.session_state.user_data['health_goals']:
                goal = 'lose_weight'
            elif 'Gain muscle' in st.session_state.user_data['health_goals']:
                goal = 'gain_muscle'
            
            macros = calculate_macro_nutrients(tdee, goal)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Protein", f"{macros['protein']} g", f"{round(macros['protein']*4/macros['calories']*100)}%")
            with col2:
                st.metric("Carbs", f"{macros['carbs']} g", f"{round(macros['carbs']*4/macros['calories']*100)}%")
            with col3:
                st.metric("Fats", f"{macros['fats']} g", f"{round(macros['fats']*9/macros['calories']*100)}%")
        
        with tab3:
            st.subheader("Weight Trends")
            
            # Weight trend chart
            weight_fig = plot_weight_trend()
            if weight_fig:
                st.plotly_chart(weight_fig, use_container_width=True)
            else:
                st.info("No weight history available. Track your weight in the Health Metrics section.")
            
            # BMI visualization
            st.subheader("BMI Category")
            bmi_fig = go.Figure()
            
            bmi_categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
            bmi_ranges = [0, 18.5, 25, 30, 40]
            bmi_colors = ['yellow', 'green', 'orange', 'red']
            
            for i, category in enumerate(bmi_categories):
                bmi_fig.add_trace(go.Bar(
                    x=[category],
                    y=[bmi_ranges[i+1] - bmi_ranges[i]],
                    base=[bmi_ranges[i]],
                    marker_color=bmi_colors[i],
                    name=category
                ))
            
            # Add marker for current BMI
            bmi_fig.add_trace(go.Scatter(
                x=[bmi_categories[0], bmi_categories[-1]],
                y=[bmi, bmi],
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name='Your BMI'
            ))
            
            bmi_fig.update_layout(
                title="BMI Classification",
                xaxis_title="Category",
                yaxis_title="BMI Range",
                barmode='stack',
                yaxis=dict(range=[0, 40])
            )
            
            st.plotly_chart(bmi_fig, use_container_width=True)
        
        with tab4:
            st.subheader("Wellness Score")
            
            # Calculate wellness score based on various metrics
            scores = {}
            
            # BMI score (0-100)
            if bmi < 18.5:
                scores['bmi'] = 70  # Underweight
            elif bmi < 25:
                scores['bmi'] = 100  # Normal weight
            elif bmi < 30:
                scores['bmi'] = 70  # Overweight
            else:
                scores['bmi'] = 50  # Obese
            
            # Activity score (0-100)
            if daily_steps >= 10000:
                scores['activity'] = 100
            else:
                scores['activity'] = min(100, daily_steps / 10000 * 100)
            
            # Sleep score (0-100)
            if 7 <= sleep_hours <= 9:
                scores['sleep'] = 100
            elif 6 <= sleep_hours < 7 or 9 < sleep_hours <= 10:
                scores['sleep'] = 80
            else:
                scores['sleep'] = 60
            
            # Water intake score (0-100)
            recommended_water = calculate_water_intake(weight, activity_level)
            if water_intake >= recommended_water:
                scores['water'] = 100
            else:
                scores['water'] = min(100, water_intake / recommended_water * 100)
            
            # Nutrition score (simplistic approach)
            nutrition = st.session_state.user_data['nutrition']
            if sum(nutrition.values()) > 0:  # If nutrition data exists
                protein_percent = nutrition['protein'] * 4 / nutrition['calories'] if nutrition['calories'] > 0 else 0
                if 0.15 <= protein_percent <= 0.35:
                    scores['nutrition'] = 100
                else:
                    scores['nutrition'] = 70
            else:
                scores['nutrition'] = 0
            
            # Calculate overall wellness score
            if scores:
                wellness_score = sum(scores.values()) / len(scores)
            else:
                wellness_score = 0
            
            # Display wellness score gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = wellness_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Wellness Score"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "blue"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': wellness_score
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed score breakdown
            st.subheader("Wellness Score Breakdown")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("BMI Score", f"{scores['bmi']}/100")
            with col2:
                st.metric("Activity Score", f"{scores['activity']}/100")
            with col3:
                st.metric("Sleep Score", f"{scores['sleep']}/100")
            with col4:
                st.metric("Water Score", f"{scores['water']}/100")
            with col5:
                st.metric("Nutrition Score", f"{scores['nutrition']}/100")

# Health Metrics Page
elif page == "Health Metrics":
    st.title("🏥 Personal Health Metrics")
    
    with st.form("health_info_form"):
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,
                value=st.session_state.user_data['age']
            )
            gender = st.selectbox(
                "Gender",
                ["", "Male", "Female", "Other"],
                index=0 if not st.session_state.user_data['gender'] else 
                       ["", "Male", "Female", "Other"].index(st.session_state.user_data['gender'])
            )
            
            # Height input with units
            if st.session_state.units == 'Metric':
                height = st.number_input(
                    "Height (cm)",
                    min_value=0.0,
                    value=st.session_state.user_data['height']
                )
            else:
                # Convert to inches for display
                height_in = convert_height(st.session_state.user_data['height'], 'in') if st.session_state.user_data['height'] > 0 else 0
                height_feet = int(height_in / 12)
                height_inches = height_in % 12
                
                feet = st.number_input("Height (feet)", min_value=0, max_value=8, value=height_feet)
                inches = st.number_input("Height (inches)", min_value=0, max_value=11, value=int(height_inches))
                
                # Convert back to cm for storage
                height = (feet * 12 + inches) / 0.393701
        
        with col2:
            # Weight input with units
            if st.session_state.units == 'Metric':
                weight = st.number_input(
                    "Weight (kg)",
                    min_value=0.0,
                    value=st.session_state.user_data['weight']
                )
            else:
                # Convert to lbs for display
                weight_lbs = convert_weight(st.session_state.user_data['weight'], 'lbs') if st.session_state.user_data['weight'] > 0 else 0
                weight_input = st.number_input("Weight (lbs)", min_value=0.0, value=float(weight_lbs), step=0.1)
                
                # Convert back to kg for storage
                weight = weight_input / 2.20462
            
            activity_level = st.selectbox(
                "Activity Level",
                ["", "Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
                index=0 if not st.session_state.user_data['activity_level'] else 
                       ["", "Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"].index(st.session_state.user_data['activity_level'])
            )
        
        st.subheader("Health Goals and Conditions")
        
        health_goals = st.multiselect(
            "Health Goals",
            ["Lose weight", "Gain muscle", "Improve cardiovascular health", "Better sleep", "Increase flexibility", "Reduce stress"],
            default=st.session_state.user_data['health_goals']
        )
        
        chronic_conditions = st.multiselect(
            "Chronic Conditions (if any)",
            ["Diabetes", "Hypertension", "High cholesterol", "Asthma", "Arthritis", "None"],
            default=st.session_state.user_data['chronic_conditions']
        )
        
        st.subheader("Weight History")
        has_weight_history = st.checkbox("Track weight history")
        
        if has_weight_history:
            weight_history = {}
            weight_history_date = st.date_input("Date", date.today())
            
            if st.session_state.units == 'Metric':
                weight_history_value = st.number_input("Weight (kg)", min_value=0.0, key="weight_history_kg")
            else:
                weight_lbs = st.number_input("Weight (lbs)", min_value=0.0, key="weight_history_lbs")
                weight_history_value = weight_lbs / 2.20462  # Convert to kg for storage
            
            add_weight_entry = st.form_submit_button("Add Weight Entry")
            if add_weight_entry:
                weight_history = st.session_state.user_data['weight_history'].copy()
                weight_history[weight_history_date.strftime("%Y-%m-%d")] = weight_history_value
                st.session_state.user_data['weight_history'] = weight_history
        
        # Submit button
        submit_button = st.form_submit_button("Calculate Health Metrics")
        
        if submit_button:
            # Update session state
            st.session_state.user_data.update({
                'age': age,
                'gender': gender,
                'height': height,
                'weight': weight,
                'activity_level': activity_level,
                'health_goals': health_goals,
                'chronic_conditions': chronic_conditions
            })
    
    # Only show health metrics if we have basic data
    if st.session_state.user_data['height'] > 0 and st.session_state.user_data['weight'] > 0:
        st.subheader("Health Metrics")
        
        # Calculate metrics
        weight = st.session_state.user_data['weight']
        height = st.session_state.user_data['height']
        age = st.session_state.user_data['age']
        gender = st.session_state.user_data['gender']
        activity = st.session_state.user_data['activity_level']
        
        bmi = calculate_bmi(weight, height)
        bmi_category, bmi_color = interpret_bmi(bmi)
        
        bmr = calculate_bmr(weight, height, age, gender) if age > 0 and gender else 0
        tdee = calculate_tdee(bmr, activity) if bmr > 0 and activity else 0
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        # Convert units for display if needed
        if st.session_state.units == 'Imperial':
            display_weight = convert_weight(weight, 'lbs')
            display_height = convert_height(height, 'in')
            weight_unit = "lbs"
            height_unit = "in"
        else:
            display_weight = weight
            display_height = height
            weight_unit = "kg"
            height_unit = "cm"
        
        col1.metric("Weight", f"{display_weight:.1f} {weight_unit}")
        col1.metric("Height", f"{display_height:.1f} {height_unit}")
        col1.metric("BMI", f"{bmi} ({bmi_category})")
        
        col2.metric("Basal Metabolic Rate", f"{bmr} calories")
        col2.metric("Daily Energy Expenditure", f"{tdee} calories")
        
        # Water intake recommendation
        water_intake = calculate_water_intake(weight, activity)
        col3.metric("Water Intake Recommendation", f"{water_intake} liters")
        
        # If we have weight history, show the trend
        if st.session_state.user_data['weight_history']:
            weight_fig = plot_weight_trend()
            if weight_fig:
                st.plotly_chart(weight_fig, use_container_width=True)
                
                # Show weight history in a table
                st.subheader("Weight History")
                weight_history = st.session_state.user_data['weight_history']
                
                # Convert to display units
                weight_history_display = {}
                for date, weight in weight_history.items():
                    if st.session_state.units == 'Imperial':
                        weight_history_display[date] = convert_weight(weight, 'lbs')
                    else:
                        weight_history_display[date] = weight
                
                # Create a DataFrame
                weight_df = pd.DataFrame(list(weight_history_display.items()), columns=['Date', 'Weight'])
                weight_df['Weight'] = weight_df['Weight'].round(1)
                weight_df['Weight'] = weight_df['Weight'].astype(str) + f" {weight_unit}"
                
                st.dataframe(weight_df, use_container_width=True)
        # Calculate body composition metrics
        body_fat = estimate_body_fat_percentage(gender, age, bmi)
        body_fat_category, body_fat_color = interpret_body_fat(gender, body_fat)
        ideal_weight_min, ideal_weight_max = calculate_ideal_weight(height, gender)

    # Display body composition metrics
    st.subheader("Body Composition Analysis")
    col1, col2 = st.columns(2)

    with col1:
        bmi = calculate_bmi(weight, height)
        body_fat = estimate_body_fat_percentage(gender, age, bmi)
        body_fat_category, body_fat_color = interpret_body_fat(gender, body_fat)
        ideal_weight_min, ideal_weight_max = calculate_ideal_weight(height, gender)
        st.metric("Estimated Body Fat", f"{body_fat:.1f}%")
        st.write(f"Category: {body_fat_category}")
        
        # Display in appropriate units
        if st.session_state.units == 'Imperial':
            min_weight_lbs = convert_weight(ideal_weight_min, 'lbs')
            max_weight_lbs = convert_weight(ideal_weight_max, 'lbs')
            st.metric("Ideal Weight Range", f"{min_weight_lbs:.1f} - {max_weight_lbs:.1f} lbs")
        else:
            st.metric("Ideal Weight Range", f"{ideal_weight_min:.1f} - {ideal_weight_max:.1f} kg")

    with col2:
        # Visualization of body fat percentage
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = body_fat,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Body Fat %"},
            gauge = {
                'axis': {'range': [0, 50]},
                'bar': {'color': body_fat_color},
                'steps': [
                    {'range': [0, 6], 'color': "lightgray"},
                    {'range': [6, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "pink"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': body_fat
                }
            }
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
        
    # Health Report Generator
    st.subheader("Comprehensive Health Report")
    if st.button("Generate Health Report"):
        report = generate_health_report(st.session_state.user_data)
        st.markdown(report)
        
        # Option to download the report
        st.download_button(
            label="Download Report",
            data=report,
            file_name="health_report.md",
            mime="text/markdown"
        )


# Fitness Tracker Page
elif page == "Fitness Tracker":
    st.title("🏃‍♂️ Fitness Tracker")
    
    if st.session_state.user_data['age'] == 0 or not st.session_state.user_data['gender']:
        st.warning("Please complete your profile in the Health Metrics section first.")
        if st.button("Go to Health Metrics"):
            st.experimental_rerun()
    else:
        with st.form("fitness_tracker_form"):
            st.subheader("Daily Activity")
            col1, col2 = st.columns(2)
            
            with col1:
                daily_steps = st.number_input(
                    "Daily Steps",
                    min_value=0,
                    value=st.session_state.user_data['daily_steps']
                )
                sleep_hours = st.number_input(
                    "Sleep Hours",
                    min_value=0.0,
                    max_value=24.0,
                    value=st.session_state.user_data['sleep_hours']
                )
            
            with col2:
                water_intake = st.number_input(
                    "Water Intake (liters)",
                    min_value=0.0,
                    value=st.session_state.user_data['water_intake']
                )
            
            st.subheader("Exercise Log")
            exercise_type = st.selectbox(
                "Exercise Type",
                ["Cardio", "Strength Training", "Flexibility", "Sports", "Other"]
            )
            
            exercise_minutes = st.number_input(
                "Duration (minutes)",
                min_value=0
            )
            
            add_exercise = st.form_submit_button("Add Exercise")
            if add_exercise:
                exercise_data = st.session_state.user_data['exercise_minutes'].copy()
                if exercise_type in exercise_data:
                    exercise_data[exercise_type] += exercise_minutes
                else:
                    exercise_data[exercise_type] = exercise_minutes
                    
                st.session_state.user_data['exercise_minutes'] = exercise_data
            
            # Submit button
            submit_button = st.form_submit_button("Update Fitness Data")
            
            if submit_button:
                # Update session state
                st.session_state.user_data.update({
                    'daily_steps': daily_steps,
                    'sleep_hours': sleep_hours,
                    'water_intake': water_intake
                })
        
        # Display fitness metrics and recommendations
        st.subheader("Fitness Metrics and Recommendations")
        
        # Calculate targets
        age = st.session_state.user_data['age']
        activity_level = st.session_state.user_data['activity_level']
        health_goals = st.session_state.user_data['health_goals']
        
        steps_target = calculate_steps_target(age, activity_level)
        exercise_target = calculate_exercise_minutes_target(age, health_goals)
        sleep_target = calculate_sleep_target(age)
        
        # Display metrics and targets
        col1, col2, col3 = st.columns(3)
        
        steps_percentage = min(100, int((daily_steps / steps_target) * 100))
        col1.metric("Daily Steps", f"{daily_steps:,}", f"{steps_percentage}% of target")
        col1.progress(steps_percentage / 100)
        
        total_exercise = sum(st.session_state.user_data['exercise_minutes'].values())
        exercise_percentage = min(100, int((total_exercise / exercise_target) * 100))
        col2.metric("Weekly Exercise", f"{total_exercise} minutes", f"{exercise_percentage}% of target")
        col2.progress(exercise_percentage / 100)
        
        col3.metric("Sleep Hours", f"{sleep_hours} hours", f"Target: {sleep_target}")
        
        # Display activity and exercise charts
        activity_chart = plot_daily_activity()
        if activity_chart:
            st.plotly_chart(activity_chart, use_container_width=True)
        
        exercise_chart = plot_exercise_distribution()
        if exercise_chart:
            st.plotly_chart(exercise_chart, use_container_width=True)
            
        # Display exercise log
        if st.session_state.user_data['exercise_minutes']:
            st.subheader("Exercise Log")
            exercise_df = pd.DataFrame({
                'Exercise Type': st.session_state.user_data['exercise_minutes'].keys(),
                'Minutes': st.session_state.user_data['exercise_minutes'].values()
            })
            st.dataframe(exercise_df, use_container_width=True)
            
            if st.button("Reset Exercise Log"):
                st.session_state.user_data['exercise_minutes'] = {}
                st.experimental_rerun()

# Nutrition Planner Page
elif page == "Nutrition Planner":
    st.title("🥗 Nutrition Planner")
    
    if st.session_state.user_data['age'] == 0 or not st.session_state.user_data['gender']:
        st.warning("Please complete your profile in the Health Metrics section first.")
        if st.button("Go to Health Metrics"):
            st.experimental_rerun()
    else:
        # Calculate recommended macros
        weight = st.session_state.user_data['weight']
        height = st.session_state.user_data['height']
        age = st.session_state.user_data['age']
        gender = st.session_state.user_data['gender']
        activity = st.session_state.user_data['activity_level']
        health_goals = st.session_state.user_data['health_goals']
        
        # Default to 'maintain' if no goal is set
        goal = 'maintain'
        if 'Lose weight' in health_goals:
            goal = 'lose_weight'
        elif 'Gain muscle' in health_goals:
            goal = 'gain_muscle'
        
        bmr = calculate_bmr(weight, height, age, gender)
        tdee = calculate_tdee(bmr, activity)
        recommended = calculate_macro_nutrients(tdee, goal)
        
        # Display recommended macros
        st.subheader("Recommended Daily Nutrition")
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Calories", f"{recommended['calories']} kcal")
        col2.metric("Protein", f"{recommended['protein']} g")
        col3.metric("Carbohydrates", f"{recommended['carbs']} g")
        col4.metric("Fats", f"{recommended['fats']} g")
        
        # Nutrition tracking form
        with st.form("nutrition_tracking_form"):
            st.subheader("Track Daily Nutrition")
            
            col1, col2 = st.columns(2)
            
            with col1:
                calories = st.number_input("Calories (kcal)", min_value=0, value=st.session_state.user_data['nutrition']['calories'])
                protein = st.number_input("Protein (g)", min_value=0, value=st.session_state.user_data['nutrition']['protein'])
            
            with col2:
                carbs = st.number_input("Carbohydrates (g)", min_value=0, value=st.session_state.user_data['nutrition']['carbs'])
                fats = st.number_input("Fats (g)", min_value=0, value=st.session_state.user_data['nutrition']['fats'])
            
            # Submit button
            submit_button = st.form_submit_button("Update Nutrition Data")
            
            if submit_button:
                # Update session state
                st.session_state.user_data['nutrition'].update({
                    'calories': calories,
                    'protein': protein,
                    'carbs': carbs,
                    'fats': fats
                })
        
        # Display nutrition charts
        nutrition_chart = plot_nutrition_breakdown()
        if nutrition_chart:
            st.plotly_chart(nutrition_chart, use_container_width=True)
        
        # Meal suggestion feature
        st.subheader("Meal Suggestions")
        
        meal_type = st.selectbox(
            "Meal Type",
            ["Breakfast", "Lunch", "Dinner", "Snack"]
        )
        
        dietary_restrictions = st.multiselect(
            "Dietary Restrictions",
            ["Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Keto", "Low-Carb", "None"]
        )
        
        if st.button("Get Meal Suggestions"):
            st.subheader(f"Suggested {meal_type} Ideas")
            
            if "Vegetarian" in dietary_restrictions or "Vegan" in dietary_restrictions:
                if meal_type == "Breakfast":
                    st.write("1. Overnight oats with almond milk, chia seeds, and berries")
                    st.write("2. Tofu scramble with vegetables and avocado")
                    st.write("3. Smoothie bowl with plant-based protein powder")
                elif meal_type == "Lunch":
                    st.write("1. Quinoa bowl with roasted vegetables and tahini dressing")
                    st.write("2. Lentil soup with whole grain bread")
                    st.write("3. Mediterranean chickpea salad")
                elif meal_type == "Dinner":
                    st.write("1. Stir-fried tofu with vegetables and brown rice")
                    st.write("2. Chickpea curry with cauliflower rice")
                    st.write("3. Bean and vegetable chili")
                else:  # Snack
                    st.write("1. Apple slices with almond butter")
                    st.write("2. Hummus with carrot sticks")
                    st.write("3. Trail mix with nuts and dried fruit")
            elif "Keto" in dietary_restrictions or "Low-Carb" in dietary_restrictions:
                if meal_type == "Breakfast":
                    st.write("1. Avocado and bacon omelet")
                    st.write("2. Chia seed pudding with coconut milk")
                    st.write("3. Greek yogurt with nuts and berries")
                elif meal_type == "Lunch":
                    st.write("1. Caesar salad with grilled chicken (no croutons)")
                    st.write("2. Tuna salad lettuce wraps")
                    st.write("3. Zucchini noodles with pesto and grilled salmon")
                elif meal_type == "Dinner":
                    st.write("1. Grilled steak with asparagus")
                    st.write("2. Baked chicken thighs with roasted broccoli")
                    st.write("3. Stuffed bell peppers with ground turkey and cauliflower rice")
                else:  # Snack
                    st.write("1. Cheese cubes and olives")
                    st.write("2. Hard-boiled eggs")
                    st.write("3. Cucumber slices with cream cheese")
            else:
                if meal_type == "Breakfast":
                    st.write("1. Greek yogurt parfait with granola and fruit")
                    st.write("2. Veggie omelet with whole grain toast")
                    st.write("3. Oatmeal with banana and peanut butter")
                elif meal_type == "Lunch":
                    st.write("1. Grilled chicken salad with mixed greens")
                    st.write("2. Turkey and avocado wrap")
                    st.write("3. Quinoa bowl with roasted vegetables and salmon")
                elif meal_type == "Dinner":
                    st.write("1. Baked fish with sweet potato and steamed vegetables")
                    st.write("2. Whole grain pasta with lean turkey meatballs")
                    st.write("3. Stir-fry with brown rice and chicken breast")
                else:  # Snack
                    st.write("1. Greek yogurt with honey")
                    st.write("2. Apple with peanut butter")
                    st.write("3. Whole grain crackers with hummus")
            
            # Add nutritional information for educational purposes
            st.info(
                f"For your {goal.replace('_', ' ')} goal, focus on meals with "
                f"approximately {recommended['calories'] // 3} calories per main meal, "
                f"with {recommended['protein'] // 3}g protein, "
                f"{recommended['carbs'] // 3}g carbs, and "
                f"{recommended['fats'] // 3}g fats."
            )

        # Meal planning calendar
        st.subheader("Weekly Meal Planning")

        with st.form("meal_planning_form"):
            selected_day = st.selectbox("Select Day", days)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Breakfast**")
                breakfast = st.text_area(f"{selected_day} Breakfast", height=100, key=f"breakfast_{selected_day}")
            
            with col2:
                st.write("**Lunch**")
                lunch = st.text_area(f"{selected_day} Lunch", height=100, key=f"lunch_{selected_day}")
            
            with col3:
                st.write("**Dinner**")
                dinner = st.text_area(f"{selected_day} Dinner", height=100, key=f"dinner_{selected_day}")
            
            # Correct form submit button
            submit_meal_plan = st.form_submit_button("Save Meal Plan")
            
            if submit_meal_plan:
                st.success(f"Meal plan for {selected_day} saved successfully!")

# Medication Tracker Page
elif page == "Medication Tracker":
    st.title("💊 Medication Tracker")
    
    # Initialize medication data structure if not exists
    if 'medications' not in st.session_state.user_data:
        st.session_state.user_data['medications'] = []
    
    # Add new medication form
    with st.form("add_medication_form"):
        st.subheader("Add New Medication")
        
        med_name = st.text_input("Medication Name")
        dosage = st.text_input("Dosage (e.g., 10mg)")
        
        col1, col2 = st.columns(2)
        with col1:
            frequency = st.selectbox(
                "Frequency",
                ["Once daily", "Twice daily", "Three times daily", "Every 4 hours", "Every 6 hours", "Every 8 hours", "As needed"]
            )
            start_date = st.date_input("Start Date", date.today())
        
        with col2:
            time_of_day = st.selectbox(
                "Time of Day",
                ["Morning", "Noon", "Evening", "Bedtime", "Multiple times", "With meals"]
            )
            duration = st.text_input("Duration (e.g., 10 days, ongoing)")
        
        notes = st.text_area("Special Instructions (e.g., take with food)")
        
        submit_med = st.form_submit_button("Add Medication")
        
        if submit_med and med_name:
            # Convert frequency to hours for calculations
            freq_hours = {
                "Once daily": 24,
                "Twice daily": 12,
                "Three times daily": 8,
                "Every 4 hours": 4,
                "Every 6 hours": 6,
                "Every 8 hours": 8,
                "As needed": 0
            }
            
            # Add medication to list
            st.session_state.user_data['medications'].append({
                'name': med_name,
                'dosage': dosage,
                'frequency': frequency,
                'frequency_hours': freq_hours.get(frequency, 0),
                'time_of_day': time_of_day,
                'start_date': start_date.strftime("%Y-%m-%d"),
                'duration': duration,
                'notes': notes,
                'last_taken': None
            })
            
            st.success(f"{med_name} added to your medication list!")
    
    # Display current medications
    if st.session_state.user_data['medications']:
        st.subheader("Current Medications")
        
        for i, med in enumerate(st.session_state.user_data['medications']):
            with st.expander(f"{med['name']} - {med['dosage']}"):
                st.write(f"**Frequency:** {med['frequency']}")
                st.write(f"**Time of Day:** {med['time_of_day']}")
                st.write(f"**Started:** {med['start_date']}")
                st.write(f"**Duration:** {med['duration']}")
                
                if med['notes']:
                    st.write(f"**Special Instructions:** {med['notes']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"Mark as Taken", key=f"take_{i}"):
                        st.session_state.user_data['medications'][i]['last_taken'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.success(f"Marked {med['name']} as taken at {datetime.now().strftime('%H:%M')}")
                
                with col2:
                    if st.button(f"Remove Medication", key=f"remove_{i}"):
                        removed = st.session_state.user_data['medications'].pop(i)
                        st.warning(f"Removed {removed['name']} from your medication list")
                        st.experimental_rerun()
                
                if med['last_taken']:
                    st.write(f"**Last Taken:** {med['last_taken']}")
                    if med['frequency_hours'] > 0:
                        last_taken = datetime.strptime(med['last_taken'], "%Y-%m-%d %H:%M:%S")
                        next_dose = calculate_next_dose(last_taken, med['frequency_hours'])
                        st.write(f"**Next Dose Due:** {next_dose.strftime('%Y-%m-%d %H:%M')}")
        
        # Check for potential interactions
        med_names = [med['name'] for med in st.session_state.user_data['medications']]
        interactions = check_medication_interactions(med_names)
        
        if interactions:
            st.subheader("Potential Medication Interactions")
            for interaction in interactions:
                st.warning(interaction)
            st.info("Please consult your healthcare provider about these potential interactions.")
        
        # Medication schedule calendar
        st.subheader("Medication Schedule")
        
        # Create a simple calendar view
        today = date.today()
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Create a weekly schedule
        schedule_df = pd.DataFrame(columns=["Medication"] + days_of_week)
        
        for med in st.session_state.user_data['medications']:
            row = {"Medication": f"{med['name']} ({med['dosage']})"}
            
            for day in days_of_week:
                row[day] = med['time_of_day'] if med['frequency_hours'] > 0 else "As needed"
            
            schedule_df = pd.concat([schedule_df, pd.DataFrame([row])], ignore_index=True)
        
        st.table(schedule_df)
        
        # Download medication list
        if st.button("Download Medication List"):
            meds_df = pd.DataFrame(st.session_state.user_data['medications'])
            csv = meds_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="my_medications.csv",
                mime="text/csv"
            )
    else:
        st.info("You haven't added any medications yet.")

# Health Advisor Chat Page
elif page == "Health Advisor Chat":
    st.title("👨‍⚕️ Health Advisor Chat")
    
    # Display chat history
    st.subheader("Chat History")
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Health Advisor:** {message['content']}")
                
    # Symptom checker
    st.subheader("Symptom Checker")
    with st.expander("Check Common Symptoms"):
        symptom_input = st.text_area("Describe your symptoms", height=100)
        if st.button("Check Symptoms"):
            if symptom_input:
                symptom_info = simple_symptom_checker(symptom_input)
                st.markdown(symptom_info)
                
                # Add this interaction to the chat history
                st.session_state.chat_history.append({"role": "user", "content": f"I have symptoms: {symptom_input}"})
                st.session_state.chat_history.append({"role": "assistant", "content": symptom_info})
    
    # User input
    with st.form("chat_form"):
        user_message = st.text_area("Your health question:", height=100)
        submit_button = st.form_submit_button("Send")
        
        if submit_button and user_message:
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": user_message})
            
            # Get AI response using local LLM
            response = get_health_advice_llm(user_message)
            
            # Add AI response to history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update the UI
            st.experimental_rerun()
    
    # Option to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()
    
    # Display usage tips
    with st.expander("Usage Tips"):
        st.markdown("""
        **Tips for using the Health Advisor:**
        
        - Be specific in your questions
        - Provide relevant health information for more personalized advice
        - Ask about general health topics, lifestyle recommendations, or wellness tips
        - Remember that this advisor cannot diagnose medical conditions or replace professional medical advice
        - For serious health concerns, please consult a healthcare professional
        """)
    
    # Disclaimer
    st.info(
        "**Disclaimer:** This health advisor provides general information only and is not a substitute "
        "for professional medical advice, diagnosis, or treatment. Always seek the advice of your "
        "physician or other qualified health provider with any questions you may have regarding a "
        "medical condition."
    )

# Add a footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "HealthBot - Your Personal Health Assistant<br>"
    "This application is for educational purposes only."
    "</div>", 
    unsafe_allow_html=True
)