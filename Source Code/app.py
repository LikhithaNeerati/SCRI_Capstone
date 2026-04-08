from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import requests
import folium
import streamlit as st
import openrouteservice
from streamlit_folium import folium_static
from tensorflow.keras.models import load_model
from transformers import pipeline

# API Keys from environment variables
HF_API_KEY = os.getenv("HF_API_KEY", "")
ORS_API_KEY = os.getenv("ORS_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

os.environ["HUGGINGFACE_TOKEN"] = HF_API_KEY
client = openrouteservice.Client(key=ORS_API_KEY)

# Hugging Face text generation
hf_pipeline = pipeline("text-generation", model="gpt2")

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_customer = load_model(r"C:\Users\Likhita\OneDrive\Documents\Likhitha\Projects\SCRI_Capstone-main\SCRI_Capstone-main\Models\customer_lstm_model (1).h5")
model_retailer = load_model(r"C:\Users\Likhita\OneDrive\Documents\Likhitha\Projects\SCRI_Capstone-main\SCRI_Capstone-main\Models\retailer_mlp_model (1).h5")
model_logistics = load_model(r"C:\Users\Likhita\OneDrive\Documents\Likhitha\Projects\SCRI_Capstone-main\SCRI_Capstone-main\Models\logistics_mlp_model.h5")
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    if data.get("main") and data.get("wind") and "coord" in data:
        return {
            "temp": data["main"].get("temp", 25),
            "wind": data["wind"].get("speed", 5),
            "rain": data.get("rain", {}).get("1h", 0.0),
            "lat": data["coord"]["lat"],
            "lon": data["coord"]["lon"]
        }
    return {"temp": 25, "wind": 5, "rain": 0.0, "lat": 0.0, "lon": 0.0}

def generate_insight(city, product, score):
    context = (
        f"A customer in {city} has ordered a {product}. "
        f"The estimated delivery risk score is {score}. "
        "Based on this, provide a recommendation to the customer about whether to proceed with the order, and explain why."
    )
    result = hf_pipeline(context, max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.7)
    return result[0]["generated_text"]

# Title
st.title("AI-Powered Supply Chain Risk Predictor")

# Tabs
tab1, tab2, tab3 = st.tabs(["Customer", "Retailer", "Logistics"])

# Tab 1: Customer Risk Prediction
with tab1:
    st.header("Customer Risk Prediction")
    customer_city = st.text_input("Enter your city")
    product = st.text_input("Enter the product you want to order")

    if customer_city and product:
        if st.button("Check Risk and Get Recommendation"):
            with st.spinner("Fetching weather and analyzing risk..."):
                try:
                    weather = get_weather(customer_city)
                    st.write("### Weather Conditions at Your Location:")
                    st.json(weather)

                    m = folium.Map(location=[weather["lat"], weather["lon"]], zoom_start=10)
                    folium.Marker([weather["lat"], weather["lon"]], tooltip="Customer Location").add_to(m)
                    folium_static(m)

                    input_array = np.array([[weather["temp"], weather["wind"], weather["rain"]]] * 7).reshape(1, 7, 3)
                    risk_score = int(model_customer.predict(input_array)[0][0] * 100)

                    st.metric("📊 Delivery Risk Score", risk_score)

                    insight = generate_insight(customer_city, product, risk_score)
                    st.subheader("📝 AI Insight and Suggested Action")
                    st.info(insight)

                except Exception as e:
                    st.error(f"Error during risk assessment: {e}")

# Tab 2: Retailer
with tab2:
    st.header("Retailer Delivery Route Details")
    customer_city = st.text_input("Customer Location")
    warehouse_city = st.text_input("Warehouse Location", key="warehouse_location_retailer")

    if customer_city and warehouse_city:
        warehouse_weather = get_weather(warehouse_city)
        customer_weather = get_weather(customer_city)

        start_coords = (warehouse_weather["lon"], warehouse_weather["lat"])
        end_coords = (customer_weather["lon"], customer_weather["lat"])

        try:
            route = client.directions(
                coordinates=[start_coords, end_coords],
                profile='driving-car',
                format='geojson',
                instructions=True
            )

            m = folium.Map(location=[
                (warehouse_weather["lat"] + customer_weather["lat"]) / 2,
                (warehouse_weather["lon"] + customer_weather["lon"]) / 2
            ], zoom_start=6)

            folium.Marker([warehouse_weather["lat"], warehouse_weather["lon"]],
                          tooltip="Warehouse", icon=folium.Icon(color="blue")).add_to(m)
            folium.Marker([customer_weather["lat"], customer_weather["lon"]],
                          tooltip="Customer", icon=folium.Icon(color="green")).add_to(m)
            folium.PolyLine(
                locations=[(coord[1], coord[0]) for coord in route['features'][0]['geometry']['coordinates']],
                color='purple', weight=4
            ).add_to(m)
            folium_static(m)

            properties = route["features"][0]["properties"]
            summary = properties["summary"]
            steps = properties.get("segments", [])[0].get("steps", [])

            st.subheader("Route Summary")
            st.write(f"**Distance:** {round(summary['distance'] / 1000, 2)} km")
            st.write(f"**Estimated Time:** {round(summary['duration'] / 60, 2)} minutes")
            st.json({"steps": [{"instruction": s["instruction"], "distance_m": s["distance"]} for s in steps]})

        except Exception as e:
            st.error(f"Route generation error: {e}")

# Tab 3: Logistics
with tab3:
    st.header("Logistics Warehouse Route Details")
    city = st.text_input("Destination City")
    product = st.text_input("Product to Deliver")
    warehouse_city = st.text_input("Warehouse Location", key="warehouse_location_logistics")

    if city and warehouse_city and product:
        destination_weather = get_weather(city)
        warehouse_weather = get_weather(warehouse_city)

        start_coords = (warehouse_weather["lon"], warehouse_weather["lat"])
        end_coords = (destination_weather["lon"], destination_weather["lat"])

        try:
            route = client.directions(
                coordinates=[start_coords, end_coords],
                profile='driving-car',
                format='geojson',
                instructions=True
            )

            m = folium.Map(location=[
                (warehouse_weather["lat"] + destination_weather["lat"]) / 2,
                (warehouse_weather["lon"] + destination_weather["lon"]) / 2
            ], zoom_start=6)

            folium.Marker([warehouse_weather["lat"], warehouse_weather["lon"]],
                          tooltip="Warehouse", icon=folium.Icon(color="red")).add_to(m)
            folium.Marker([destination_weather["lat"], destination_weather["lon"]],
                          tooltip="Destination", icon=folium.Icon(color="orange")).add_to(m)
            folium.PolyLine(
                locations=[(c[1], c[0]) for c in route['features'][0]['geometry']['coordinates']],
                color='darkred', weight=4
            ).add_to(m)
            folium_static(m)

            properties = route["features"][0]["properties"]
            summary = properties["summary"]
            steps = properties.get("segments", [])[0].get("steps", [])

            st.subheader("Logistics Route Summary")
            st.write(f"**Distance:** {round(summary['distance'] / 1000, 2)} km")
            st.write(f"**Estimated Time:** {round(summary['duration'] / 60, 2)} minutes")
            st.json({"steps": [{"instruction": s["instruction"], "distance_m": s["distance"]} for s in steps]})

        except Exception as e:
            st.error(f"Routing error: {e}")