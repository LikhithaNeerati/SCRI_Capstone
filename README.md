# 🔗 Supply Chain Resilience Index (SCRI)

An AI-powered web application that predicts delivery risk and optimizes logistics routing using real-time weather data, machine learning models, and generative AI.

---

## 📌 Overview

The SCRI application helps customers, retailers, and logistics teams assess supply chain risk and plan delivery routes intelligently. It integrates live weather data, trained ML models, and interactive maps to provide actionable insights across three user personas.

---

## ✨ Features

- 👤 **Customer Tab** — Enter a city and product to get a real-time delivery risk score powered by an LSTM model trained on weather data, with an AI-generated recommendation
- 🏪 **Retailer Tab** — Visualize the optimal driving route between a warehouse and customer location with distance and time estimates
- 🚚 **Logistics Tab** — Plan warehouse-to-destination delivery routes with full step-by-step routing details on an interactive map

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| Web App | Streamlit |
| Machine Learning | TensorFlow / Keras (LSTM, MLP) |
| Generative AI | HuggingFace Transformers (GPT-2) |
| Weather Data | OpenWeather API |
| Routing | OpenRouteService API |
| Maps | Folium |
| Data Processing | Pandas, NumPy |

---

## 📁 Project Structure

📂 **Datasets/** — Raw and processed datasets

📂 **Models/** — Trained ML models (.h5 files)
- customer_lstm_model.h5
- retailer_mlp_model.h5
- logistics_mlp_model.h5

📂 **Source Code/**
- app.py — Main Streamlit application

📂 **Poster and Presentation/**

📄 .env — API keys (not tracked by git)

📄 .gitignore

📄 requirements.txt

📄 README.md
---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/LikhithaNeerati/SCRI_Capstone.git
cd SCRI_Capstone
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**
**4. Run the application**
```bash
cd "Source Code"
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 🔍 How It Works

1. 🌦️ User inputs a city name — the app fetches live weather data (temperature, wind speed, rainfall)
2. 🤖 Weather features are fed into a trained LSTM model to generate a delivery risk score (0–100)
3. 💬 GPT-2 generates a natural language recommendation based on the city, product, and risk score
4. 🗺️ For routing tabs, OpenRouteService calculates the optimal driving route displayed on an interactive Folium map

---

## 🧠 Models

| Model | Type | Purpose |
|---|---|---|
| Customer LSTM Model | LSTM | Predicts delivery disruption risk from 7-day weather sequences |
| Retailer MLP Model | MLP | Predicts retailer-side delivery feasibility based on route conditions |
| Logistics MLP Model | MLP | Estimates logistics risk for warehouse-to-destination shipments |

---

## 👩‍💻 Author

**Likhitha Neerati**
MS Data Science, University of Missouri-Kansas City
[🔗 LinkedIn](https://linkedin.com/in/likhitha-neerati-50609a1a6) | 📧 likhithaneerati@gmail.com