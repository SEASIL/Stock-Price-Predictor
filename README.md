# ⬡ Stock-Price-Predictor

### Deep Learning LSTM · Cinematic Dark UI · Neon Analytics

**Nexus Alpha** is a high-performance financial analytics dashboard that leverages **Long Short-Term Memory (LSTM)** neural networks to predict future stock prices based on historical market data. Built with a focus on both technical accuracy and "Glassmorphism" aesthetics, it provides a professional-grade tool for time-series forecasting.

---

## 🚀 Live Deployment
**[View the Live App Here](INSERT_YOUR_STREAMLIT_URL_HERE)**

---

## 💎 Key Features
* **Deep Learning Engine**: Utilizes a multi-layered LSTM architecture built on **TensorFlow/Keras** to capture non-linear market trends.
* **Cinematic UI**: A custom-coded "Void" dark theme featuring neon cyan and purple accents for a high-end fintech experience.
* **Interactive Analytics**: Real-time candlestick charts and technical indicators (Moving Averages, RSI) powered by **Plotly**.
* **Model Validation**: Transparent performance metrics including **RMSE** (Root Mean Squared Error) and **MAPE** to evaluate prediction confidence.
* **Automated Pipeline**: Real-time data fetching via `yfinance` with an iterative 15-day future forecast.

---

## 🛠️ Technical Stack
* **Frontend**: Streamlit (Python-based Web Framework)
* **AI/ML**: TensorFlow 2.15+, Keras, Scikit-Learn
* **Data Science**: Pandas, NumPy, yfinance
* **Visualization**: Plotly Express, Plotly Graph Objects

---

## 🧠 Model Architecture
The core of Nexus Alpha is a Recurrent Neural Network (RNN) optimized for time-series data:
1.  **Input Layer**: Accepts a 60-day lookback window of scaled price data.
2.  **LSTM Layer 1**: 128 units with Dropout (0.2) and Batch Normalization for stability.
3.  **LSTM Layer 2**: 64 units to refine pattern recognition.
4.  **Dense Layers**: A 32-unit ReLU layer followed by a single linear output for price prediction.
5.  **Optimizer**: Adam (Learning Rate: 0.001) | **Loss Function**: Mean Squared Error (MSE).

---

## 🖥️ Local Setup
To run this project on your machine (Recommended: Python 3.12):

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/nexus-alpha-stock-ai.git](https://github.com/YOUR_USERNAME/nexus-alpha-stock-ai.git)
   cd nexus-alpha-stock-ai
