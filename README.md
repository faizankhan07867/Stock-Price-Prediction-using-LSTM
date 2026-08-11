# Stock-Price-Prediction-using-LSTM

# 📈 Stock Price Prediction using LSTM

## 📌 Overview

Stock Price Prediction is a Deep Learning project that predicts future stock closing prices using historical market data. The project uses an LSTM (Long Short-Term Memory) neural network to analyze stock trends and provides predictions through a Flask web application.

---

## ✨ Features

- 📈 Stock Price Forecasting
- 🤖 LSTM Deep Learning Model
- 📊 Historical Data Analysis
- 📉 Next Day Price Prediction
- 🌐 Flask Web Application
- 📈 Price Trend Visualization

---

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- Flask
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- HTML
- CSS

---

## 📂 Dataset

Dataset Name:

Stock Market Dataset

Features:

- Date
- Open
- High
- Low
- Close
- Volume

Output:

- Predicted Closing Price

---

## 📁 Project Structure

```
Stock-Price-Prediction/

│── app.py
│── train.py
│── predict.py
│── preprocess.py
│── requirements.txt
│── README.md

├── model/
│     lstm_stock_model.keras
│     scaler.pkl

├── dataset/
│     stock_data.csv

├── templates/
│     index.html

├── static/
│     style.css

├── uploads/

└── screenshots/
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Train Model

```bash
python preprocess.py
python train.py
```

---

## ▶️ Run Project

```bash
python app.py
```

Open Browser

```
http://127.0.0.1:5000
```

---

## 📊 Output

- Enter Historical Stock Prices
- Click **Predict**
- AI Forecasts Next Closing Price
- Displays Predicted Price
- Shows Trend Analysis

---

## 🎯 Future Improvements

- Live Stock API Integration
- Multi-Stock Prediction
- Candlestick Charts
- Buy/Sell Signal Prediction
- Mobile Application
- Cloud Deployment

---

## 👨‍💻 Author

**Faizan Khan**

B.Tech Information Technology

AI | Machine Learning | Data Science | Analytics

---

## 📜 License

MIT License
