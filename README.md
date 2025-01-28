postingan linkedin :
https://www.linkedin.com/posts/wellyoktariana_im-excited-to-share-my-latest-project-a-activity-7288632679695032320-rzA3?utm_source=share&utm_medium=member_desktop

# Ads Click Prediction 

## Project Overview

This project demonstrates an end-to-end machine learning application designed to predict ad clicks. It combines FastAPI for the backend service and Streamlit for an interactive frontend interface. The system is capable of processing user inputs, running a machine learning model for prediction, and presenting the results in real-time, all through a seamless web interface.


## Features
- 🔍 Interactive data exploration and visualization
- 📊 Comprehensive model analytics and performance metrics
- 🤖 Real-time price predictions using Logistic Regression
- 📈 Feature importance analysis
- 🎯 Model performance tracking
- 🖥️ User-friendly web interface

## Technology Stack
- **Backend**: FastAPI, Python 3.9
- **Frontend**: Streamlit
- **ML Framework**: Scikit-learn, Logistic Regression
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Containerization**: Docker
- **Version Control**: Git
- **Development**: VS Code

## Project Structure
```plaintext
portofolio/
│
├── .streamlit/                      # Streamlit configuration
│   └── config.toml                  # Streamlit settings
│
├── artifacts/                       # Model artifacts
│   ├── boston.csv                   # Dataset
│   ├── best_model.pkl              # Trained model
│   ├── scaler.pkl                  # Fitted scaler
│   └── metrics.json                # Model metrics
│
├── config/                          # Configuration
│   ├── __init__.py
│   └── config.py                   # Project configuration
│
├── logs/                           # Application logs
│   └── app.log
│
├── notebooks 
│   ├── research.ipynb
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── data_preparation.py         # Data preprocessing
│   ├── model.py                    # Model training
│   └── evaluation.py               # Model evaluation

│
├── static/                         # Static files
│   ├── css/
│   │   └── style.css              # Custom styling
│   └── img/
│       ├── profile.jpg            # Profile image
│       └── project.png            # Project diagram
│
├── utils/                          # Utilities
│   ├── __init__.py
│   ├── logger.py                   # Logging setup
│   └── styling.py                  # Styling utilities
│
├── app.py                          # FastAPI application
├── Home.py                         # Streamlit main page
├── Dockerfile.fastapi              # FastAPI Dockerfile
├── Dockerfile.streamlit            # Streamlit Dockerfile
├── docker-compose.yml              # Docker composition
├── requirements.txt                # Dependencies
└── README.md                       # Documentation
```

## Installation and Setup

### Prerequisites
- Python 3.9 or higher
- Docker and Docker Compose
- Git

### Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/wellyokt/portofolio.git
cd portofolio
```

2. Create and activate virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Windows
.\venv\Scripts\activate
# For Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model:
```bash
python train.py
```

5. Run the applications:
```bash
# Terminal 1 - Run FastAPI
uvicorn app:app --reload --port 8000

# Terminal 2 - Run Streamlit
streamlit run Home.py
```