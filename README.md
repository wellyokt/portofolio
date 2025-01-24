# Ads Click Prediction 🏠

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
boston_house_price/
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
git clone https://github.com/bayuzen19/boston_house_price.git
cd boston_house_price
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

### Docker Setup

#### Building Individual Images

1. FastAPI Image:
```bash
# Build FastAPI image
docker build -t ad-fastapi:latest -f Dockerfile.fastapi .

# Run FastAPI container
docker run -d -p 8000:8000 --name ad-fastapi ad-fastapi:latest
```

2. Streamlit Image:
```bash
# Build Streamlit image
docker build -t ad-streamlit:latest -f Dockerfile.streamlit .

# Run Streamlit container
docker run -d -p 8501:8501 --name ad-streamlit ad-streamlit:latest
```

3. Running with Network:
```bash
# Create network
docker network create ad-network

# Run FastAPI with network
docker run -d -p 8000:8000 --name ad-fastapi --network ad-network ad-fastapi:latest

# Run Streamlit with network
docker run -d -p 8501:8501 --name ad-streamlit --network ad-network ad-streamlit:latest
```

### Container Management
```bash
# List containers
docker ps

# Stop containers
docker stop ad-fastapi ad-streamlit

# Remove containers
docker rm ad-fastapi ad-streamlit
```

### Image Management
```bash
# List images
docker images

# Remove images
docker rmi ad-fastapi:latest ad-streamlit:latest
```

### Logs and Debugging
```bash
# View logs
docker logs ad-fastapi
docker logs ad-streamlit

# Follow logs
docker logs -f ad-fastapi
```

## Troubleshooting

### Common Issues

1. Port Conflicts
```bash
# Check port usage
lsof -i :8000
lsof -i :8501

# Use alternative ports
docker run -d -p 8001:8000 ad-fastapi:latest
```

2. Permission Issues
```bash
# Run with sudo (Linux)
sudo docker-compose up

# Add user to docker group
sudo usermod -aG docker $USER
```

3. Memory Issues
```bash
# View resource usage
docker stats

# Set memory limits
docker run -d -p 8000:8000 --memory=1g ad-fastapi:latest
```
