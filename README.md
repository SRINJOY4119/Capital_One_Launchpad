# PRAGATI - Agricultural AI Platform 🌾🤖

**Precision Retrieval & AI for Generative Agriculture Technology & Insights**

A comprehensive RAG-powered multi-agent intelligence system tailored for agricultural domain knowledge. PRAGATI integrates semantic search with domain-specific knowledge base retrieval, ensuring all AI-generated responses are grounded in verified agricultural data.

## 🌟 Key Features

- **🤖 Multi-Agent AI System** with specialized agents for different agricultural domains
- **🔍 RAG-Powered Intelligence** for context-aware, grounded responses
- **🌍 Multi-lingual & Multi-modal** support (text, image, voice inputs)
- **🔬 Deep Research Pipelines** for enhanced coverage and analysis
- **👥 Human-in-the-Loop (HITL)** mechanism for validation and trustworthiness
- **📊 Real-time Market Intelligence** and weather forecasting
- **🌱 Precision Agriculture** recommendations and insights

## 🚀 Quick Start with Docker

### Prerequisites

- Docker installed on your system
- Docker Compose (usually comes with Docker Desktop)
- Git for cloning the repository

### 1. Clone the Repository

```bash
git clone https://github.com/SRINJOY59/Capital_One_Launchpad.git
cd Capital_One_Launchpad
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory with your API keys:

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

Required API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
PINECONE_API_KEY=your_pinecone_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
SERPER_API_KEY=your_serper_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_WEATHER_API_KEY=your_google_weather_api_key_here
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here
```

### 3. Build and Run with Docker Compose

```bash
# Build and start the application
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 4. Access the Application

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Root Endpoint**: http://localhost:8000/

## 🛠️ Manual Docker Build (Alternative)

If you prefer to build and run the Docker container manually:

```bash
# Build the Docker image
docker build -t pragati .

# Run the container
docker run -p 8000:8000 \
  --env-file .env \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/google-creds.json \
  -v $(pwd)/google-creds.json:/app/google-creds.json \
  pragati
```

## 📡 API Endpoints

### Core Features

- **🌾 Crop Recommendations**: `/api/v1/crop-recommendation`
- **🦠 Disease Detection**: `/api/v1/crop-disease/detect`
- **🌤️ Weather Forecasting**: `/api/v1/weather/forecast-tool`
- **📈 Market Prices**: `/api/v1/market-price`
- **🔬 Pest Prediction**: `/api/v1/pest-prediction`
- **🌱 Crop Yield Prediction**: `/api/v1/crop-yield/predict`
- **💧 Fertilizer Recommendations**: `/api/v1/fertilizer/recommendation`

### AI Agents

- **🌐 Multilingual Support**: `/api/v1/agent/multilingual`
- **⚠️ Risk Management**: `/api/v1/agent/risk-management`
- **📰 News Analysis**: `/api/v1/agent/agri-news`
- **💰 Credit Policy & Market**: `/api/v1/agent/creditpolicy`
- **🔍 Deep Research**: `/api/v1/agent/deep-research`

### Utilities

- **🌍 Translation**: `/api/v1/translate/text`
- **🕷️ Web Scraping**: `/api/v1/webscrapper/agri-prices`
- **📍 Location Info**: `/api/v1/agent/location-information`

## 🧪 Testing the API

### Health Check

```bash
curl http://localhost:8000/health
```

### Crop Recommendation Example

```bash
curl -X POST "http://localhost:8000/api/v1/crop-recommendation" \
-H "Content-Type: application/json" \
-d '{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9,
  "model_type": "stacked"
}'
```

### Disease Detection Example

```bash
curl -X POST "http://localhost:8000/api/v1/crop-disease/detect" \
-H "Content-Type: multipart/form-data" \
-F "file=@path/to/your/crop_image.jpg"
```

## 🔧 Development Setup

For local development without Docker:

### Prerequisites

- Python 3.11+
- Virtual environment tool (venv, conda, etc.)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the application
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## 🏗️ Project Structure

```
Capital_One_Launchpad/
├── Agents/                     # AI Agent modules
│   ├── Crop_Disease/          # Disease detection agent
│   ├── Weather_forcast/       # Weather forecasting agent
│   ├── Multi_Lingual/         # Translation agent
│   └── ...                    # Other specialized agents
├── Tools/                     # Utility tools and functions
├── Models/                    # ML model files
├── Dataset/                   # Training and reference datasets
├── Notebooks/                 # Jupyter notebooks for analysis
├── Deep_Research/             # Research and analysis tools
├── RAG/                       # Retrieval-Augmented Generation
├── uploads/                   # File upload directory
├── app.py                     # Main FastAPI application
├── workflow.py                # Core workflow logic
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
└── README.md                  # This file
```

## 🚦 Monitoring & Health Checks

The application includes built-in monitoring:

- **Health Endpoint**: `/health` - Returns application status
- **Docker Health Checks**: Automatic container health monitoring
- **Logging**: Comprehensive logging for debugging and monitoring

## 🐛 Troubleshooting

### Common Issues

1. **Port 8000 already in use**

   ```bash
   # Change port in docker-compose.yml or kill existing process
   lsof -i :8000
   kill -9 <PID>
   ```

2. **API Key errors**

   - Verify all required API keys are set in `.env`
   - Check API key validity and quotas

3. **Model loading issues**

   - Ensure sufficient memory (8GB+ recommended)
   - Check internet connection for model downloads

4. **Docker build fails**
   - Clear Docker cache: `docker system prune -a`
   - Ensure sufficient disk space

### Logs

```bash
# View application logs
docker-compose logs -f agricultural-ai

# View specific service logs
docker logs agricultural-ai-app
```

## 🏛️ PRAGATI Architecture

### RAG-Powered Intelligence

- **Semantic Search**: Advanced retrieval from agricultural knowledge bases
- **Query Complexity Assessment**: Optimized retrieval and generation
- **Domain-Specific Knowledge**: Grounded in verified agricultural data

### Multi-Agent Orchestration

- **Specialized Agents**: Crop selection, irrigation, fertilizer, pest detection
- **Weather Forecasting**: Real-time meteorological analysis
- **Market Analysis**: Price trends and supply chain insights
- **Risk Assessment**: Agricultural decision support

### Multi-Modal Capabilities

- **Text Processing**: Natural language queries and responses
- **Image Analysis**: Crop disease detection and visual diagnostics
- **Voice Input**: Accessibility for diverse user base

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Agricultural research institutions for datasets
- Open source ML community
- FastAPI and modern Python ecosystem
- Capital One Launchpad Program

---

**Made with ❤️ for sustainable agriculture and AI innovation**
