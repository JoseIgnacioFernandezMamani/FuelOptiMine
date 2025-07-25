# Installation Guide

This guide provides step-by-step instructions to install and set up FuelOptiMine on your system.

## 🔧 System Requirements

### Minimum Requirements

**Hardware:**
- CPU: 4 cores minimum (8 cores recommended)
- RAM: 8GB minimum (16GB recommended for AI components)
- Storage: 20GB free space (50GB recommended)
- Network: Stable internet connection for initial setup

**Software:**
- Python 3.10 or higher
- Git
- Docker (optional, for containerized deployment)

### Production Requirements

**For Production Environment (VMware/Cloud):**
- CPU: 8 cores minimum
- RAM: 32GB recommended
- Storage: 100GB+ SSD storage
- Operating System: Ubuntu 22.04 LTS or CentOS 8+
- Network: High-speed connection to data sources

## 📦 Installation Steps

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/JoseIgnacioFernandezMamani/FuelOptiMine.git
cd FuelOptiMine
```

### 2. Set Up Python Environment

**Using venv (Recommended):**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

**Using conda (Alternative):**
```bash
# Create conda environment
conda create -n fueloptimine python=3.10
conda activate fueloptimine
```

### 3. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import django, pandas, streamlit; print('All dependencies installed successfully')"
```

### 4. Database Setup

**ClickHouse Database:**
```bash
# Option 1: Using Docker
docker run -d --name clickhouse-server \
  -p 8123:8123 -p 9000:9000 \
  --ulimit nofile=262144:262144 \
  clickhouse/clickhouse-server

# Option 2: Install directly (Ubuntu)
sudo apt-get install -y apt-transport-https ca-certificates dirmngr
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv E0C56BD4
echo "deb https://repo.clickhouse.tech/deb/stable/ main/" | sudo tee /etc/apt/sources.list.d/clickhouse.list
sudo apt-get update
sudo apt-get install -y clickhouse-server clickhouse-client
```

**Django Database:**
```bash
# Run Django migrations
cd backend
python manage.py migrate
python manage.py collectstatic --noinput
```

### 5. Environment Configuration

Create a `.env` file in the project root:

```bash
# Copy example environment file
cp .env.example .env
```

Edit `.env` with your configuration:

```ini
# Database Configuration
CLICKHOUSE_HOST=localhost
CLICKHOUSE_NATIVE_PORT=9000
CLICKHOUSE_HTTP_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=
CLICKHOUSE_DB=fuel_optimine

# Django Configuration
DJANGO_SECRET_KEY=your-secret-key-here
DJANGO_DEBUG=False
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit Configuration
STREAMLIT_PORT=8501
STREAMLIT_HOST=0.0.0.0

# ML Model Configuration
MODEL_PATH=./models/
ENABLE_AI_FEATURES=true
```

### 6. Initialize Database Schema

```bash
# Create ClickHouse database and tables
python scripts/init_database.py

# Load sample data (optional)
python scripts/load_sample_data.py
```

## 🚀 Running the Application

### Development Mode

**Start Backend (Django):**
```bash
cd backend
python manage.py runserver 0.0.0.0:8000
```

**Start Frontend (Streamlit):**
```bash
# In a new terminal
streamlit run tests/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

**Start Orchestration (Kedro):**
```bash
cd core/orchestration
kedro run
```

### Production Mode

```bash
# Use the provided startup script
chmod +x scripts/start_production.sh
./scripts/start_production.sh
```

## 🔍 Verification

### Check Installation

1. **Database Connection:**
```bash
python -c "
import clickhouse_connect
client = clickhouse_connect.get_client(host='localhost', port=8123)
print('ClickHouse connection:', client.ping())
"
```

2. **Django Backend:**
```bash
curl http://localhost:8000/admin/
# Should return Django admin login page
```

3. **Streamlit Frontend:**
```bash
curl http://localhost:8501/
# Should return Streamlit application
```

### Access the Application

- **Frontend Dashboard:** http://localhost:8501
- **Backend Admin:** http://localhost:8000/admin/
- **API Documentation:** http://localhost:8000/api/docs/

## 🐳 Docker Installation (Alternative)

For containerized deployment:

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check container status
docker-compose ps

# View logs
docker-compose logs -f
```

## 🔧 Common Installation Issues

### Python Dependencies
```bash
# If you encounter dependency conflicts
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### ClickHouse Connection Issues
```bash
# Check ClickHouse service status
sudo systemctl status clickhouse-server

# Restart if needed
sudo systemctl restart clickhouse-server
```

### Port Conflicts
```bash
# Check if ports are in use
netstat -tlnp | grep :8501
netstat -tlnp | grep :8000

# Kill processes if needed
sudo fuser -k 8501/tcp
sudo fuser -k 8000/tcp
```

## 🔄 Next Steps

After successful installation:

1. **Configuration**: Review the [Configuration Guide](configuration.md)
2. **User Guide**: Learn how to use the system with the [User Guide](user-guide.md)
3. **Development**: Set up development environment with [Developer Guide](developer-guide.md)

## 🆘 Getting Help

If you encounter issues during installation:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Review system logs: `journalctl -u fueloptimine -f`
3. Open an issue on GitHub with detailed error information
4. Contact the development team

## 📋 Installation Checklist

- [ ] System requirements verified
- [ ] Repository cloned
- [ ] Python environment created and activated
- [ ] Dependencies installed successfully
- [ ] Database configured and initialized
- [ ] Environment variables configured
- [ ] Application starts without errors
- [ ] All services accessible via web browser
- [ ] Sample data loaded (optional)
- [ ] System verification completed