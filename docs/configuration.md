# Configuration Guide

This guide provides detailed information on configuring FuelOptiMine for different environments and use cases.

## 🛠️ Environment Configuration

### Environment Variables

FuelOptiMine uses environment variables for configuration management. Create a `.env` file in the project root:

```bash
# Core Application Settings
ENVIRONMENT=development          # development, staging, production
DEBUG=True                      # Set to False in production
SECRET_KEY=your-secret-key-here # Generate a secure secret key

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/fueloptimine
CLICKHOUSE_HOST=localhost
CLICKHOUSE_HTTP_PORT=8123
CLICKHOUSE_NATIVE_PORT=9000
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=
CLICKHOUSE_DB=fuel_optimine

# Redis Configuration (for caching and Celery)
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com

# Streamlit Configuration
STREAMLIT_HOST=0.0.0.0
STREAMLIT_PORT=8501
STREAMLIT_THEME=dark

# ML Models Configuration
MODEL_STORAGE_PATH=./models/
AUTO_RETRAIN=True
RETRAIN_THRESHOLD=0.85
MODEL_REGISTRY_URL=http://localhost:5000

# Monitoring and Logging
LOG_LEVEL=INFO
SENTRY_DSN=your-sentry-dsn-here
PROMETHEUS_ENABLED=True

# Email Configuration (for reports and alerts)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-email@example.com
EMAIL_HOST_PASSWORD=your-email-password

# External Integrations
WEATHER_API_KEY=your-weather-api-key
FUEL_PRICE_API_URL=https://api.fuelprices.com/v1/
```

### Configuration Files

#### Django Settings

**`backend/config/settings.py`**

```python
import os
from pathlib import Path
from decouple import config, Csv

BASE_DIR = Path(__file__).resolve().parent.parent

# Environment-specific settings
ENVIRONMENT = config('ENVIRONMENT', default='development')
DEBUG = config('DEBUG', default=True, cast=bool)
SECRET_KEY = config('SECRET_KEY', default='dev-key-only')

# Security settings
ALLOWED_HOSTS = config('ALLOWED_HOSTS', default='localhost', cast=Csv())
CORS_ALLOWED_ORIGINS = config('CORS_ALLOWED_ORIGINS', default='http://localhost:8501', cast=Csv())

# Database configuration
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': config('DB_NAME', default='fueloptimine'),
        'USER': config('DB_USER', default='postgres'),
        'PASSWORD': config('DB_PASSWORD', default=''),
        'HOST': config('DB_HOST', default='localhost'),
        'PORT': config('DB_PORT', default='5432'),
    }
}

# ClickHouse configuration
CLICKHOUSE_CONFIG = {
    'host': config('CLICKHOUSE_HOST', default='localhost'),
    'port': config('CLICKHOUSE_HTTP_PORT', default=8123, cast=int),
    'username': config('CLICKHOUSE_USER', default='default'),
    'password': config('CLICKHOUSE_PASSWORD', default=''),
    'database': config('CLICKHOUSE_DB', default='fuel_optimine'),
    'settings': {
        'use_numpy': True,
        'compress': True,
    }
}

# Celery configuration
CELERY_BROKER_URL = config('REDIS_URL', default='redis://localhost:6379/0')
CELERY_RESULT_BACKEND = config('REDIS_URL', default='redis://localhost:6379/0')
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_TIMEZONE = 'UTC'

# ML Model settings
ML_CONFIG = {
    'storage_path': config('MODEL_STORAGE_PATH', default='./models/'),
    'auto_retrain': config('AUTO_RETRAIN', default=True, cast=bool),
    'retrain_threshold': config('RETRAIN_THRESHOLD', default=0.85, cast=float),
    'prediction_cache_ttl': 3600,  # 1 hour
}

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'logs/fueloptimine.log',
            'formatter': 'verbose',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': config('LOG_LEVEL', default='INFO'),
    },
}
```

#### Streamlit Configuration

**`.streamlit/config.toml`**

```toml
[global]
developmentMode = false
logLevel = "info"

[server]
headless = true
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[browser]
gatherUsageStats = false

[mapbox]
token = "your-mapbox-token-here"
```

#### Kedro Configuration

**`core/orchestration/conf/base/catalog.yml`**

```yaml
# Data sources
raw_fuel_data:
  type: pandas.CSVDataSet
  filepath: data/01_raw/fuel_readings.csv

processed_fuel_data:
  type: pandas.ParquetDataSet
  filepath: data/02_intermediate/processed_fuel_data.pq

# ClickHouse connections
clickhouse_fuel_readings:
  type: fueloptimine.extras.datasets.ClickHouseDataSet
  connection:
    host: ${clickhouse.host}
    port: ${clickhouse.port}
    database: ${clickhouse.database}
  table_name: fuel_readings
  load_args:
    query: "SELECT * FROM fuel_readings WHERE timestamp >= yesterday()"

# Model artifacts
fuel_prediction_model:
  type: pickle.PickleDataSet
  filepath: data/06_models/fuel_prediction_model.pkl
  versioned: true

# Model metrics
model_metrics:
  type: json.JSONDataSet
  filepath: data/08_reporting/model_metrics.json
```

**`core/orchestration/conf/base/parameters.yml`**

```yaml
# Data processing parameters
data_processing:
  outlier_threshold: 3.0
  missing_value_strategy: "interpolate"
  time_window_hours: 24
  min_records_per_equipment: 100

# Feature engineering parameters
feature_engineering:
  temporal_features:
    - hour_of_day
    - day_of_week
    - is_weekend
    - is_holiday
  
  rolling_window_sizes: [6, 12, 24]  # hours
  
  equipment_features:
    - fuel_capacity
    - engine_power
    - max_payload
    - age_years

# Model training parameters
model_training:
  test_size: 0.2
  validation_size: 0.1
  random_state: 42
  
  xgboost:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    random_state: 42
  
  hyperparameter_tuning:
    enabled: true
    n_trials: 100
    timeout: 3600  # 1 hour

# Prediction parameters
prediction:
  horizon_hours: 24
  confidence_level: 0.95
  update_frequency: 300  # 5 minutes
```

## 🗄️ Database Configuration

### ClickHouse Setup

#### Database Schema

```sql
-- Create database
CREATE DATABASE IF NOT EXISTS fuel_optimine;

-- Use the database
USE fuel_optimine;

-- Create main fuel readings table
CREATE TABLE fuel_readings (
    timestamp DateTime64(3),
    equipment_id String,
    fuel_level_liters Float64,
    location_lat Float64,
    location_lon Float64,
    engine_status Enum8('running' = 1, 'idle' = 2, 'off' = 3),
    operator_id String,
    temperature Float32,
    engine_hours Float64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (equipment_id, timestamp)
SETTINGS index_granularity = 8192;

-- Create equipment metadata table
CREATE TABLE equipment (
    equipment_id String,
    name String,
    equipment_type String,
    fuel_capacity Float64,
    max_payload Float64,
    engine_power Float64,
    manufacture_year UInt16,
    is_active UInt8 DEFAULT 1,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
ORDER BY equipment_id;

-- Create predictions table
CREATE TABLE fuel_predictions (
    prediction_id String,
    equipment_id String,
    prediction_timestamp DateTime,
    target_timestamp DateTime,
    predicted_consumption Float64,
    confidence_lower Float64,
    confidence_upper Float64,
    model_version String,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(prediction_timestamp)
ORDER BY (equipment_id, prediction_timestamp, target_timestamp);

-- Create materialized views for real-time aggregations
CREATE MATERIALIZED VIEW fuel_hourly_mv TO fuel_hourly_aggregates AS
SELECT 
    toStartOfHour(timestamp) as hour,
    equipment_id,
    avg(fuel_level_liters) as avg_fuel_level,
    min(fuel_level_liters) as min_fuel_level,
    max(fuel_level_liters) as max_fuel_level,
    count() as reading_count
FROM fuel_readings
GROUP BY hour, equipment_id;

CREATE TABLE fuel_hourly_aggregates (
    hour DateTime,
    equipment_id String,
    avg_fuel_level Float64,
    min_fuel_level Float64,
    max_fuel_level Float64,
    reading_count UInt64
) ENGINE = SummingMergeTree()
ORDER BY (hour, equipment_id);
```

#### ClickHouse Configuration File

**`/etc/clickhouse-server/config.xml`** (key sections):

```xml
<clickhouse>
    <logger>
        <level>information</level>
        <log>/var/log/clickhouse-server/clickhouse-server.log</log>
        <errorlog>/var/log/clickhouse-server/clickhouse-server.err.log</errorlog>
        <size>1000M</size>
        <count>10</count>
    </logger>

    <http_port>8123</http_port>
    <tcp_port>9000</tcp_port>
    
    <listen_host>0.0.0.0</listen_host>
    
    <max_connections>4096</max_connections>
    <keep_alive_timeout>3</keep_alive_timeout>
    <max_concurrent_queries>100</max_concurrent_queries>
    
    <uncompressed_cache_size>8589934592</uncompressed_cache_size>
    <mark_cache_size>5368709120</mark_cache_size>
    
    <path>/var/lib/clickhouse/</path>
    <tmp_path>/var/lib/clickhouse/tmp/</tmp_path>
    <user_files_path>/var/lib/clickhouse/user_files/</user_files_path>
    
    <users_config>users.xml</users_config>
    
    <default_profile>default</default_profile>
    <default_database>fuel_optimine</default_database>
    
    <timezone>UTC</timezone>
</clickhouse>
```

### PostgreSQL Configuration

#### Django Models Configuration

```python
# backend/analytics/models.py
from django.db import models
from django.contrib.auth.models import User

class Equipment(models.Model):
    """Equipment metadata model."""
    equipment_id = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=200)
    equipment_type = models.CharField(max_length=50, choices=[
        ('haul_truck', 'Haul Truck'),
        ('excavator', 'Excavator'),
        ('loader', 'Wheel Loader'),
        ('dozer', 'Bulldozer'),
        ('grader', 'Motor Grader'),
    ])
    fuel_capacity = models.FloatField()
    max_payload = models.FloatField()
    engine_power = models.FloatField()
    manufacture_year = models.IntegerField()
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'equipment'
        indexes = [
            models.Index(fields=['equipment_type']),
            models.Index(fields=['is_active']),
        ]

class UserProfile(models.Model):
    """Extended user profile."""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=[
        ('viewer', 'Viewer'),
        ('operator', 'Operator'),
        ('analyst', 'Analyst'),
        ('admin', 'Administrator'),
    ])
    department = models.CharField(max_length=50)
    phone = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class PredictionModel(models.Model):
    """ML model registry."""
    name = models.CharField(max_length=100)
    version = models.CharField(max_length=20)
    model_type = models.CharField(max_length=50)
    accuracy_score = models.FloatField()
    training_date = models.DateTimeField()
    is_active = models.BooleanField(default=False)
    model_file_path = models.CharField(max_length=500)
    hyperparameters = models.JSONField()
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)

    class Meta:
        unique_together = ['name', 'version']
```

## 🔧 Application Configuration

### API Configuration

#### Django REST Framework Settings

```python
# REST Framework configuration
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
    ],
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
        'rest_framework.filters.SearchFilter',
        'rest_framework.filters.OrderingFilter',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 100,
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle'
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': '100/hour',
        'user': '1000/hour'
    },
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
}

# JWT Configuration
from datetime import timedelta

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(hours=1),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
    'BLACKLIST_AFTER_ROTATION': True,
    'ALGORITHM': 'HS256',
    'SIGNING_KEY': SECRET_KEY,
    'AUTH_HEADER_TYPES': ('Bearer',),
}
```

### Monitoring Configuration

#### Prometheus Metrics

```python
# backend/config/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, Info

# Application metrics
REQUEST_COUNT = Counter(
    'fueloptimine_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'fueloptimine_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)

FUEL_READINGS_PROCESSED = Counter(
    'fueloptimine_fuel_readings_processed_total',
    'Total fuel readings processed',
    ['equipment_type']
)

ACTIVE_EQUIPMENT = Gauge(
    'fueloptimine_active_equipment',
    'Number of active equipment',
    ['equipment_type']
)

MODEL_PREDICTION_ACCURACY = Gauge(
    'fueloptimine_model_accuracy',
    'Model prediction accuracy',
    ['model_name', 'model_version']
)

APPLICATION_INFO = Info(
    'fueloptimine_application',
    'Application information'
)

# Set application info
APPLICATION_INFO.info({
    'version': '1.0.0',
    'environment': ENVIRONMENT,
    'python_version': '3.10'
})
```

#### Logging Configuration

```python
# Enhanced logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'json': {
            'format': '{"level": "%(levelname)s", "time": "%(asctime)s", "module": "%(module)s", "message": "%(message)s"}',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/fueloptimine.log',
            'maxBytes': 1024*1024*10,  # 10MB
            'backupCount': 5,
            'formatter': 'verbose',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'json_file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/fueloptimine.json',
            'maxBytes': 1024*1024*10,
            'backupCount': 5,
            'formatter': 'json',
        },
    },
    'loggers': {
        'fueloptimine': {
            'handlers': ['file', 'console', 'json_file'],
            'level': 'INFO',
            'propagate': False,
        },
        'analytics': {
            'handlers': ['file'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'etl_core': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
    'root': {
        'handlers': ['console'],
        'level': config('LOG_LEVEL', default='INFO'),
    },
}
```

## 🔐 Security Configuration

### Authentication Settings

```python
# Security configuration
SECURITY_CONFIG = {
    'password_min_length': 8,
    'password_require_uppercase': True,
    'password_require_lowercase': True,
    'password_require_numbers': True,
    'password_require_special': True,
    'session_timeout': 3600,  # 1 hour
    'failed_login_attempts': 5,
    'lockout_duration': 900,  # 15 minutes
}

# IP Whitelist
ALLOWED_IPS = config('ALLOWED_IPS', default='', cast=Csv())

# CORS settings
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = config('CORS_ALLOWED_ORIGINS', default='http://localhost:8501', cast=Csv())

# CSRF settings
CSRF_TRUSTED_ORIGINS = config('CSRF_TRUSTED_ORIGINS', default='http://localhost:8501', cast=Csv())
```

### Data Encryption

```python
# Encryption configuration
ENCRYPTION_CONFIG = {
    'algorithm': 'AES-256-GCM',
    'key_rotation_days': 90,
    'encrypted_fields': [
        'location_data',
        'operator_id',
        'sensitive_measurements'
    ]
}

# Field-level encryption
FIELD_ENCRYPTION_KEY = config('FIELD_ENCRYPTION_KEY')
```

## 📊 Performance Configuration

### Database Optimization

#### ClickHouse Performance Settings

```xml
<!-- Performance settings for ClickHouse -->
<profiles>
    <default>
        <max_memory_usage>10000000000</max_memory_usage>
        <use_uncompressed_cache>1</use_uncompressed_cache>
        <load_balancing>random</load_balancing>
        <max_threads>8</max_threads>
        <max_execution_time>300</max_execution_time>
        <min_count_to_compile>3</min_count_to_compile>
        <group_by_two_level_threshold>100000</group_by_two_level_threshold>
        <group_by_two_level_threshold_bytes>50000000</group_by_two_level_threshold_bytes>
        <distributed_aggregation_memory_efficient>1</distributed_aggregation_memory_efficient>
        <aggregation_memory_efficient_merge_threads>4</aggregation_memory_efficient_merge_threads>
    </default>
    
    <readonly>
        <readonly>1</readonly>
        <max_memory_usage>5000000000</max_memory_usage>
        <max_execution_time>60</max_execution_time>
    </readonly>
</profiles>
```

#### Django Database Optimization

```python
# Database optimization settings
DATABASES['default']['OPTIONS'] = {
    'MAX_CONNS': 20,
    'OPTIONS': {
        'MAX_CONNS': 20,
        'charset': 'utf8mb4',
        'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
    }
}

# Connection pooling
DATABASE_CONNECTION_POOLING = {
    'enabled': True,
    'max_connections': 20,
    'min_connections': 5,
    'connection_timeout': 30,
}

# Query optimization
DATABASE_QUERY_OPTIMIZATION = {
    'enable_query_cache': True,
    'cache_timeout': 300,
    'slow_query_threshold': 1.0,  # seconds
    'log_slow_queries': True,
}
```

### Caching Configuration

```python
# Redis caching configuration
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': config('REDIS_URL', default='redis://localhost:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 50,
                'retry_on_timeout': True,
            },
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
            'SERIALIZER': 'django_redis.serializers.json.JSONSerializer',
        },
        'TIMEOUT': 300,  # 5 minutes
        'KEY_PREFIX': 'fueloptimine',
        'VERSION': 1,
    },
    'sessions': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': config('REDIS_URL', default='redis://localhost:6379/2'),
        'TIMEOUT': 3600,  # 1 hour
    }
}

# Session configuration
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'sessions'
SESSION_COOKIE_AGE = 3600  # 1 hour
```

## 📱 Frontend Configuration

### Streamlit Advanced Configuration

```python
# streamlit_config.py
import streamlit as st

def configure_streamlit():
    """Configure Streamlit application settings."""
    
    # Page configuration
    st.set_page_config(
        page_title="FuelOptiMine Dashboard",
        page_icon="⛽",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://docs.fueloptimine.com',
            'Report a bug': 'https://github.com/fueloptimine/issues',
            'About': "FuelOptiMine - Intelligent Mining Optimization"
        }
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #ff6b6b;
        }
        
        .alert-critical {
            background-color: #ffe6e6;
            border-left: 4px solid #ff4444;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .alert-warning {
            background-color: #fff8e1;
            border-left: 4px solid #ffb300;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

# Theme configuration
STREAMLIT_THEME = {
    'light': {
        'primary_color': '#FF6B6B',
        'background_color': '#FFFFFF',
        'secondary_background_color': '#F0F2F6',
        'text_color': '#262730',
    },
    'dark': {
        'primary_color': '#FF6B6B',
        'background_color': '#0E1117',
        'secondary_background_color': '#262730',
        'text_color': '#FAFAFA',
    }
}
```

## 🔧 Development vs Production

### Development Configuration

```bash
# .env.development
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=DEBUG

# Use local databases
DATABASE_URL=postgresql://postgres:password@localhost:5432/fueloptimine_dev
CLICKHOUSE_HOST=localhost
REDIS_URL=redis://localhost:6379/0

# Disable external services
EMAIL_BACKEND=django.core.mail.backends.console.EmailBackend
SENTRY_DSN=

# Enable development tools
ENABLE_DEBUG_TOOLBAR=True
ENABLE_SILK_PROFILING=True
```

### Production Configuration

```bash
# .env.production
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=INFO

# Production databases
DATABASE_URL=postgresql://user:password@prod-db:5432/fueloptimine
CLICKHOUSE_HOST=prod-clickhouse
REDIS_URL=redis://prod-redis:6379/0

# Security settings
SECRET_KEY=your-production-secret-key
ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com
CORS_ALLOWED_ORIGINS=https://yourdomain.com

# External services
EMAIL_BACKEND=django.core.mail.backends.smtp.EmailBackend
SENTRY_DSN=your-production-sentry-dsn

# Performance settings
ENABLE_QUERY_CACHE=True
COMPRESS_ENABLED=True
STATICFILES_STORAGE=whitenoise.storage.CompressedManifestStaticFilesStorage
```

## 🔄 Configuration Management Tools

### Using Environment Files

```bash
# Load configuration based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    source .env.production
elif [ "$ENVIRONMENT" = "staging" ]; then
    source .env.staging
else
    source .env.development
fi
```

### Configuration Validation

```python
# config_validator.py
from decouple import config
import sys

def validate_configuration():
    """Validate required configuration settings."""
    
    required_settings = [
        'SECRET_KEY',
        'DATABASE_URL',
        'CLICKHOUSE_HOST',
        'REDIS_URL',
    ]
    
    missing_settings = []
    
    for setting in required_settings:
        try:
            value = config(setting)
            if not value:
                missing_settings.append(setting)
        except Exception:
            missing_settings.append(setting)
    
    if missing_settings:
        print(f"Missing required configuration: {', '.join(missing_settings)}")
        sys.exit(1)
    
    print("Configuration validation passed!")

if __name__ == "__main__":
    validate_configuration()
```

This configuration guide provides comprehensive coverage of all configuration aspects for FuelOptiMine, enabling proper setup for different environments and use cases.