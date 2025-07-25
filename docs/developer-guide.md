# Developer Guide

This guide provides comprehensive information for developers working on the FuelOptiMine project.

## 🛠️ Development Environment Setup

### Prerequisites

- Python 3.10+
- Git
- VS Code or PyCharm (recommended)
- Docker (optional)
- ClickHouse database

### Clone and Setup

```bash
# Clone the repository
git clone https://github.com/JoseIgnacioFernandezMamani/FuelOptiMine.git
cd FuelOptiMine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### IDE Configuration

**VS Code Extensions:**
- Python
- Django
- Jupyter
- GitLens
- Black Formatter
- Pylint

**PyCharm Configuration:**
- Set Python interpreter to `.venv/bin/python`
- Enable Django support
- Configure Black as code formatter

## 🏗️ Project Architecture

### High-Level Architecture

```
FuelOptiMine/
├── backend/          # Django backend API
├── frontend/         # Streamlit dashboard (legacy)
├── etl_core/        # Data processing pipeline
├── analytics/       # Real-time analytics
├── core/           # Orchestration with Kedro
├── docs/           # Documentation
└── tests/          # Test suite
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend API** | Django + DRF | REST API and business logic |
| **Database** | ClickHouse | Time-series data storage |
| **Frontend** | Streamlit | Dashboard and visualization |
| **Data Processing** | Pandas, Polars | ETL operations |
| **Orchestration** | Kedro | Pipeline management |
| **Analytics** | NumPy, SciPy | Mathematical operations |
| **Visualization** | Plotly, Matplotlib | Charts and graphs |

### Design Patterns

The project follows these architectural patterns:

1. **Clean Architecture**: Separation of concerns with clear layers
2. **Repository Pattern**: Data access abstraction
3. **Factory Pattern**: Object creation in ETL components
4. **Strategy Pattern**: Interchangeable algorithms
5. **Observer Pattern**: Real-time data updates

## 🧪 Development Workflow

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and commit
git add .
git commit -m "feat: add new feature description"

# Push to remote
git push origin feature/your-feature-name

# Create pull request on GitHub
```

### Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new analytics endpoint
fix: resolve fuel calculation bug
docs: update API documentation
test: add unit tests for ETL pipeline
refactor: improve data validation logic
```

### Code Quality Standards

**Formatting:**
```bash
# Format code with Black
black .

# Sort imports
isort .

# Check with flake8
flake8 .
```

**Type Hints:**
```python
from typing import List, Dict, Optional

def calculate_fuel_efficiency(
    fuel_data: List[Dict[str, float]], 
    time_window: Optional[int] = 24
) -> Dict[str, float]:
    """Calculate fuel efficiency metrics."""
    pass
```

## 🔧 Backend Development

### Django Project Structure

```
backend/
├── config/           # Django settings
├── analytics/        # Analytics app
├── users/           # User management
├── model/           # ML model integration
└── manage.py        # Django management
```

### Creating New Apps

```bash
cd backend
python manage.py startapp new_app_name
```

### Database Migrations

```bash
# Create migrations
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser
```

### API Development

**Example ViewSet:**
```python
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

class FuelDataViewSet(viewsets.ModelViewSet):
    """Fuel data API endpoints."""
    
    @action(detail=False, methods=['get'])
    def realtime(self, request):
        """Get real-time fuel data."""
        data = self.get_realtime_data()
        return Response(data)
        
    def get_realtime_data(self):
        """Fetch real-time data from ClickHouse."""
        # Implementation here
        pass
```

**URL Configuration:**
```python
from django.urls import path, include
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'fuel-data', FuelDataViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
]
```

## 📊 ETL Core Development

### ETL Architecture

```
etl_core/
├── extract/          # Data extraction
│   ├── interfaces/   # Abstract interfaces
│   ├── implementations/  # Concrete extractors
│   └── factories/    # Extractor factories
├── transform/        # Data transformation
└── load/            # Data loading
```

### Creating New Extractors

```python
from etl_core.extract.interfaces.base_extractor import BaseExtractor

class CSVExtractor(BaseExtractor):
    """Extract data from CSV files."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def extract(self) -> pd.DataFrame:
        """Extract data from CSV file."""
        return pd.read_csv(self.file_path)
    
    def validate(self) -> bool:
        """Validate extracted data."""
        return os.path.exists(self.file_path)
```

### Data Transformation Pipeline

```python
from etl_core.transform.base_transformer import BaseTransformer

class FuelDataTransformer(BaseTransformer):
    """Transform fuel consumption data."""
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations to fuel data."""
        # Clean data
        data = self.clean_outliers(data)
        
        # Calculate metrics
        data['efficiency'] = data['distance'] / data['fuel_consumed']
        
        # Add time features
        data = self.add_time_features(data)
        
        return data
```

## 🎯 Analytics Development

### Real-time Analytics

```python
import polars as pl
from analytics.base import BaseAnalyzer

class FuelEfficiencyAnalyzer(BaseAnalyzer):
    """Analyze fuel efficiency patterns."""
    
    def analyze(self, data: pl.DataFrame) -> Dict:
        """Perform fuel efficiency analysis."""
        return {
            'avg_efficiency': data['efficiency'].mean(),
            'trend': self.calculate_trend(data),
            'anomalies': self.detect_anomalies(data)
        }
```

### Visualization Components

```python
import plotly.graph_objects as go

def create_fuel_trend_chart(data: pd.DataFrame) -> go.Figure:
    """Create fuel consumption trend chart."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['fuel_level'],
        mode='lines+markers',
        name='Fuel Level'
    ))
    
    fig.update_layout(
        title='Fuel Consumption Trend',
        xaxis_title='Time',
        yaxis_title='Fuel Level (L)'
    )
    
    return fig
```

## 🧮 Orchestration with Kedro

### Pipeline Development

```python
from kedro.pipeline import Pipeline, node

def create_fuel_analysis_pipeline():
    """Create fuel analysis pipeline."""
    return Pipeline([
        node(
            func=extract_fuel_data,
            inputs="raw_fuel_data",
            outputs="extracted_fuel_data",
            name="extract_fuel_data"
        ),
        node(
            func=transform_fuel_data,
            inputs="extracted_fuel_data",
            outputs="transformed_fuel_data",
            name="transform_fuel_data"
        ),
        node(
            func=analyze_fuel_efficiency,
            inputs="transformed_fuel_data",
            outputs="fuel_analysis_results",
            name="analyze_fuel_efficiency"
        )
    ])
```

### Running Pipelines

```bash
# Run specific pipeline
kedro run --pipeline fuel_analysis

# Run with parameters
kedro run --params "start_date: 2024-01-01, end_date: 2024-01-31"

# Visualize pipeline
kedro viz
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/             # Unit tests
├── integration/      # Integration tests
├── e2e/             # End-to-end tests
└── fixtures/        # Test data
```

### Writing Unit Tests

```python
import pytest
import pandas as pd
from etl_core.extract.local.csv_extractor import CSVExtractor

class TestCSVExtractor:
    """Test CSV extractor functionality."""
    
    def test_extract_valid_csv(self, sample_csv_file):
        """Test extracting valid CSV file."""
        extractor = CSVExtractor(sample_csv_file)
        result = extractor.extract()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_extract_invalid_file(self):
        """Test extracting non-existent file."""
        extractor = CSVExtractor("nonexistent.csv")
        
        with pytest.raises(FileNotFoundError):
            extractor.extract()
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_extractors.py

# Run with coverage
pytest --cov=etl_core

# Run integration tests
pytest tests/integration/
```

### Test Fixtures

```python
@pytest.fixture
def sample_fuel_data():
    """Provide sample fuel data for testing."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'fuel_level': np.random.uniform(50, 1000, 100),
        'equipment_id': ['TRUCK_001'] * 100
    })
```

## 🔍 Debugging and Monitoring

### Logging Configuration

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def process_fuel_data(data):
    """Process fuel data with logging."""
    logger.info(f"Processing {len(data)} fuel records")
    
    try:
        result = perform_analysis(data)
        logger.info("Analysis completed successfully")
        return result
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise
```

### Performance Monitoring

```python
import time
from functools import wraps

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@timing_decorator
def expensive_calculation(data):
    """Perform expensive calculation with timing."""
    pass
```

## 📚 Documentation

### Code Documentation

```python
def calculate_fuel_efficiency(
    fuel_data: pd.DataFrame,
    distance_data: pd.DataFrame,
    time_window: int = 24
) -> Dict[str, float]:
    """
    Calculate fuel efficiency metrics for mining equipment.
    
    Args:
        fuel_data: DataFrame with fuel consumption data
        distance_data: DataFrame with distance traveled data
        time_window: Time window in hours for calculation
        
    Returns:
        Dictionary containing efficiency metrics:
        - avg_efficiency: Average fuel efficiency
        - max_efficiency: Maximum efficiency achieved
        - min_efficiency: Minimum efficiency recorded
        
    Raises:
        ValueError: If input data is empty or invalid
        
    Example:
        >>> fuel_df = pd.DataFrame({'fuel': [100, 90, 80]})
        >>> distance_df = pd.DataFrame({'distance': [10, 15, 12]})
        >>> metrics = calculate_fuel_efficiency(fuel_df, distance_df)
        >>> print(metrics['avg_efficiency'])
        0.125
    """
    if fuel_data.empty or distance_data.empty:
        raise ValueError("Input data cannot be empty")
        
    # Implementation here
    pass
```

### API Documentation

Use Django REST Framework's built-in documentation:

```python
# settings.py
INSTALLED_APPS = [
    # ...
    'rest_framework',
    'drf_spectacular',
]

REST_FRAMEWORK = {
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
}

# urls.py
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

urlpatterns = [
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
]
```

## 🚀 Deployment

### Development Deployment

```bash
# Start development servers
make dev-start

# Or manually:
python backend/manage.py runserver &
streamlit run tests/dashboard.py &
cd core/orchestration && kedro run &
```

### Production Deployment

```bash
# Build production image
docker build -t fueloptimine:latest .

# Deploy with docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Check deployment
make prod-health-check
```

## 🔧 Common Development Tasks

### Adding New Features

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/new-feature
   ```

2. **Implement Backend API**
   ```bash
   cd backend
   python manage.py startapp new_feature
   ```

3. **Add Tests**
   ```bash
   # Create test files
   touch tests/unit/test_new_feature.py
   ```

4. **Update Documentation**
   ```bash
   # Update relevant docs
   vim docs/api/README.md
   ```

5. **Submit Pull Request**

### Database Operations

```bash
# Create new migration
python manage.py makemigrations app_name

# Apply migrations
python manage.py migrate

# Reset database (development only)
python manage.py flush

# Load fixtures
python manage.py loaddata fixtures/sample_data.json
```

### Performance Optimization

```python
# Use database indexes
class FuelReading(models.Model):
    timestamp = models.DateTimeField(db_index=True)
    equipment_id = models.CharField(max_length=50, db_index=True)
    
    class Meta:
        indexes = [
            models.Index(fields=['timestamp', 'equipment_id']),
        ]

# Optimize queries
readings = FuelReading.objects.select_related('equipment').filter(
    timestamp__gte=start_date
).order_by('timestamp')

# Use database aggregation
from django.db.models import Avg, Sum
stats = FuelReading.objects.aggregate(
    avg_fuel=Avg('fuel_level'),
    total_consumption=Sum('consumption')
)
```

## 🤝 Contributing Guidelines

### Code Review Process

1. All code must be reviewed before merging
2. Tests must pass in CI/CD pipeline
3. Code coverage should not decrease
4. Documentation must be updated for new features

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Documentation
- [ ] API documentation updated
- [ ] User guide updated
- [ ] Developer guide updated
```

## 📞 Support and Resources

### Getting Help

1. **Documentation**: Check this guide and API docs
2. **Issues**: Search existing GitHub issues
3. **Community**: Ask questions in discussions
4. **Code Review**: Request review from team members

### Useful Resources

- [Django Documentation](https://docs.djangoproject.com/)
- [Kedro Documentation](https://kedro.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Plotly Documentation](https://plotly.com/python/)
- [ClickHouse Documentation](https://clickhouse.com/docs/)

### Development Tools

```bash
# Install development tools
pip install black isort flake8 mypy pytest pytest-cov

# Pre-commit hooks
pip install pre-commit
pre-commit install
```

This developer guide provides a comprehensive foundation for working on the FuelOptiMine project. For specific implementation details, refer to the inline code documentation and API specifications.