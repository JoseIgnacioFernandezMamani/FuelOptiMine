# Contributing Guide

Thank you for your interest in contributing to FuelOptiMine! This guide will help you get started with contributing to the project.

## 🤝 How to Contribute

We welcome contributions in many forms:

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new functionality
- **Code Contributions**: Submit bug fixes and new features
- **Documentation**: Improve or add documentation
- **Testing**: Help test new features and find bugs
- **Community Support**: Help other users in discussions

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

- Python 3.10 or higher
- Git installed and configured
- Basic knowledge of Django, Streamlit, and data analytics
- Familiarity with the project structure (see [Developer Guide](developer-guide.md))

### Setting Up Development Environment

1. **Fork the Repository**
   ```bash
   # Go to GitHub and fork the repository
   # Then clone your fork
   git clone https://github.com/your-username/FuelOptiMine.git
   cd FuelOptiMine
   ```

2. **Set Up Local Environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate   # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   
   # Set up pre-commit hooks
   pre-commit install
   ```

3. **Configure Environment**
   ```bash
   # Copy example environment file
   cp .env.example .env.development
   
   # Edit configuration for development
   nano .env.development
   ```

4. **Initialize Databases**
   ```bash
   # Set up PostgreSQL
   python backend/manage.py migrate
   
   # Initialize ClickHouse
   python scripts/init_clickhouse.py
   
   # Load sample data
   python scripts/load_sample_data.py
   ```

5. **Verify Setup**
   ```bash
   # Run tests
   pytest
   
   # Start development server
   make dev-start
   
   # Access application
   # Backend: http://localhost:8000
   # Frontend: http://localhost:8501
   ```

## 📋 Contribution Workflow

### 1. Choose What to Work On

- **Issues**: Check the [Issues](https://github.com/JoseIgnacioFernandezMamani/FuelOptiMine/issues) tab for open issues
- **Good First Issues**: Look for issues labeled `good first issue`
- **Feature Requests**: Check for `enhancement` labeled issues
- **Bugs**: Look for `bug` labeled issues

### 2. Create a Branch

```bash
# Create and switch to a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description

# Keep branch names descriptive and concise
```

### 3. Make Changes

Follow our coding standards:
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Follow PEP 8 style guidelines
- Add type hints where appropriate
- Write tests for new functionality

### 4. Test Your Changes

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_extractors.py

# Run with coverage
pytest --cov=backend --cov=etl_core

# Run linting
black .
isort .
flake8 .
mypy backend/
```

### 5. Commit Changes

Follow our commit message conventions:

```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat: add fuel efficiency calculator

- Implement new efficiency calculation algorithm
- Add unit tests for edge cases
- Update API documentation"
```

**Commit Message Format:**
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(analytics): add real-time fuel monitoring
fix(api): resolve authentication token refresh issue
docs(readme): update installation instructions
test(etl): add unit tests for CSV extractor
```

### 6. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Go to GitHub and create a Pull Request
# Fill out the PR template completely
```

## 📝 Pull Request Guidelines

### PR Template

When creating a PR, please include:

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## How Has This Been Tested?
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Tested on development environment

## Screenshots (if applicable)
Add screenshots to show UI changes.

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published
```

### Review Process

1. **Automated Checks**: All PRs must pass CI/CD checks
2. **Code Review**: At least one maintainer must review and approve
3. **Testing**: Verify that tests pass and functionality works
4. **Documentation**: Ensure documentation is updated if needed

### PR Requirements

- **Tests**: All new features must include tests
- **Documentation**: Update relevant documentation
- **Backwards Compatibility**: Avoid breaking changes when possible
- **Code Quality**: Follow project coding standards
- **Performance**: Consider performance implications

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interaction
├── e2e/           # End-to-end tests for full workflows
├── fixtures/      # Test data and fixtures
└── conftest.py    # Pytest configuration
```

### Writing Tests

#### Unit Tests

```python
# tests/unit/test_fuel_calculator.py
import pytest
from analytics.calculations import FuelEfficiencyCalculator

class TestFuelEfficiencyCalculator:
    """Test fuel efficiency calculation functions."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator instance for testing."""
        return FuelEfficiencyCalculator()
    
    @pytest.fixture
    def sample_data(self):
        """Provide sample fuel data."""
        return {
            'fuel_consumed': 100.0,
            'distance_traveled': 50.0,
            'cargo_weight': 1000.0
        }
    
    def test_calculate_basic_efficiency(self, calculator, sample_data):
        """Test basic efficiency calculation."""
        efficiency = calculator.calculate_efficiency(sample_data)
        
        expected = 50.0 / 100.0  # distance / fuel
        assert efficiency == expected
    
    def test_calculate_efficiency_with_weight(self, calculator, sample_data):
        """Test efficiency calculation considering cargo weight."""
        efficiency = calculator.calculate_weighted_efficiency(sample_data)
        
        # Should return efficiency per ton-km
        assert efficiency > 0
        assert isinstance(efficiency, float)
    
    def test_handle_zero_fuel_consumption(self, calculator):
        """Test handling of edge case with zero fuel consumption."""
        data = {
            'fuel_consumed': 0.0,
            'distance_traveled': 50.0,
            'cargo_weight': 1000.0
        }
        
        with pytest.raises(ValueError, match="Fuel consumption cannot be zero"):
            calculator.calculate_efficiency(data)
    
    @pytest.mark.parametrize("fuel,distance,expected", [
        (100.0, 50.0, 0.5),
        (200.0, 100.0, 0.5),
        (50.0, 25.0, 0.5),
    ])
    def test_efficiency_parametrized(self, calculator, fuel, distance, expected):
        """Test efficiency calculation with multiple data points."""
        data = {
            'fuel_consumed': fuel,
            'distance_traveled': distance,
            'cargo_weight': 1000.0
        }
        
        efficiency = calculator.calculate_efficiency(data)
        assert efficiency == expected
```

#### Integration Tests

```python
# tests/integration/test_api_analytics.py
import pytest
from django.test import TestCase
from rest_framework.test import APIClient
from django.contrib.auth.models import User

class TestAnalyticsAPI(TestCase):
    """Test analytics API integration."""
    
    def setUp(self):
        """Set up test data."""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)
    
    def test_fuel_consumption_endpoint(self):
        """Test fuel consumption data retrieval."""
        response = self.client.get('/api/analytics/fuel-consumption/')
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('results', response.data)
    
    def test_fuel_consumption_filtering(self):
        """Test fuel consumption data filtering."""
        response = self.client.get(
            '/api/analytics/fuel-consumption/',
            {'equipment_id': 'TRUCK_001', 'start_date': '2024-01-01'}
        )
        
        self.assertEqual(response.status_code, 200)
        # Add assertions for filtered data
```

#### End-to-End Tests

```python
# tests/e2e/test_dashboard_workflow.py
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class TestDashboardWorkflow:
    """Test complete dashboard workflow."""
    
    @pytest.fixture
    def driver(self):
        """Set up Selenium driver."""
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(options=options)
        yield driver
        driver.quit()
    
    def test_dashboard_load_and_display_data(self, driver):
        """Test dashboard loads and displays data correctly."""
        # Navigate to dashboard
        driver.get('http://localhost:8501')
        
        # Wait for page to load
        wait = WebDriverWait(driver, 10)
        
        # Check if fuel level metric is displayed
        fuel_metric = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "metric-container"))
        )
        assert fuel_metric.is_displayed()
        
        # Check if chart is rendered
        chart_element = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "plotly-chart"))
        )
        assert chart_element.is_displayed()
```

### Test Data Management

#### Fixtures

```python
# tests/conftest.py
import pytest
import pandas as pd
from datetime import datetime, timedelta

@pytest.fixture(scope="session")
def sample_fuel_data():
    """Create sample fuel consumption data."""
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
        freq='H'
    )
    
    return pd.DataFrame({
        'timestamp': dates,
        'equipment_id': ['TRUCK_001'] * len(dates),
        'fuel_level_liters': np.random.uniform(500, 1000, len(dates)),
        'location_lat': np.random.uniform(-22.5, -22.0, len(dates)),
        'location_lon': np.random.uniform(-67.5, -67.0, len(dates))
    })

@pytest.fixture
def authenticated_api_client():
    """Create authenticated API client."""
    from rest_framework.test import APIClient
    from django.contrib.auth.models import User
    
    client = APIClient()
    user = User.objects.create_user(
        username='testuser',
        password='testpass123'
    )
    client.force_authenticate(user=user)
    return client
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_fuel_calculator.py

# Run tests with coverage
pytest --cov=backend --cov=etl_core --cov-report=html

# Run tests in parallel
pytest -n auto

# Run only failed tests from last run
pytest --lf

# Run tests matching pattern
pytest -k "test_fuel"

# Run tests with specific markers
pytest -m "slow"
```

## 📚 Documentation Guidelines

### Documentation Standards

- Use clear, concise language
- Include code examples where appropriate
- Keep documentation up-to-date with code changes
- Use proper Markdown formatting
- Include links to related documentation

### Types of Documentation

#### Code Documentation

```python
def calculate_fuel_efficiency(
    fuel_consumed: float, 
    distance_traveled: float, 
    cargo_weight: Optional[float] = None
) -> float:
    """
    Calculate fuel efficiency for mining equipment.
    
    Args:
        fuel_consumed: Amount of fuel consumed in liters
        distance_traveled: Distance traveled in kilometers
        cargo_weight: Optional cargo weight in tons
        
    Returns:
        Fuel efficiency in km/liter or ton-km/liter if cargo_weight provided
        
    Raises:
        ValueError: If fuel_consumed is zero or negative
        TypeError: If inputs are not numeric
        
    Example:
        >>> calculate_fuel_efficiency(100.0, 50.0)
        0.5
        >>> calculate_fuel_efficiency(100.0, 50.0, 10.0)
        5.0
    """
    if fuel_consumed <= 0:
        raise ValueError("Fuel consumption must be positive")
    
    if cargo_weight is not None:
        return (distance_traveled * cargo_weight) / fuel_consumed
    
    return distance_traveled / fuel_consumed
```

#### API Documentation

```python
class FuelConsumptionViewSet(viewsets.ModelViewSet):
    """
    API endpoint for fuel consumption data.
    
    Provides CRUD operations for fuel consumption records with filtering,
    pagination, and real-time capabilities.
    
    Endpoints:
        GET /api/fuel-consumption/ - List fuel consumption records
        POST /api/fuel-consumption/ - Create new record
        GET /api/fuel-consumption/{id}/ - Retrieve specific record
        PUT /api/fuel-consumption/{id}/ - Update record
        DELETE /api/fuel-consumption/{id}/ - Delete record
        GET /api/fuel-consumption/realtime/ - Real-time data stream
    
    Query Parameters:
        equipment_id (str): Filter by equipment ID
        start_date (date): Filter records after this date
        end_date (date): Filter records before this date
        granularity (str): Time granularity (hour, day, week, month)
    
    Example:
        GET /api/fuel-consumption/?equipment_id=TRUCK_001&start_date=2024-01-01
    """
```

#### User Guide Documentation

Use clear headings, step-by-step instructions, and screenshots:

```markdown
### Adding New Equipment

To add new equipment to the system:

1. **Navigate to Equipment Management**
   - Click on "Equipment" in the sidebar
   - Select "Add New Equipment"

2. **Fill Equipment Details**
   - Equipment ID: Unique identifier (e.g., TRUCK_001)
   - Name: Descriptive name (e.g., "CAT 797F Unit 1")
   - Type: Select equipment type from dropdown
   - Specifications: Enter fuel capacity, payload, etc.

3. **Save and Verify**
   - Click "Save Equipment"
   - Verify equipment appears in the list
   - Test by viewing in dashboard

![Equipment Management Screenshot](images/equipment-management.png)
```

## 🎨 Code Style Guidelines

### Python Code Style

We follow PEP 8 with some project-specific guidelines:

#### Imports

```python
# Standard library imports first
import os
import sys
from datetime import datetime, timedelta

# Third-party imports
import pandas as pd
import numpy as np
from django.db import models

# Local imports last
from analytics.models import FuelReading
from etl_core.extractors import CSVExtractor
```

#### Naming Conventions

```python
# Variables and functions: snake_case
fuel_consumption_rate = 42.5
user_profile_data = get_user_profile()

# Classes: PascalCase
class FuelConsumptionAnalyzer:
    pass

# Constants: UPPER_CASE
MAX_FUEL_CAPACITY = 5000
DEFAULT_TIMEOUT = 30

# Private methods: _leading_underscore
def _calculate_internal_metric(self):
    pass
```

#### Function Documentation

```python
def process_fuel_data(
    data: pd.DataFrame, 
    equipment_types: List[str], 
    normalize: bool = True
) -> Dict[str, float]:
    """
    Process fuel consumption data for analysis.
    
    This function cleans, validates, and processes fuel consumption data
    for different equipment types. It can optionally normalize the data
    based on equipment specifications.
    
    Args:
        data: DataFrame containing fuel consumption records with columns:
            - timestamp: datetime
            - equipment_id: str
            - fuel_level_liters: float
        equipment_types: List of equipment types to include in processing
        normalize: Whether to normalize data by equipment capacity
        
    Returns:
        Dictionary containing processed metrics:
        - total_consumption: Total fuel consumed
        - average_efficiency: Average fuel efficiency
        - equipment_count: Number of equipment processed
        
    Raises:
        ValueError: If data is empty or missing required columns
        TypeError: If data is not a pandas DataFrame
        
    Example:
        >>> data = pd.DataFrame({
        ...     'timestamp': [datetime.now()],
        ...     'equipment_id': ['TRUCK_001'],
        ...     'fuel_level_liters': [850.5]
        ... })
        >>> result = process_fuel_data(data, ['haul_truck'])
        >>> result['total_consumption']
        149.5
    """
```

### Frontend Code Style

#### Streamlit Applications

```python
import streamlit as st
import plotly.graph_objects as go

def create_fuel_dashboard():
    """Create main fuel monitoring dashboard."""
    # Page configuration
    st.set_page_config(
        page_title="Fuel Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("⛽ Fuel Monitoring Dashboard")
    st.markdown("Real-time fuel consumption monitoring for mining operations")
    
    # Sidebar controls
    with st.sidebar:
        equipment_id = st.selectbox(
            "Select Equipment",
            options=get_equipment_list(),
            help="Choose equipment to monitor"
        )
        
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=7), datetime.now()),
            help="Select date range for analysis"
        )
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_fuel_level_metric(equipment_id)
    
    with col2:
        display_consumption_rate(equipment_id)
    
    with col3:
        display_efficiency_score(equipment_id)
    
    # Charts
    display_fuel_trend_chart(equipment_id, date_range)
```

## 🏷️ Issue and PR Labels

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `question`: Further information is requested
- `wontfix`: This will not be worked on
- `priority:high`: High priority issue
- `priority:medium`: Medium priority issue
- `priority:low`: Low priority issue

### Component Labels

- `backend`: Related to Django backend
- `frontend`: Related to Streamlit frontend
- `etl`: Related to data processing
- `analytics`: Related to data analysis
- `api`: Related to API endpoints
- `database`: Related to database issues
- `deployment`: Related to deployment and infrastructure

## 🚀 Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Workflow

1. **Feature Freeze**: Stop adding new features
2. **Testing**: Comprehensive testing of release candidate
3. **Documentation**: Update documentation and changelog
4. **Release Notes**: Prepare detailed release notes
5. **Tag Release**: Create git tag and GitHub release
6. **Deploy**: Deploy to production environments

## 🎯 Development Priorities

### Current Focus Areas

1. **Performance Optimization**
   - Query optimization
   - Caching improvements
   - Frontend responsiveness

2. **Feature Completeness**
   - Advanced analytics
   - Prediction models
   - Optimization algorithms

3. **User Experience**
   - Dashboard improvements
   - Mobile responsiveness
   - Accessibility

4. **Monitoring and Reliability**
   - Better error handling
   - Comprehensive monitoring
   - Automated testing

### How to Help

- **New Contributors**: Start with `good first issue` labeled issues
- **Experienced Developers**: Tackle complex features or performance improvements
- **Domain Experts**: Help with mining industry requirements and validation
- **DevOps Engineers**: Improve deployment and monitoring
- **Technical Writers**: Enhance documentation and user guides

## 📞 Community and Support

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussions
- **Email**: Direct contact with maintainers

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

### Recognition

Contributors are recognized in:
- Release notes
- Contributors section in README
- GitHub contributors page
- Annual acknowledgments

Thank you for contributing to FuelOptiMine! Your efforts help improve mining operations and reduce environmental impact through better fuel efficiency.