# ETL Core Documentation

This document provides comprehensive information about the ETL (Extract, Transform, Load) core module of FuelOptiMine.

## 🏗️ Overview

The ETL Core module is responsible for extracting data from various sources, transforming it into a consistent format, and loading it into the target databases. It follows enterprise-grade design patterns and provides a flexible, extensible architecture.

## 📁 Architecture Overview

```
etl_core/
├── extract/                    # Data extraction components
│   ├── interfaces/            # Abstract base classes and contracts
│   ├── implementations/       # Concrete extractor implementations
│   │   ├── local/            # Local file extractors
│   │   └── external/         # External API/database extractors
│   ├── factories/            # Factory pattern for extractor creation
│   ├── config/               # Configuration schemas and validation
│   ├── exceptions/           # Custom exception classes
│   ├── models/               # Data models and structures
│   ├── tests/                # Unit and integration tests
│   └── utils/                # Utility functions and helpers
├── transform/                 # Data transformation components
│   ├── core/                 # Base transformation classes
│   ├── implementations/      # Specific transformers
│   ├── validators/           # Data validation rules
│   └── scripts/              # Transformation scripts
├── load/                     # Data loading components
│   ├── destinations/         # Target database adapters
│   ├── serializers/          # Data format serializers
│   └── batching/             # Batch processing utilities
└── utils/                    # Shared utilities and constants
```

## 🔧 Design Patterns

### 1. Factory Pattern

The ETL Core uses the Factory pattern to create appropriate extractors based on configuration:

```python
# etl_core/extract/factories/extractor_factory.py
from typing import Dict, Type
from etl_core.extract.interfaces.base_extractor import BaseExtractor
from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
from etl_core.extract.implementations.external.api_extractor import APIExtractor

class ExtractorFactory:
    """Factory for creating data extractors."""
    
    _extractors: Dict[str, Type[BaseExtractor]] = {
        'csv': CSVExtractor,
        'api': APIExtractor,
        'database': DatabaseExtractor,
        'json': JSONExtractor,
    }
    
    @classmethod
    def create_extractor(cls, extractor_type: str, **kwargs) -> BaseExtractor:
        """
        Create an extractor instance based on type.
        
        Args:
            extractor_type: Type of extractor to create
            **kwargs: Configuration parameters for the extractor
            
        Returns:
            Configured extractor instance
            
        Raises:
            ValueError: If extractor type is not supported
        """
        if extractor_type not in cls._extractors:
            raise ValueError(f"Unsupported extractor type: {extractor_type}")
        
        extractor_class = cls._extractors[extractor_type]
        return extractor_class(**kwargs)
    
    @classmethod
    def register_extractor(cls, name: str, extractor_class: Type[BaseExtractor]):
        """Register a new extractor type."""
        cls._extractors[name] = extractor_class
    
    @classmethod
    def get_available_extractors(cls) -> List[str]:
        """Get list of available extractor types."""
        return list(cls._extractors.keys())
```

### 2. Strategy Pattern

Different transformation strategies can be applied based on data type and requirements:

```python
# etl_core/transform/core/base_transformer.py
from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd

class TransformationStrategy(ABC):
    """Abstract base class for transformation strategies."""
    
    @abstractmethod
    def transform(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply transformation to data."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate transformation configuration."""
        pass

class FuelDataTransformer(TransformationStrategy):
    """Transformation strategy for fuel consumption data."""
    
    def transform(self, data: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Transform fuel consumption data."""
        # Clean data
        data = self._clean_outliers(data, config.get('outlier_threshold', 3.0))
        
        # Handle missing values
        data = self._handle_missing_values(data, config.get('missing_strategy', 'interpolate'))
        
        # Calculate derived metrics
        data = self._calculate_efficiency_metrics(data)
        
        # Add temporal features
        data = self._add_temporal_features(data)
        
        return data
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate fuel data transformation configuration."""
        required_columns = ['timestamp', 'equipment_id', 'fuel_level_liters']
        
        if 'required_columns' in config:
            return all(col in config['required_columns'] for col in required_columns)
        
        return True
    
    def _clean_outliers(self, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Remove outliers using statistical methods."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            z_scores = np.abs(stats.zscore(data[column].dropna()))
            data = data[z_scores < threshold]
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values based on strategy."""
        if strategy == 'interpolate':
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns] = data[numeric_columns].interpolate()
        elif strategy == 'forward_fill':
            data = data.fillna(method='ffill')
        elif strategy == 'drop':
            data = data.dropna()
        
        return data
    
    def _calculate_efficiency_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate fuel efficiency metrics."""
        if 'distance_traveled' in data.columns:
            data['fuel_efficiency'] = data['distance_traveled'] / data['fuel_consumed']
        
        if 'cargo_weight' in data.columns:
            data['ton_km_efficiency'] = (
                data['distance_traveled'] * data['cargo_weight'] / data['fuel_consumed']
            )
        
        return data
    
    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour_of_day'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['is_weekend'] = data['day_of_week'].isin([5, 6])
        data['month'] = data['timestamp'].dt.month
        data['quarter'] = data['timestamp'].dt.quarter
        
        return data
```

### 3. Template Method Pattern

The ETL pipeline follows a template method pattern:

```python
# etl_core/pipeline/etl_pipeline.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging

class ETLPipeline(ABC):
    """Abstract ETL pipeline template."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def execute(self) -> Dict[str, Any]:
        """Execute the complete ETL pipeline."""
        self.logger.info("Starting ETL pipeline execution")
        
        try:
            # Template method - defines the algorithm structure
            self._validate_config()
            extracted_data = self._extract()
            transformed_data = self._transform(extracted_data)
            load_result = self._load(transformed_data)
            self._cleanup()
            
            result = {
                'status': 'success',
                'records_processed': len(transformed_data),
                'load_result': load_result
            }
            
            self.logger.info(f"ETL pipeline completed successfully: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"ETL pipeline failed: {str(e)}")
            self._handle_error(e)
            raise
    
    def _validate_config(self) -> None:
        """Validate pipeline configuration."""
        required_keys = ['extract', 'transform', 'load']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration: {key}")
    
    @abstractmethod
    def _extract(self) -> Any:
        """Extract data from source - to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _transform(self, data: Any) -> Any:
        """Transform extracted data - to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _load(self, data: Any) -> Dict[str, Any]:
        """Load transformed data - to be implemented by subclasses."""
        pass
    
    def _cleanup(self) -> None:
        """Cleanup resources after pipeline execution."""
        self.logger.info("Cleaning up pipeline resources")
    
    def _handle_error(self, error: Exception) -> None:
        """Handle pipeline errors."""
        self.logger.error(f"Pipeline error: {str(error)}")
        # Could send alerts, cleanup resources, etc.

class FuelDataETLPipeline(ETLPipeline):
    """Concrete ETL pipeline for fuel consumption data."""
    
    def _extract(self) -> pd.DataFrame:
        """Extract fuel data from configured sources."""
        extractor_config = self.config['extract']
        extractor = ExtractorFactory.create_extractor(
            extractor_config['type'],
            **extractor_config['params']
        )
        
        data = extractor.extract()
        self.logger.info(f"Extracted {len(data)} records")
        return data
    
    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform fuel consumption data."""
        transform_config = self.config['transform']
        transformer = FuelDataTransformer()
        
        transformed_data = transformer.transform(data, transform_config)
        self.logger.info(f"Transformed data: {len(transformed_data)} records")
        return transformed_data
    
    def _load(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Load transformed data to target destinations."""
        load_config = self.config['load']
        loader = LoaderFactory.create_loader(
            load_config['type'],
            **load_config['params']
        )
        
        result = loader.load(data)
        self.logger.info(f"Loaded data: {result}")
        return result
```

## 📥 Extract Components

### Base Extractor Interface

```python
# etl_core/extract/interfaces/base_extractor.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd

class BaseExtractor(ABC):
    """Abstract base class for all data extractors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._connection = None
    
    @abstractmethod
    def extract(self) -> pd.DataFrame:
        """
        Extract data from the source.
        
        Returns:
            DataFrame containing extracted data
            
        Raises:
            ExtractionError: If extraction fails
        """
        pass
    
    @abstractmethod
    def validate_source(self) -> bool:
        """
        Validate that the data source is accessible and valid.
        
        Returns:
            True if source is valid, False otherwise
        """
        pass
    
    def get_schema(self) -> Dict[str, str]:
        """
        Get the schema of the data source.
        
        Returns:
            Dictionary mapping column names to data types
        """
        data = self.extract()
        return {col: str(dtype) for col, dtype in data.dtypes.items()}
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the data source.
        
        Returns:
            Dictionary containing metadata information
        """
        return {
            'source_type': self.__class__.__name__,
            'config': self.config,
            'schema': self.get_schema()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self._connection:
            self._connection.close()
```

### CSV Extractor Implementation

```python
# etl_core/extract/implementations/local/csv_extractor.py
import os
import pandas as pd
from typing import Dict, Any, Optional
from etl_core.extract.interfaces.base_extractor import BaseExtractor
from etl_core.extract.exceptions import ExtractionError

class CSVExtractor(BaseExtractor):
    """Extractor for CSV files."""
    
    def __init__(self, file_path: str, **kwargs):
        super().__init__(kwargs)
        self.file_path = file_path
        self.read_options = {
            'encoding': kwargs.get('encoding', 'utf-8'),
            'delimiter': kwargs.get('delimiter', ','),
            'header': kwargs.get('header', 0),
            'skiprows': kwargs.get('skiprows', None),
            'nrows': kwargs.get('nrows', None),
            'dtype': kwargs.get('dtype', None),
            'parse_dates': kwargs.get('parse_dates', None),
            'date_parser': kwargs.get('date_parser', None),
        }
    
    def extract(self) -> pd.DataFrame:
        """Extract data from CSV file."""
        try:
            if not self.validate_source():
                raise ExtractionError(f"Invalid CSV source: {self.file_path}")
            
            # Filter None values from read_options
            read_options = {k: v for k, v in self.read_options.items() if v is not None}
            
            data = pd.read_csv(self.file_path, **read_options)
            
            # Post-processing
            data = self._post_process(data)
            
            return data
            
        except FileNotFoundError:
            raise ExtractionError(f"CSV file not found: {self.file_path}")
        except pd.errors.EmptyDataError:
            raise ExtractionError(f"CSV file is empty: {self.file_path}")
        except pd.errors.ParserError as e:
            raise ExtractionError(f"CSV parsing error: {str(e)}")
        except Exception as e:
            raise ExtractionError(f"Unexpected error reading CSV: {str(e)}")
    
    def validate_source(self) -> bool:
        """Validate CSV file exists and is readable."""
        return (
            os.path.exists(self.file_path) and
            os.path.isfile(self.file_path) and
            os.access(self.file_path, os.R_OK)
        )
    
    def _post_process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply post-processing transformations."""
        # Remove empty rows
        data = data.dropna(how='all')
        
        # Convert timestamp columns if specified
        if 'timestamp_columns' in self.config:
            for col in self.config['timestamp_columns']:
                if col in data.columns:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
        
        # Apply column mapping if specified
        if 'column_mapping' in self.config:
            data = data.rename(columns=self.config['column_mapping'])
        
        return data
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get information about the CSV file."""
        if not self.validate_source():
            return {}
        
        stat = os.stat(self.file_path)
        return {
            'file_size': stat.st_size,
            'modified_time': stat.st_mtime,
            'created_time': stat.st_ctime,
        }
```

### Database Extractor Implementation

```python
# etl_core/extract/implementations/external/database_extractor.py
import pandas as pd
import clickhouse_connect
from typing import Dict, Any, Optional
from etl_core.extract.interfaces.base_extractor import BaseExtractor
from etl_core.extract.exceptions import ExtractionError

class ClickHouseExtractor(BaseExtractor):
    """Extractor for ClickHouse database."""
    
    def __init__(self, connection_config: Dict[str, Any], query: str, **kwargs):
        super().__init__(kwargs)
        self.connection_config = connection_config
        self.query = query
        self.batch_size = kwargs.get('batch_size', 10000)
    
    def extract(self) -> pd.DataFrame:
        """Extract data from ClickHouse database."""
        try:
            with self._get_connection() as client:
                if self.batch_size:
                    return self._extract_batched(client)
                else:
                    return client.query_df(self.query)
                    
        except Exception as e:
            raise ExtractionError(f"ClickHouse extraction failed: {str(e)}")
    
    def validate_source(self) -> bool:
        """Validate ClickHouse connection."""
        try:
            with self._get_connection() as client:
                client.command("SELECT 1")
                return True
        except Exception:
            return False
    
    def _get_connection(self):
        """Get ClickHouse connection."""
        return clickhouse_connect.get_client(**self.connection_config)
    
    def _extract_batched(self, client) -> pd.DataFrame:
        """Extract data in batches to manage memory."""
        all_data = []
        offset = 0
        
        while True:
            batch_query = f"{self.query} LIMIT {self.batch_size} OFFSET {offset}"
            batch_data = client.query_df(batch_query)
            
            if batch_data.empty:
                break
            
            all_data.append(batch_data)
            offset += self.batch_size
        
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    
    def test_query(self) -> Dict[str, Any]:
        """Test the query and return basic statistics."""
        try:
            with self._get_connection() as client:
                # Test query with LIMIT 1
                test_query = f"SELECT COUNT(*) as total_rows FROM ({self.query})"
                result = client.query_df(test_query)
                
                return {
                    'query_valid': True,
                    'estimated_rows': result.iloc[0]['total_rows'],
                }
        except Exception as e:
            return {
                'query_valid': False,
                'error': str(e)
            }
```

## 🔄 Transform Components

### Data Validation

```python
# etl_core/transform/validators/data_validator.py
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class ValidationRule:
    """Data validation rule definition."""
    column: str
    rule_type: str
    parameters: Dict[str, Any]
    severity: str = 'error'  # error, warning, info
    message: Optional[str] = None

class DataValidator:
    """Comprehensive data validation for ETL processes."""
    
    def __init__(self):
        self.validation_results = []
    
    def validate(self, data: pd.DataFrame, rules: List[ValidationRule]) -> Dict[str, Any]:
        """
        Validate data against a set of rules.
        
        Args:
            data: DataFrame to validate
            rules: List of validation rules to apply
            
        Returns:
            Dictionary containing validation results
        """
        self.validation_results = []
        
        for rule in rules:
            try:
                result = self._apply_rule(data, rule)
                self.validation_results.append(result)
            except Exception as e:
                self.validation_results.append({
                    'rule': rule,
                    'status': 'error',
                    'message': f"Validation rule failed: {str(e)}"
                })
        
        return self._summarize_results()
    
    def _apply_rule(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Apply a single validation rule."""
        if rule.rule_type == 'not_null':
            return self._validate_not_null(data, rule)
        elif rule.rule_type == 'range':
            return self._validate_range(data, rule)
        elif rule.rule_type == 'unique':
            return self._validate_unique(data, rule)
        elif rule.rule_type == 'format':
            return self._validate_format(data, rule)
        elif rule.rule_type == 'custom':
            return self._validate_custom(data, rule)
        else:
            raise ValueError(f"Unknown validation rule type: {rule.rule_type}")
    
    def _validate_not_null(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate that column has no null values."""
        column = rule.column
        if column not in data.columns:
            return {
                'rule': rule,
                'status': 'error',
                'message': f"Column '{column}' not found in data"
            }
        
        null_count = data[column].isnull().sum()
        total_count = len(data)
        
        if null_count > 0:
            return {
                'rule': rule,
                'status': rule.severity,
                'message': f"Column '{column}' has {null_count}/{total_count} null values",
                'details': {
                    'null_count': null_count,
                    'null_percentage': (null_count / total_count) * 100
                }
            }
        
        return {
            'rule': rule,
            'status': 'passed',
            'message': f"Column '{column}' has no null values"
        }
    
    def _validate_range(self, data: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate that numeric column values are within specified range."""
        column = rule.column
        min_val = rule.parameters.get('min')
        max_val = rule.parameters.get('max')
        
        if column not in data.columns:
            return {
                'rule': rule,
                'status': 'error',
                'message': f"Column '{column}' not found in data"
            }
        
        # Convert to numeric if possible
        try:
            numeric_data = pd.to_numeric(data[column], errors='coerce')
        except:
            return {
                'rule': rule,
                'status': 'error',
                'message': f"Column '{column}' cannot be converted to numeric"
            }
        
        violations = 0
        if min_val is not None:
            violations += (numeric_data < min_val).sum()
        if max_val is not None:
            violations += (numeric_data > max_val).sum()
        
        if violations > 0:
            return {
                'rule': rule,
                'status': rule.severity,
                'message': f"Column '{column}' has {violations} values outside range [{min_val}, {max_val}]",
                'details': {
                    'violations': violations,
                    'violation_percentage': (violations / len(data)) * 100
                }
            }
        
        return {
            'rule': rule,
            'status': 'passed',
            'message': f"Column '{column}' values are within valid range"
        }
    
    def _summarize_results(self) -> Dict[str, Any]:
        """Summarize validation results."""
        total_rules = len(self.validation_results)
        passed = sum(1 for r in self.validation_results if r['status'] == 'passed')
        errors = sum(1 for r in self.validation_results if r['status'] == 'error')
        warnings = sum(1 for r in self.validation_results if r['status'] == 'warning')
        
        return {
            'total_rules': total_rules,
            'passed': passed,
            'errors': errors,
            'warnings': warnings,
            'overall_status': 'passed' if errors == 0 else 'failed',
            'details': self.validation_results
        }
```

### Data Quality Monitoring

```python
# etl_core/transform/monitoring/data_quality.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class QualityMetric:
    """Data quality metric definition."""
    name: str
    value: float
    threshold: float
    status: str
    timestamp: datetime

class DataQualityMonitor:
    """Monitor data quality metrics during ETL processing."""
    
    def __init__(self):
        self.metrics = []
    
    def assess_quality(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall data quality."""
        quality_report = {
            'timestamp': datetime.now(),
            'record_count': len(data),
            'metrics': {}
        }
        
        # Completeness metrics
        quality_report['metrics']['completeness'] = self._calculate_completeness(data)
        
        # Accuracy metrics
        quality_report['metrics']['accuracy'] = self._calculate_accuracy(data, config)
        
        # Consistency metrics
        quality_report['metrics']['consistency'] = self._calculate_consistency(data, config)
        
        # Timeliness metrics
        quality_report['metrics']['timeliness'] = self._calculate_timeliness(data, config)
        
        # Overall quality score
        quality_report['overall_score'] = self._calculate_overall_score(quality_report['metrics'])
        
        return quality_report
    
    def _calculate_completeness(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate data completeness metrics."""
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        
        return {
            'overall_completeness': ((total_cells - missing_cells) / total_cells) * 100,
            'column_completeness': {
                col: ((len(data) - data[col].isnull().sum()) / len(data)) * 100
                for col in data.columns
            },
            'row_completeness': (
                (data.isnull().sum(axis=1) == 0).sum() / len(data)
            ) * 100
        }
    
    def _calculate_accuracy(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
        """Calculate data accuracy metrics."""
        accuracy_metrics = {}
        
        # Check for outliers using IQR method
        for column in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
            accuracy_metrics[f'{column}_outlier_rate'] = (len(outliers) / len(data)) * 100
        
        return accuracy_metrics
    
    def _calculate_consistency(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
        """Calculate data consistency metrics."""
        consistency_metrics = {}
        
        # Check for duplicate records
        duplicate_rate = (data.duplicated().sum() / len(data)) * 100
        consistency_metrics['duplicate_rate'] = duplicate_rate
        
        # Check format consistency for string columns
        for column in data.select_dtypes(include=['object']).columns:
            if column in config.get('format_patterns', {}):
                pattern = config['format_patterns'][column]
                valid_format = data[column].str.match(pattern, na=False)
                consistency_metrics[f'{column}_format_consistency'] = (
                    valid_format.sum() / len(data)
                ) * 100
        
        return consistency_metrics
    
    def _calculate_timeliness(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
        """Calculate data timeliness metrics."""
        timeliness_metrics = {}
        
        # Check if timestamp columns exist and are recent
        timestamp_columns = config.get('timestamp_columns', [])
        current_time = datetime.now()
        
        for column in timestamp_columns:
            if column in data.columns:
                timestamps = pd.to_datetime(data[column], errors='coerce')
                
                # Calculate age of most recent record
                if not timestamps.isna().all():
                    most_recent = timestamps.max()
                    age_hours = (current_time - most_recent).total_seconds() / 3600
                    timeliness_metrics[f'{column}_freshness_hours'] = age_hours
        
        return timeliness_metrics
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        scores = []
        
        # Completeness score (weight: 30%)
        if 'completeness' in metrics:
            completeness_score = metrics['completeness'].get('overall_completeness', 0)
            scores.append(completeness_score * 0.3)
        
        # Accuracy score (weight: 40%)
        if 'accuracy' in metrics:
            accuracy_scores = [
                100 - rate for rate in metrics['accuracy'].values()
                if 'outlier_rate' in str(rate)
            ]
            if accuracy_scores:
                avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
                scores.append(avg_accuracy * 0.4)
        
        # Consistency score (weight: 30%)
        if 'consistency' in metrics:
            consistency_score = 100 - metrics['consistency'].get('duplicate_rate', 0)
            scores.append(consistency_score * 0.3)
        
        return sum(scores) if scores else 0
```

## 📤 Load Components

### ClickHouse Loader

```python
# etl_core/load/destinations/clickhouse_loader.py
import pandas as pd
import clickhouse_connect
from typing import Dict, Any, List, Optional
from etl_core.load.interfaces.base_loader import BaseLoader
from etl_core.load.exceptions import LoadError

class ClickHouseLoader(BaseLoader):
    """Loader for ClickHouse database."""
    
    def __init__(self, connection_config: Dict[str, Any], table_name: str, **kwargs):
        super().__init__(kwargs)
        self.connection_config = connection_config
        self.table_name = table_name
        self.batch_size = kwargs.get('batch_size', 10000)
        self.create_table = kwargs.get('create_table', False)
        self.table_schema = kwargs.get('table_schema', {})
    
    def load(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Load data into ClickHouse table."""
        try:
            with self._get_connection() as client:
                # Create table if requested
                if self.create_table:
                    self._create_table_if_not_exists(client, data)
                
                # Load data in batches
                total_rows = len(data)
                loaded_rows = 0
                
                for batch_start in range(0, total_rows, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, total_rows)
                    batch_data = data.iloc[batch_start:batch_end]
                    
                    client.insert_df(self.table_name, batch_data)
                    loaded_rows += len(batch_data)
                
                return {
                    'status': 'success',
                    'table': self.table_name,
                    'total_rows': total_rows,
                    'loaded_rows': loaded_rows,
                    'batches': (total_rows + self.batch_size - 1) // self.batch_size
                }
                
        except Exception as e:
            raise LoadError(f"ClickHouse load failed: {str(e)}")
    
    def _get_connection(self):
        """Get ClickHouse connection."""
        return clickhouse_connect.get_client(**self.connection_config)
    
    def _create_table_if_not_exists(self, client, data: pd.DataFrame):
        """Create table if it doesn't exist."""
        if self.table_schema:
            columns = []
            for column, dtype in self.table_schema.items():
                columns.append(f"{column} {dtype}")
            
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                {', '.join(columns)}
            ) ENGINE = MergeTree()
            ORDER BY tuple()
            """
        else:
            # Infer schema from DataFrame
            columns = []
            for column, dtype in data.dtypes.items():
                clickhouse_type = self._map_pandas_to_clickhouse_type(dtype)
                columns.append(f"{column} {clickhouse_type}")
            
            create_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                {', '.join(columns)}
            ) ENGINE = MergeTree()
            ORDER BY tuple()
            """
        
        client.command(create_sql)
    
    def _map_pandas_to_clickhouse_type(self, pandas_dtype) -> str:
        """Map pandas dtype to ClickHouse type."""
        dtype_mapping = {
            'int64': 'Int64',
            'int32': 'Int32',
            'float64': 'Float64',
            'float32': 'Float32',
            'object': 'String',
            'bool': 'UInt8',
            'datetime64[ns]': 'DateTime64(3)',
        }
        
        dtype_str = str(pandas_dtype)
        return dtype_mapping.get(dtype_str, 'String')
    
    def validate_destination(self) -> bool:
        """Validate ClickHouse connection and table."""
        try:
            with self._get_connection() as client:
                client.command("SELECT 1")
                
                # Check if table exists
                result = client.query(
                    f"EXISTS TABLE {self.table_name}"
                ).first_row
                
                return bool(result[0])
        except Exception:
            return False
```

## ⚙️ Configuration Management

### Configuration Schema

```python
# etl_core/config/schemas.py
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, validator
from enum import Enum

class ExtractorType(str, Enum):
    CSV = "csv"
    JSON = "json"
    DATABASE = "database"
    API = "api"

class LoaderType(str, Enum):
    CLICKHOUSE = "clickhouse"
    POSTGRES = "postgres"
    CSV = "csv"
    JSON = "json"

class ExtractConfig(BaseModel):
    """Configuration for data extraction."""
    type: ExtractorType
    source: str
    params: Dict[str, Any] = {}
    
    @validator('params')
    def validate_params(cls, v, values):
        """Validate extractor parameters based on type."""
        extractor_type = values.get('type')
        
        if extractor_type == ExtractorType.CSV:
            required_params = ['file_path']
            for param in required_params:
                if param not in v:
                    raise ValueError(f"Missing required parameter for CSV extractor: {param}")
        
        return v

class TransformConfig(BaseModel):
    """Configuration for data transformation."""
    strategy: str = "default"
    outlier_threshold: float = 3.0
    missing_strategy: str = "interpolate"
    validation_rules: List[Dict[str, Any]] = []
    quality_thresholds: Dict[str, float] = {}

class LoadConfig(BaseModel):
    """Configuration for data loading."""
    type: LoaderType
    destination: str
    params: Dict[str, Any] = {}
    batch_size: int = 10000
    create_table: bool = False

class ETLConfig(BaseModel):
    """Complete ETL pipeline configuration."""
    name: str
    description: Optional[str] = None
    extract: ExtractConfig
    transform: TransformConfig
    load: LoadConfig
    monitoring: Dict[str, Any] = {}
    
    class Config:
        use_enum_values = True
```

### Configuration Examples

```yaml
# config/fuel_data_etl.yaml
name: "fuel_consumption_etl"
description: "ETL pipeline for fuel consumption data"

extract:
  type: "csv"
  source: "data/raw/fuel_readings.csv"
  params:
    encoding: "utf-8"
    delimiter: ","
    parse_dates: ["timestamp"]
    timestamp_columns: ["timestamp"]
    column_mapping:
      "Timestamp": "timestamp"
      "EquipmentID": "equipment_id"
      "FuelLevel": "fuel_level_liters"

transform:
  strategy: "fuel_data"
  outlier_threshold: 3.0
  missing_strategy: "interpolate"
  validation_rules:
    - column: "timestamp"
      rule_type: "not_null"
      severity: "error"
    - column: "fuel_level_liters"
      rule_type: "range"
      parameters:
        min: 0
        max: 5000
      severity: "warning"
    - column: "equipment_id"
      rule_type: "not_null"
      severity: "error"
  quality_thresholds:
    completeness: 95.0
    accuracy: 90.0
    consistency: 95.0

load:
  type: "clickhouse"
  destination: "fuel_readings"
  params:
    host: "localhost"
    port: 8123
    database: "fuel_optimine"
    username: "default"
    password: ""
  batch_size: 5000
  create_table: true

monitoring:
  enable_quality_monitoring: true
  quality_report_path: "logs/quality_reports/"
  alert_thresholds:
    error_rate: 5.0
    processing_time: 300  # seconds
```

## 🧪 Testing ETL Components

### Unit Tests Example

```python
# tests/unit/test_csv_extractor.py
import pytest
import pandas as pd
import tempfile
import os
from etl_core.extract.implementations.local.csv_extractor import CSVExtractor
from etl_core.extract.exceptions import ExtractionError

class TestCSVExtractor:
    """Test CSV extractor functionality."""
    
    @pytest.fixture
    def sample_csv_content(self):
        """Sample CSV content for testing."""
        return """timestamp,equipment_id,fuel_level_liters
2024-01-01 00:00:00,TRUCK_001,950.5
2024-01-01 01:00:00,TRUCK_001,945.2
2024-01-01 02:00:00,TRUCK_001,940.8
2024-01-01 00:00:00,TRUCK_002,1200.0
2024-01-01 01:00:00,TRUCK_002,1195.5"""
    
    @pytest.fixture
    def sample_csv_file(self, sample_csv_content):
        """Create temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(sample_csv_content)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        os.unlink(temp_file)
    
    def test_extract_valid_csv(self, sample_csv_file):
        """Test extracting data from valid CSV file."""
        extractor = CSVExtractor(
            file_path=sample_csv_file,
            parse_dates=['timestamp']
        )
        
        result = extractor.extract()
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert list(result.columns) == ['timestamp', 'equipment_id', 'fuel_level_liters']
        assert result['equipment_id'].unique().tolist() == ['TRUCK_001', 'TRUCK_002']
    
    def test_extract_nonexistent_file(self):
        """Test extracting from non-existent file raises error."""
        extractor = CSVExtractor(file_path="nonexistent.csv")
        
        with pytest.raises(ExtractionError, match="CSV file not found"):
            extractor.extract()
    
    def test_validate_source_valid_file(self, sample_csv_file):
        """Test source validation for valid file."""
        extractor = CSVExtractor(file_path=sample_csv_file)
        
        assert extractor.validate_source() is True
    
    def test_validate_source_invalid_file(self):
        """Test source validation for invalid file."""
        extractor = CSVExtractor(file_path="nonexistent.csv")
        
        assert extractor.validate_source() is False
    
    def test_get_schema(self, sample_csv_file):
        """Test schema extraction."""
        extractor = CSVExtractor(
            file_path=sample_csv_file,
            parse_dates=['timestamp']
        )
        
        schema = extractor.get_schema()
        
        assert 'timestamp' in schema
        assert 'equipment_id' in schema
        assert 'fuel_level_liters' in schema
    
    def test_column_mapping(self, sample_csv_content):
        """Test column mapping functionality."""
        # Create CSV with different column names
        modified_content = sample_csv_content.replace(
            'timestamp,equipment_id,fuel_level_liters',
            'Timestamp,EquipmentID,FuelLevel'
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(modified_content)
            temp_file = f.name
        
        try:
            extractor = CSVExtractor(
                file_path=temp_file,
                column_mapping={
                    'Timestamp': 'timestamp',
                    'EquipmentID': 'equipment_id',
                    'FuelLevel': 'fuel_level_liters'
                }
            )
            
            result = extractor.extract()
            
            assert list(result.columns) == ['timestamp', 'equipment_id', 'fuel_level_liters']
            
        finally:
            os.unlink(temp_file)
```

## 📊 Monitoring and Observability

### ETL Metrics Collection

```python
# etl_core/monitoring/metrics.py
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

@dataclass
class ETLMetrics:
    """ETL pipeline execution metrics."""
    pipeline_name: str
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: str = "running"
    records_extracted: int = 0
    records_transformed: int = 0
    records_loaded: int = 0
    errors: int = 0
    warnings: int = 0
    processing_time: float = 0.0
    memory_usage: Dict[str, float] = field(default_factory=dict)
    quality_score: float = 0.0
    
    def mark_completed(self, status: str = "success"):
        """Mark pipeline as completed."""
        self.end_time = datetime.now()
        self.status = status
        self.processing_time = (self.end_time - self.start_time).total_seconds()

class ETLMonitor:
    """Monitor ETL pipeline execution."""
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.metrics = ETLMetrics(pipeline_name=pipeline_name)
        self.logger = logging.getLogger(f"ETL.{pipeline_name}")
        
    def start_extraction(self):
        """Mark start of extraction phase."""
        self.logger.info("Starting data extraction")
        self.metrics.extraction_start = datetime.now()
    
    def complete_extraction(self, record_count: int):
        """Mark completion of extraction phase."""
        self.metrics.records_extracted = record_count
        self.logger.info(f"Extraction completed: {record_count} records")
    
    def start_transformation(self):
        """Mark start of transformation phase."""
        self.logger.info("Starting data transformation")
        self.metrics.transformation_start = datetime.now()
    
    def complete_transformation(self, record_count: int, quality_score: float = 0.0):
        """Mark completion of transformation phase."""
        self.metrics.records_transformed = record_count
        self.metrics.quality_score = quality_score
        self.logger.info(f"Transformation completed: {record_count} records, quality score: {quality_score}")
    
    def start_loading(self):
        """Mark start of loading phase."""
        self.logger.info("Starting data loading")
        self.metrics.loading_start = datetime.now()
    
    def complete_loading(self, record_count: int):
        """Mark completion of loading phase."""
        self.metrics.records_loaded = record_count
        self.logger.info(f"Loading completed: {record_count} records")
    
    def record_error(self, error: Exception):
        """Record an error during processing."""
        self.metrics.errors += 1
        self.logger.error(f"Pipeline error: {str(error)}")
    
    def record_warning(self, message: str):
        """Record a warning during processing."""
        self.metrics.warnings += 1
        self.logger.warning(f"Pipeline warning: {message}")
    
    def finalize(self, status: str = "success") -> Dict[str, Any]:
        """Finalize monitoring and return metrics."""
        self.metrics.mark_completed(status)
        
        metrics_dict = {
            'pipeline_name': self.metrics.pipeline_name,
            'execution_time': self.metrics.processing_time,
            'status': self.metrics.status,
            'records': {
                'extracted': self.metrics.records_extracted,
                'transformed': self.metrics.records_transformed,
                'loaded': self.metrics.records_loaded
            },
            'quality_score': self.metrics.quality_score,
            'errors': self.metrics.errors,
            'warnings': self.metrics.warnings
        }
        
        self.logger.info(f"Pipeline completed: {metrics_dict}")
        return metrics_dict
```

This comprehensive ETL Core documentation provides the foundation for understanding and extending the data processing capabilities of FuelOptiMine. The modular design allows for easy addition of new extractors, transformers, and loaders while maintaining consistency and reliability.