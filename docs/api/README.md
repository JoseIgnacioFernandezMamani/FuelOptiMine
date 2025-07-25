# API Documentation

FuelOptiMine provides a comprehensive REST API built with Django REST Framework for programmatic access to all system functionality.

## 🌐 Base URL

- **Development:** `http://localhost:8000/api/`
- **Production:** `https://your-domain.com/api/`

## 🔐 Authentication

The API uses JWT (JSON Web Token) authentication.

### Obtaining a Token

```bash
curl -X POST http://localhost:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'
```

**Response:**
```json
{
  "access": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "user": {
    "id": 1,
    "username": "your_username",
    "email": "user@example.com"
  }
}
```

### Using the Token

Include the access token in the Authorization header:

```bash
curl -H "Authorization: Bearer <access_token>" \
  http://localhost:8000/api/analytics/fuel-consumption/
```

## 📊 Analytics Endpoints

### Fuel Consumption Data

#### Get Fuel Consumption History

```http
GET /api/analytics/fuel-consumption/
```

**Parameters:**
- `start_date` (optional): Start date in YYYY-MM-DD format
- `end_date` (optional): End date in YYYY-MM-DD format
- `equipment_id` (optional): Filter by specific equipment
- `granularity` (optional): `hour`, `day`, `week`, `month`

**Example Request:**
```bash
curl -H "Authorization: Bearer <token>" \
  "http://localhost:8000/api/analytics/fuel-consumption/?start_date=2024-01-01&granularity=day"
```

**Response:**
```json
{
  "count": 30,
  "results": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "equipment_id": "TRUCK_001",
      "fuel_level_liters": 850.5,
      "consumption_rate": 45.2,
      "efficiency_score": 0.85
    }
  ],
  "aggregations": {
    "total_consumption": 1350.8,
    "average_efficiency": 0.82,
    "peak_consumption": 65.4
  }
}
```

#### Get Real-time Fuel Data

```http
GET /api/analytics/fuel-consumption/realtime/
```

**Response:**
```json
{
  "timestamp": "2024-07-25T18:30:00Z",
  "active_equipment": [
    {
      "equipment_id": "TRUCK_001",
      "current_fuel_level": 750.2,
      "status": "operating",
      "location": {
        "lat": -22.3456,
        "lon": -67.1234
      },
      "last_update": "2024-07-25T18:29:45Z"
    }
  ],
  "system_totals": {
    "total_fuel_level": 15680.5,
    "active_vehicles": 12,
    "consumption_rate_per_hour": 245.8
  }
}
```

## 🤖 Prediction Endpoints

### Generate Fuel Consumption Predictions

```http
POST /api/predictions/fuel-consumption/
```

**Request Body:**
```json
{
  "equipment_id": "TRUCK_001",
  "prediction_horizon": 24,
  "scenario_parameters": {
    "weather_conditions": "normal",
    "load_factor": 0.85,
    "route_efficiency": 0.92
  }
}
```

**Response:**
```json
{
  "prediction_id": "pred_12345",
  "equipment_id": "TRUCK_001",
  "generated_at": "2024-07-25T18:30:00Z",
  "predictions": [
    {
      "hour": 1,
      "predicted_consumption": 42.5,
      "confidence_interval": [38.2, 46.8],
      "factors": {
        "weather_impact": 0.05,
        "load_impact": 0.15,
        "route_impact": -0.08
      }
    }
  ],
  "model_info": {
    "model_version": "v1.2.3",
    "accuracy_score": 0.94,
    "last_trained": "2024-07-20T00:00:00Z"
  }
}
```

### Get Prediction History

```http
GET /api/predictions/fuel-consumption/{prediction_id}/
```

## ⚙️ Optimization Endpoints

### Route Optimization

```http
POST /api/optimization/routes/
```

**Request Body:**
```json
{
  "vehicles": [
    {
      "vehicle_id": "TRUCK_001",
      "current_location": {"lat": -22.3456, "lon": -67.1234},
      "fuel_level": 750.2,
      "capacity": 100.0
    }
  ],
  "destinations": [
    {
      "location": {"lat": -22.3500, "lon": -67.1300},
      "priority": 1,
      "time_window": {"start": "08:00", "end": "17:00"}
    }
  ],
  "constraints": {
    "max_distance": 500,
    "fuel_efficiency_weight": 0.7,
    "time_weight": 0.3
  }
}
```

**Response:**
```json
{
  "optimization_id": "opt_67890",
  "status": "completed",
  "generated_at": "2024-07-25T18:30:00Z",
  "routes": [
    {
      "vehicle_id": "TRUCK_001",
      "route": [
        {
          "location": {"lat": -22.3456, "lon": -67.1234},
          "arrival_time": "08:00:00",
          "action": "start"
        },
        {
          "location": {"lat": -22.3500, "lon": -67.1300},
          "arrival_time": "08:45:00",
          "action": "delivery",
          "estimated_fuel_consumption": 15.2
        }
      ],
      "total_distance": 75.5,
      "estimated_time": 450,
      "estimated_fuel_consumption": 35.8
    }
  ],
  "optimization_metrics": {
    "total_fuel_saved": 12.5,
    "efficiency_improvement": 0.15,
    "solver_time": 2.3
  }
}
```

## 📈 Equipment Management

### Get Equipment List

```http
GET /api/equipment/
```

**Response:**
```json
{
  "count": 25,
  "results": [
    {
      "id": "TRUCK_001",
      "name": "CAT 797F - Unit 001",
      "type": "haul_truck",
      "specifications": {
        "fuel_capacity": 4500,
        "max_payload": 400,
        "engine_power": 3500
      },
      "status": "active",
      "last_maintenance": "2024-07-15T00:00:00Z",
      "current_location": {
        "lat": -22.3456,
        "lon": -67.1234
      }
    }
  ]
}
```

### Update Equipment Status

```http
PATCH /api/equipment/{equipment_id}/
```

**Request Body:**
```json
{
  "status": "maintenance",
  "notes": "Scheduled maintenance - fuel system check"
}
```

## 📊 Reporting Endpoints

### Generate Report

```http
POST /api/reports/generate/
```

**Request Body:**
```json
{
  "report_type": "fuel_efficiency",
  "date_range": {
    "start": "2024-07-01",
    "end": "2024-07-31"
  },
  "format": "pdf",
  "include_predictions": true,
  "email_recipients": ["manager@company.com"]
}
```

**Response:**
```json
{
  "report_id": "rpt_11111",
  "status": "processing",
  "estimated_completion": "2024-07-25T18:35:00Z",
  "download_url": null
}
```

### Download Report

```http
GET /api/reports/{report_id}/download/
```

## 🔍 System Information

### Health Check

```http
GET /api/health/
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-07-25T18:30:00Z",
  "services": {
    "database": "healthy",
    "clickhouse": "healthy",
    "redis": "healthy",
    "ml_models": "healthy"
  },
  "version": "1.0.0",
  "uptime": 86400
}
```

### System Metrics

```http
GET /api/system/metrics/
```

**Response:**
```json
{
  "cpu_usage": 45.2,
  "memory_usage": 68.5,
  "disk_usage": 32.1,
  "active_connections": 15,
  "requests_per_minute": 120,
  "model_predictions_today": 1250
}
```

## 🔄 WebSocket API

For real-time updates, connect to the WebSocket endpoint:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime/');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Real-time update:', data);
};

// Subscribe to specific channels
ws.send(JSON.stringify({
    'action': 'subscribe',
    'channels': ['fuel_levels', 'alerts', 'predictions']
}));
```

## 🚨 Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "equipment_id": ["This field is required"],
      "date_range": ["Start date must be before end date"]
    },
    "timestamp": "2024-07-25T18:30:00Z",
    "request_id": "req_12345"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|------------|-------------|
| `AUTHENTICATION_REQUIRED` | 401 | Missing or invalid authentication token |
| `PERMISSION_DENIED` | 403 | Insufficient permissions for this operation |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource does not exist |
| `VALIDATION_ERROR` | 400 | Request data validation failed |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Internal server error |

## 📚 SDK and Examples

### Python SDK

```python
from fueloptimine_sdk import FuelOptimineClient

client = FuelOptimineClient(
    base_url="http://localhost:8000/api/",
    token="your_access_token"
)

# Get fuel consumption data
data = client.analytics.get_fuel_consumption(
    start_date="2024-07-01",
    equipment_id="TRUCK_001"
)

# Generate predictions
prediction = client.predictions.create_fuel_prediction(
    equipment_id="TRUCK_001",
    horizon=24
)
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

const client = axios.create({
  baseURL: 'http://localhost:8000/api/',
  headers: {
    'Authorization': 'Bearer your_access_token'
  }
});

// Get real-time data
const realTimeData = await client.get('/analytics/fuel-consumption/realtime/');
console.log(realTimeData.data);
```

## 📋 Rate Limits

- **Standard users:** 1000 requests per hour
- **Premium users:** 5000 requests per hour
- **Webhook endpoints:** 100 requests per minute

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Request limit per time window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Time when the current window resets

## 🔗 API Versioning

The API uses URL versioning:
- Current version: `v1`
- Future versions: `v2`, `v3`, etc.

Example: `http://localhost:8000/api/v1/analytics/fuel-consumption/`

## 📞 Support

For API support:
- Check this documentation
- Review example code in the `/examples` directory
- Open an issue on GitHub
- Contact the API team