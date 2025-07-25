# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with FuelOptiMine.

## 🔍 General Troubleshooting Steps

### 1. Check System Status

```bash
# Check all services status
python scripts/health_check.py

# Check individual components
curl http://localhost:8000/health/  # Backend
curl http://localhost:8501/_stcore/health  # Frontend
curl http://localhost:8123/ping  # ClickHouse
redis-cli ping  # Redis
```

### 2. Check Logs

```bash
# Django logs
tail -f logs/fueloptimine.log

# Streamlit logs
streamlit logs

# ClickHouse logs
sudo tail -f /var/log/clickhouse-server/clickhouse-server.log

# Docker logs
docker-compose logs -f

# Kubernetes logs
kubectl logs -f deployment/fueloptimine-backend -n fueloptimine
```

### 3. Verify Configuration

```bash
# Check environment variables
python -c "
from decouple import config
print('DATABASE_URL:', config('DATABASE_URL', default='Not set'))
print('CLICKHOUSE_HOST:', config('CLICKHOUSE_HOST', default='Not set'))
print('REDIS_URL:', config('REDIS_URL', default='Not set'))
"

# Validate configuration
python scripts/config_validator.py
```

## 🗄️ Database Issues

### ClickHouse Problems

#### Issue: ClickHouse Connection Failed

**Symptoms:**
- `Connection refused` errors
- API endpoints returning 500 errors
- Dashboard not loading data

**Diagnosis:**
```bash
# Check if ClickHouse is running
systemctl status clickhouse-server

# Test connection
clickhouse-client --query "SELECT 1"

# Check port accessibility
telnet localhost 8123
telnet localhost 9000
```

**Solutions:**

1. **Start ClickHouse Service:**
```bash
sudo systemctl start clickhouse-server
sudo systemctl enable clickhouse-server
```

2. **Check Configuration:**
```bash
# Verify config file
sudo nano /etc/clickhouse-server/config.xml

# Check if ports are correctly configured
sudo netstat -tlnp | grep clickhouse
```

3. **Reset ClickHouse:**
```bash
# Stop service
sudo systemctl stop clickhouse-server

# Clear data (CAUTION: This deletes all data)
sudo rm -rf /var/lib/clickhouse/data/fuel_optimine/

# Restart and reinitialize
sudo systemctl start clickhouse-server
python scripts/init_clickhouse.py
```

#### Issue: ClickHouse Out of Memory

**Symptoms:**
- Queries failing with memory errors
- Slow query performance
- Server becoming unresponsive

**Diagnosis:**
```sql
-- Check memory usage
SELECT 
    formatReadableSize(memory_usage) as memory_usage,
    formatReadableSize(max_memory_usage) as max_memory_usage
FROM system.processes 
WHERE query != '';

-- Check running queries
SELECT query_id, user, memory_usage, query FROM system.processes;
```

**Solutions:**

1. **Increase Memory Limits:**
```xml
<!-- In /etc/clickhouse-server/users.xml -->
<profiles>
    <default>
        <max_memory_usage>10000000000</max_memory_usage>
    </default>
</profiles>
```

2. **Optimize Queries:**
```sql
-- Use LIMIT for large datasets
SELECT * FROM fuel_readings ORDER BY timestamp DESC LIMIT 10000;

-- Use proper WHERE clauses
SELECT * FROM fuel_readings 
WHERE timestamp >= yesterday() 
AND equipment_id = 'TRUCK_001';
```

3. **Partition Management:**
```sql
-- Check partition sizes
SELECT 
    partition,
    rows,
    formatReadableSize(bytes_on_disk) as size
FROM system.parts 
WHERE table = 'fuel_readings' 
AND active = 1;

-- Drop old partitions if needed
ALTER TABLE fuel_readings DROP PARTITION '202401';
```

### PostgreSQL Problems

#### Issue: Database Connection Errors

**Symptoms:**
- Django migration failures
- API authentication errors
- User management issues

**Diagnosis:**
```bash
# Test PostgreSQL connection
psql -h localhost -U postgres -d fueloptimine -c "SELECT version();"

# Check Django database connection
python backend/manage.py shell -c "
from django.db import connection
cursor = connection.cursor()
cursor.execute('SELECT 1')
print('Database connection successful')
"
```

**Solutions:**

1. **Check PostgreSQL Service:**
```bash
sudo systemctl status postgresql
sudo systemctl start postgresql
```

2. **Verify Database Exists:**
```bash
psql -h localhost -U postgres -c "\l" | grep fueloptimine
```

3. **Run Migrations:**
```bash
cd backend
python manage.py migrate
```

#### Issue: Migration Conflicts

**Symptoms:**
- Migration errors during deployment
- Inconsistent database schema
- Django admin not working

**Diagnosis:**
```bash
# Check migration status
python manage.py showmigrations

# Check for conflicts
python manage.py check
```

**Solutions:**

1. **Reset Migrations (Development Only):**
```bash
# Delete migration files (keep __init__.py)
find . -path "*/migrations/*.py" -not -name "__init__.py" -delete
find . -path "*/migrations/*.pyc" -delete

# Create fresh migrations
python manage.py makemigrations
python manage.py migrate
```

2. **Fake Migrations (Production):**
```bash
# Mark migrations as applied without running them
python manage.py migrate --fake-initial
```

## 🖥️ Application Issues

### Django Backend Problems

#### Issue: 500 Internal Server Error

**Symptoms:**
- API endpoints returning 500 errors
- White page in admin interface
- No detailed error messages

**Diagnosis:**
```bash
# Check Django logs
tail -f logs/fueloptimine.log

# Check error details
python backend/manage.py shell -c "
import django
django.setup()
# Test problematic code here
"

# Run with debug mode temporarily
DEBUG=True python backend/manage.py runserver
```

**Solutions:**

1. **Check Dependencies:**
```bash
pip install -r requirements.txt
pip check
```

2. **Clear Cache:**
```bash
# Clear Django cache
python backend/manage.py clearcache

# Clear Redis cache
redis-cli FLUSHALL
```

3. **Check Static Files:**
```bash
cd backend
python manage.py collectstatic --noinput
```

#### Issue: Celery Tasks Not Running

**Symptoms:**
- Background tasks not processing
- Reports not generating
- Real-time updates not working

**Diagnosis:**
```bash
# Check Celery worker status
celery -A config inspect active

# Check Celery logs
celery -A config events

# Monitor task queue
celery -A config flower  # Web interface on :5555
```

**Solutions:**

1. **Start Celery Worker:**
```bash
# Development
celery -A config worker -l info

# Production
systemctl start celery-worker
```

2. **Clear Task Queue:**
```bash
# Purge all tasks
celery -A config purge

# Or through Redis
redis-cli FLUSHDB
```

3. **Check Celery Configuration:**
```python
# In Django settings
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
```

### Streamlit Frontend Problems

#### Issue: Dashboard Not Loading

**Symptoms:**
- Blank page or loading spinner
- Connection errors
- Data not displaying

**Diagnosis:**
```bash
# Check Streamlit logs
streamlit logs

# Test connection to backend
curl http://localhost:8000/api/health/

# Check browser console for JavaScript errors
```

**Solutions:**

1. **Restart Streamlit:**
```bash
# Kill existing process
pkill -f streamlit

# Start fresh
streamlit run tests/dashboard.py --server.port 8501
```

2. **Clear Streamlit Cache:**
```bash
# Clear cache directory
rm -rf ~/.streamlit/cache/

# Or programmatically
import streamlit as st
st.cache_data.clear()
```

3. **Check API Configuration:**
```python
# In dashboard.py, verify API URL
API_BASE_URL = "http://localhost:8000/api/"

# Test API connectivity
import requests
response = requests.get(f"{API_BASE_URL}health/")
print(response.status_code, response.json())
```

#### Issue: Charts Not Rendering

**Symptoms:**
- Empty chart areas
- Plotly errors
- Data loading but no visualization

**Diagnosis:**
```python
# Test data retrieval
import pandas as pd
import plotly.graph_objects as go

# Check if data is valid
df = get_fuel_data()  # Your data function
print(df.head())
print(df.dtypes)
print(df.isnull().sum())

# Test simple chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1,2,3], y=[1,2,3]))
st.plotly_chart(fig)
```

**Solutions:**

1. **Update Dependencies:**
```bash
pip install --upgrade plotly streamlit pandas
```

2. **Fix Data Issues:**
```python
# Handle missing data
df = df.dropna()

# Ensure correct data types
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['fuel_level'] = pd.to_numeric(df['fuel_level'], errors='coerce')

# Check for empty datasets
if df.empty:
    st.warning("No data available for the selected period.")
    return
```

## 🔧 Performance Issues

### Slow Query Performance

#### Issue: Dashboard Loading Slowly

**Symptoms:**
- Long loading times
- Timeouts on data requests
- Unresponsive interface

**Diagnosis:**
```sql
-- Check ClickHouse query performance
SELECT 
    query,
    elapsed,
    read_rows,
    formatReadableSize(read_bytes) as read_bytes
FROM system.query_log 
WHERE event_time > now() - INTERVAL 1 HOUR
ORDER BY elapsed DESC
LIMIT 10;

-- Check slow queries in PostgreSQL
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

**Solutions:**

1. **Optimize ClickHouse Queries:**
```sql
-- Add appropriate indexes
ALTER TABLE fuel_readings ADD INDEX idx_equipment_time (equipment_id, timestamp) TYPE minmax GRANULARITY 3;

-- Use materialized views for aggregations
CREATE MATERIALIZED VIEW fuel_daily_mv TO fuel_daily_aggregates AS
SELECT 
    toDate(timestamp) as date,
    equipment_id,
    avg(fuel_level_liters) as avg_fuel_level
FROM fuel_readings
GROUP BY date, equipment_id;
```

2. **Implement Caching:**
```python
# In Django views
from django.core.cache import cache

def get_fuel_data(equipment_id, start_date, end_date):
    cache_key = f"fuel_data_{equipment_id}_{start_date}_{end_date}"
    data = cache.get(cache_key)
    
    if data is None:
        data = fetch_fuel_data(equipment_id, start_date, end_date)
        cache.set(cache_key, data, timeout=300)  # 5 minutes
    
    return data
```

3. **Optimize Frontend:**
```python
# Use Streamlit caching
@st.cache_data(ttl=300)  # 5 minutes
def load_fuel_data():
    return api_client.get_fuel_data()

# Implement pagination
page_size = 1000
offset = (page_number - 1) * page_size
data = get_paginated_data(offset, page_size)
```

### Memory Issues

#### Issue: High Memory Usage

**Symptoms:**
- Out of memory errors
- Server becoming unresponsive
- Killed processes

**Diagnosis:**
```bash
# Check memory usage
free -h
top -p $(pgrep -f streamlit)
top -p $(pgrep -f python)

# Check Django memory usage
python backend/manage.py shell -c "
import tracemalloc
tracemalloc.start()
# Run your code
current, peak = tracemalloc.get_traced_memory()
print(f'Current: {current / 1024 / 1024:.1f} MB')
print(f'Peak: {peak / 1024 / 1024:.1f} MB')
"
```

**Solutions:**

1. **Optimize Data Loading:**
```python
# Use chunked reading for large datasets
def read_large_dataset(file_path, chunk_size=10000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield process_chunk(chunk)

# Use generators instead of loading all data
def get_fuel_readings():
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM fuel_readings")
    while True:
        rows = cursor.fetchmany(1000)
        if not rows:
            break
        yield rows
```

2. **Implement Memory Limits:**
```python
# Set pandas memory limits
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.memory_usage', 'deep')

# Use more efficient data types
df['equipment_id'] = df['equipment_id'].astype('category')
df['fuel_level'] = pd.to_numeric(df['fuel_level'], downcast='float')
```

## 🔐 Security Issues

### Authentication Problems

#### Issue: JWT Token Errors

**Symptoms:**
- Login failures
- Unauthorized API responses
- Token expiration errors

**Diagnosis:**
```bash
# Test token generation
curl -X POST http://localhost:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "password": "password"}'

# Verify token
python -c "
import jwt
token = 'your-jwt-token-here'
try:
    decoded = jwt.decode(token, verify=False)
    print('Token payload:', decoded)
except Exception as e:
    print('Token error:', e)
"
```

**Solutions:**

1. **Refresh JWT Settings:**
```python
# In Django settings
from datetime import timedelta

SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(hours=1),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=7),
    'ROTATE_REFRESH_TOKENS': True,
}
```

2. **Clear Invalid Tokens:**
```bash
# Clear Redis token blacklist
redis-cli KEYS "*blacklist*" | xargs redis-cli DEL
```

### Permission Issues

#### Issue: Access Denied Errors

**Symptoms:**
- 403 Forbidden responses
- Users can't access features
- Admin interface blocked

**Diagnosis:**
```python
# Check user permissions
python backend/manage.py shell -c "
from django.contrib.auth.models import User
user = User.objects.get(username='your-username')
print('User groups:', user.groups.all())
print('User permissions:', user.user_permissions.all())
print('Is superuser:', user.is_superuser)
"
```

**Solutions:**

1. **Fix User Permissions:**
```python
# Add user to appropriate group
from django.contrib.auth.models import User, Group
user = User.objects.get(username='your-username')
group = Group.objects.get(name='operators')
user.groups.add(group)

# Grant specific permissions
from django.contrib.auth.models import Permission
permission = Permission.objects.get(codename='view_fuelreading')
user.user_permissions.add(permission)
```

## 🐳 Docker Issues

### Container Problems

#### Issue: Container Won't Start

**Symptoms:**
- Docker containers exiting immediately
- Build failures
- Service unavailable

**Diagnosis:**
```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs backend

# Check resource usage
docker stats

# Inspect container
docker inspect fueloptimine_backend_1
```

**Solutions:**

1. **Fix Docker Configuration:**
```bash
# Rebuild containers
docker-compose build --no-cache

# Restart services
docker-compose down
docker-compose up -d

# Check Docker daemon
sudo systemctl status docker
```

2. **Volume Issues:**
```bash
# Remove old volumes
docker-compose down -v

# Recreate volumes
docker volume create fueloptimine_postgres_data
docker volume create fueloptimine_clickhouse_data
```

#### Issue: Port Conflicts

**Symptoms:**
- Port already in use errors
- Cannot bind to port
- Service unreachable

**Diagnosis:**
```bash
# Check port usage
netstat -tlnp | grep :8000
netstat -tlnp | grep :8501
netstat -tlnp | grep :8123

# Check Docker port mapping
docker port fueloptimine_backend_1
```

**Solutions:**

1. **Change Ports:**
```yaml
# In docker-compose.yml
services:
  backend:
    ports:
      - "8001:8000"  # Change host port
```

2. **Kill Conflicting Processes:**
```bash
# Find and kill processes using the port
sudo fuser -k 8000/tcp
sudo fuser -k 8501/tcp
```

## 📊 Monitoring and Alerting

### Log Analysis

#### Common Error Patterns

**Database Connection Errors:**
```bash
# Search for database errors
grep -i "database" logs/fueloptimine.log | tail -20
grep -i "connection" logs/fueloptimine.log | tail -20
```

**Memory Errors:**
```bash
# Search for memory issues
grep -i "memory\|oom" logs/fueloptimine.log | tail -20
dmesg | grep -i "killed process"
```

**Performance Issues:**
```bash
# Find slow operations
grep -i "slow\|timeout" logs/fueloptimine.log | tail -20
```

### Automated Monitoring

**Health Check Script:**
```python
#!/usr/bin/env python3
import requests
import time
import smtplib
from email.mime.text import MIMEText

def send_alert(message):
    """Send email alert."""
    msg = MIMEText(message)
    msg['Subject'] = 'FuelOptiMine Alert'
    msg['From'] = 'alerts@fueloptimine.com'
    msg['To'] = 'admin@fueloptimine.com'
    
    server = smtplib.SMTP('localhost')
    server.send_message(msg)
    server.quit()

def check_health():
    """Check system health."""
    services = {
        'backend': 'http://localhost:8000/health/',
        'frontend': 'http://localhost:8501/_stcore/health',
        'clickhouse': 'http://localhost:8123/ping'
    }
    
    failed_services = []
    
    for service, url in services.items():
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                failed_services.append(f"{service}: {response.status_code}")
        except Exception as e:
            failed_services.append(f"{service}: {str(e)}")
    
    if failed_services:
        message = f"FuelOptiMine services failing:\n" + "\n".join(failed_services)
        send_alert(message)
        print("CRITICAL:", message)
        return False
    
    print("All services healthy")
    return True

if __name__ == "__main__":
    while True:
        check_health()
        time.sleep(300)  # Check every 5 minutes
```

## 🆘 Emergency Procedures

### System Down

1. **Immediate Response:**
```bash
# Check critical services
systemctl status clickhouse-server postgresql redis-server

# Restart if needed
sudo systemctl restart clickhouse-server
sudo systemctl restart postgresql
sudo systemctl restart redis-server

# Check application services
docker-compose ps
docker-compose restart
```

2. **Data Recovery:**
```bash
# PostgreSQL backup restore
pg_restore -h localhost -U postgres -d fueloptimine backup.sql

# ClickHouse backup restore
clickhouse-client --query "RESTORE TABLE fuel_readings FROM 'backup/'"
```

### Data Corruption

1. **Identify Corruption:**
```sql
-- Check ClickHouse data integrity
CHECK TABLE fuel_readings;

-- Verify PostgreSQL integrity
VACUUM (VERBOSE, ANALYZE);
```

2. **Recovery Steps:**
```bash
# Stop applications
docker-compose down

# Restore from backup
./scripts/restore_from_backup.sh

# Restart services
docker-compose up -d

# Verify data integrity
python scripts/verify_data.py
```

## 📞 Getting Help

### Internal Resources

1. **Documentation:** Check this troubleshooting guide and API docs
2. **Logs:** Always check application and system logs first
3. **Health Checks:** Run automated health check scripts
4. **Team Knowledge:** Consult with development team

### External Support

1. **GitHub Issues:** Report bugs and issues
2. **Community Forums:** Ask questions in discussions
3. **Professional Support:** Contact for enterprise support

### Creating Support Tickets

Include this information:
- **Environment:** Development/Staging/Production
- **Version:** Application version and commit hash
- **Error Messages:** Exact error messages and stack traces
- **Steps to Reproduce:** Clear steps that led to the issue
- **System Information:** OS, Python version, dependencies
- **Logs:** Relevant log entries with timestamps

**Template:**
```
**Environment:** Production
**Version:** v1.2.3 (commit abc123)
**Issue:** Dashboard not loading fuel data

**Error Message:**
```
ConnectionError: Unable to connect to ClickHouse at localhost:8123
```

**Steps to Reproduce:**
1. Navigate to dashboard
2. Select equipment TRUCK_001
3. Set date range to last 7 days
4. Click "Load Data"

**System Info:**
- OS: Ubuntu 22.04
- Python: 3.10.8
- Dependencies: requirements.txt

**Logs:**
```
2024-07-25 18:30:15 ERROR [analytics.views] ClickHouse connection failed
```
```

This troubleshooting guide provides comprehensive coverage of common issues and their solutions, helping users quickly diagnose and resolve problems with FuelOptiMine.