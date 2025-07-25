# Deployment Guide

This guide provides comprehensive instructions for deploying FuelOptiMine in different environments, from development to production.

## 🎯 Deployment Overview

FuelOptiMine supports multiple deployment strategies:

- **Development**: Local development environment
- **Staging**: Pre-production testing environment  
- **Production**: Full production deployment
- **Docker**: Containerized deployment
- **Kubernetes**: Orchestrated container deployment
- **Cloud**: AWS, GCP, Azure deployment

## 🖥️ Local Development Deployment

### Quick Start

```bash
# Clone repository
git clone https://github.com/JoseIgnacioFernandezMamani/FuelOptiMine.git
cd FuelOptiMine

# Setup environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Initialize databases
python backend/manage.py migrate
python scripts/init_clickhouse.py

# Start services
make dev-start
```

### Manual Service Start

```bash
# Terminal 1: Django Backend
cd backend
python manage.py runserver 0.0.0.0:8000

# Terminal 2: Streamlit Frontend
streamlit run tests/dashboard.py --server.port 8501

# Terminal 3: Celery Worker
celery -A config worker -l info

# Terminal 4: Kedro Pipeline (optional)
cd core/orchestration
kedro run
```

## 🐳 Docker Deployment

### Docker Compose Setup

**`docker-compose.yml`**

```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: fueloptimine
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_postgres.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ClickHouse Database
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    environment:
      CLICKHOUSE_DB: fuel_optimine
      CLICKHOUSE_USER: default
      CLICKHOUSE_PASSWORD: ""
    volumes:
      - clickhouse_data:/var/lib/clickhouse
      - ./scripts/init_clickhouse.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "8123:8123"
      - "9000:9000"
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:8123/ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Django Backend
  backend:
    build:
      context: .
      dockerfile: docker/backend/Dockerfile
      target: development
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/fueloptimine
      - CLICKHOUSE_HOST=clickhouse
      - REDIS_URL=redis://redis:6379/0
      - DEBUG=True
    volumes:
      - ./backend:/app/backend
      - ./config:/app/config
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      clickhouse:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Streamlit Frontend
  frontend:
    build:
      context: .
      dockerfile: docker/frontend/Dockerfile
    environment:
      - API_URL=http://backend:8000
    volumes:
      - ./tests:/app/tests
      - ./docs:/app/docs
    ports:
      - "8501:8501"
    depends_on:
      backend:
        condition: service_healthy

  # Celery Worker
  celery-worker:
    build:
      context: .
      dockerfile: docker/backend/Dockerfile
      target: development
    command: celery -A config worker -l info
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/fueloptimine
      - CLICKHOUSE_HOST=clickhouse
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./backend:/app/backend
      - ./config:/app/config
      - ./logs:/app/logs
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

  # Celery Beat (Scheduler)
  celery-beat:
    build:
      context: .
      dockerfile: docker/backend/Dockerfile
      target: development
    command: celery -A config beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/fueloptimine
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./backend:/app/backend
      - ./config:/app/config
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy

volumes:
  postgres_data:
  clickhouse_data:
  redis_data:
```

### Dockerfile Examples

**`docker/backend/Dockerfile`**

```dockerfile
# Multi-stage build for Django backend
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .

# Create logs directory
RUN mkdir -p logs

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

CMD ["python", "backend/manage.py", "runserver", "0.0.0.0:8000"]

# Production stage
FROM base as production

COPY . .

# Collect static files
RUN python backend/manage.py collectstatic --noinput

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--chdir", "backend", "config.wsgi:application"]
```

**`docker/frontend/Dockerfile`**

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Streamlit and dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir streamlit pandas plotly

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "tests/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Commands

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale celery-worker=3

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Build specific service
docker-compose build backend

# Execute commands in running container
docker-compose exec backend python manage.py shell
```

## ☸️ Kubernetes Deployment

### Namespace and ConfigMap

**`k8s/namespace.yaml`**

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: fueloptimine
```

**`k8s/configmap.yaml`**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fueloptimine-config
  namespace: fueloptimine
data:
  ENVIRONMENT: "production"
  DEBUG: "False"
  CLICKHOUSE_HOST: "clickhouse-service"
  CLICKHOUSE_HTTP_PORT: "8123"
  CLICKHOUSE_DB: "fuel_optimine"
  REDIS_HOST: "redis-service"
  LOG_LEVEL: "INFO"
```

### Secrets

**`k8s/secrets.yaml`**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: fueloptimine-secrets
  namespace: fueloptimine
type: Opaque
data:
  # Base64 encoded values
  SECRET_KEY: <base64-encoded-secret>
  DATABASE_URL: <base64-encoded-db-url>
  CLICKHOUSE_PASSWORD: <base64-encoded-password>
```

### Database Deployments

**`k8s/postgres.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: fueloptimine
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: fueloptimine
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: fueloptimine-secrets
              key: postgres-password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: fueloptimine
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: fueloptimine
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

### Application Deployment

**`k8s/backend.yaml`**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fueloptimine-backend
  namespace: fueloptimine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fueloptimine-backend
  template:
    metadata:
      labels:
        app: fueloptimine-backend
    spec:
      containers:
      - name: backend
        image: fueloptimine/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: fueloptimine-secrets
              key: SECRET_KEY
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: fueloptimine-secrets
              key: DATABASE_URL
        envFrom:
        - configMapRef:
            name: fueloptimine-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health/
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: fueloptimine-backend-service
  namespace: fueloptimine
spec:
  selector:
    app: fueloptimine-backend
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

### Ingress Configuration

**`k8s/ingress.yaml`**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: fueloptimine-ingress
  namespace: fueloptimine
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.fueloptimine.com
    - app.fueloptimine.com
    secretName: fueloptimine-tls
  rules:
  - host: api.fueloptimine.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fueloptimine-backend-service
            port:
              number: 8000
  - host: app.fueloptimine.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: fueloptimine-frontend-service
            port:
              number: 8501
```

### Horizontal Pod Autoscaler

**`k8s/hpa.yaml`**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fueloptimine-backend-hpa
  namespace: fueloptimine
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fueloptimine-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## ☁️ Cloud Deployment

### AWS Deployment

#### ECS with Fargate

**`aws/ecs-task-definition.json`**

```json
{
  "family": "fueloptimine-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "your-ecr-repo/fueloptimine-backend:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "SECRET_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:fueloptimine/secret-key"
        },
        {
          "name": "DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:fueloptimine/database-url"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fueloptimine",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health/ || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### CloudFormation Template

**`aws/infrastructure.yaml`**

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'FuelOptiMine Infrastructure'

Parameters:
  Environment:
    Type: String
    Default: production
    AllowedValues: [development, staging, production]

Resources:
  # VPC and Networking
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: 10.0.0.0/16
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-fueloptimine-vpc"

  # RDS PostgreSQL
  DatabaseSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupDescription: Subnet group for FuelOptiMine database
      SubnetIds:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
      Tags:
        - Key: Name
          Value: !Sub "${Environment}-fueloptimine-db-subnet-group"

  Database:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceIdentifier: !Sub "${Environment}-fueloptimine-db"
      DBInstanceClass: db.t3.medium
      Engine: postgres
      EngineVersion: '15.4'
      AllocatedStorage: 100
      StorageType: gp2
      DBName: fueloptimine
      MasterUsername: postgres
      MasterUserPassword: !Ref DatabasePassword
      VPCSecurityGroups:
        - !Ref DatabaseSecurityGroup
      DBSubnetGroupName: !Ref DatabaseSubnetGroup
      BackupRetentionPeriod: 7
      MultiAZ: !If [IsProduction, true, false]
      StorageEncrypted: true

  # ElastiCache Redis
  RedisSubnetGroup:
    Type: AWS::ElastiCache::SubnetGroup
    Properties:
      Description: Subnet group for Redis cluster
      SubnetIds:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2

  RedisCluster:
    Type: AWS::ElastiCache::CacheCluster
    Properties:
      CacheNodeType: cache.t3.micro
      Engine: redis
      NumCacheNodes: 1
      CacheSubnetGroupName: !Ref RedisSubnetGroup
      VpcSecurityGroupIds:
        - !Ref RedisSecurityGroup

  # ECS Cluster
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub "${Environment}-fueloptimine"
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT

  # Application Load Balancer
  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: !Sub "${Environment}-fueloptimine-alb"
      Scheme: internet-facing
      Type: application
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref LoadBalancerSecurityGroup

Conditions:
  IsProduction: !Equals [!Ref Environment, production]

Outputs:
  DatabaseEndpoint:
    Description: RDS Database Endpoint
    Value: !GetAtt Database.Endpoint.Address
    Export:
      Name: !Sub "${Environment}-fueloptimine-db-endpoint"

  RedisEndpoint:
    Description: Redis Cluster Endpoint
    Value: !GetAtt RedisCluster.RedisEndpoint.Address
    Export:
      Name: !Sub "${Environment}-fueloptimine-redis-endpoint"
```

### GCP Deployment

#### Cloud Run Service

**`gcp/service.yaml`**

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: fueloptimine-backend
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
        autoscaling.knative.dev/minScale: "1"
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/execution-environment: gen2
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
      containers:
      - image: gcr.io/project-id/fueloptimine-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: production
        - name: SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: fueloptimine-secrets
              key: secret-key
        resources:
          limits:
            cpu: "2"
            memory: "2Gi"
          requests:
            cpu: "1"
            memory: "1Gi"
        livenessProbe:
          httpGet:
            path: /health/
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

## 🚀 Production Deployment

### Production Checklist

#### Pre-deployment

- [ ] **Security Review**
  - [ ] Secret keys rotated
  - [ ] SSL certificates configured
  - [ ] Firewall rules configured
  - [ ] Database credentials secured
  - [ ] API rate limiting enabled

- [ ] **Performance Testing**
  - [ ] Load testing completed
  - [ ] Database queries optimized
  - [ ] Caching configured
  - [ ] CDN setup for static files

- [ ] **Monitoring Setup**
  - [ ] Application monitoring (Sentry)
  - [ ] Infrastructure monitoring (Prometheus)
  - [ ] Log aggregation (ELK stack)
  - [ ] Alerting configured

- [ ] **Backup Strategy**
  - [ ] Database backups automated
  - [ ] File storage backups
  - [ ] Disaster recovery plan

#### Deployment Steps

```bash
# 1. Build production images
docker build -t fueloptimine/backend:v1.0.0 -f docker/backend/Dockerfile --target production .
docker build -t fueloptimine/frontend:v1.0.0 -f docker/frontend/Dockerfile .

# 2. Push to registry
docker push fueloptimine/backend:v1.0.0
docker push fueloptimine/frontend:v1.0.0

# 3. Update Kubernetes manifests
kubectl apply -f k8s/

# 4. Perform rolling update
kubectl set image deployment/fueloptimine-backend backend=fueloptimine/backend:v1.0.0 -n fueloptimine

# 5. Monitor deployment
kubectl rollout status deployment/fueloptimine-backend -n fueloptimine

# 6. Run post-deployment tests
kubectl run test-pod --image=fueloptimine/test:latest --rm -it -- python test_production.py

# 7. Update DNS (if needed)
# Update DNS records to point to new load balancer
```

### Zero-downtime Deployment

**`scripts/deploy.sh`**

```bash
#!/bin/bash
set -e

VERSION=$1
ENVIRONMENT=${2:-production}

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version> [environment]"
    exit 1
fi

echo "Deploying FuelOptiMine $VERSION to $ENVIRONMENT"

# 1. Pre-deployment checks
echo "Running pre-deployment checks..."
python scripts/check_deployment.py --environment $ENVIRONMENT

# 2. Database migrations
echo "Running database migrations..."
kubectl exec deployment/fueloptimine-backend -n fueloptimine -- python manage.py migrate

# 3. Rolling update
echo "Performing rolling update..."
kubectl set image deployment/fueloptimine-backend backend=fueloptimine/backend:$VERSION -n fueloptimine
kubectl set image deployment/fueloptimine-frontend frontend=fueloptimine/frontend:$VERSION -n fueloptimine

# 4. Wait for rollout
echo "Waiting for rollout to complete..."
kubectl rollout status deployment/fueloptimine-backend -n fueloptimine --timeout=600s
kubectl rollout status deployment/fueloptimine-frontend -n fueloptimine --timeout=600s

# 5. Health checks
echo "Running health checks..."
sleep 30
python scripts/health_check.py --environment $ENVIRONMENT

# 6. Smoke tests
echo "Running smoke tests..."
python scripts/smoke_tests.py --environment $ENVIRONMENT

echo "Deployment completed successfully!"
```

### Rollback Strategy

```bash
#!/bin/bash
# rollback.sh

ENVIRONMENT=${1:-production}

echo "Rolling back FuelOptiMine in $ENVIRONMENT"

# Get previous revision
BACKEND_REVISION=$(kubectl rollout history deployment/fueloptimine-backend -n fueloptimine | tail -2 | head -1 | awk '{print $1}')
FRONTEND_REVISION=$(kubectl rollout history deployment/fueloptimine-frontend -n fueloptimine | tail -2 | head -1 | awk '{print $1}')

# Rollback deployments
kubectl rollout undo deployment/fueloptimine-backend -n fueloptimine --to-revision=$BACKEND_REVISION
kubectl rollout undo deployment/fueloptimine-frontend -n fueloptimine --to-revision=$FRONTEND_REVISION

# Wait for rollback
kubectl rollout status deployment/fueloptimine-backend -n fueloptimine
kubectl rollout status deployment/fueloptimine-frontend -n fueloptimine

# Health check
python scripts/health_check.py --environment $ENVIRONMENT

echo "Rollback completed!"
```

## 📊 Monitoring and Maintenance

### Health Check Scripts

**`scripts/health_check.py`**

```python
#!/usr/bin/env python3
import requests
import sys
import time
import argparse

def check_backend_health(base_url):
    """Check backend API health."""
    try:
        response = requests.get(f"{base_url}/health/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'healthy':
                print("✅ Backend health check passed")
                return True
            else:
                print(f"❌ Backend unhealthy: {data}")
                return False
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend health check error: {e}")
        return False

def check_frontend_health(base_url):
    """Check frontend health."""
    try:
        response = requests.get(f"{base_url}/_stcore/health", timeout=10)
        if response.status_code == 200:
            print("✅ Frontend health check passed")
            return True
        else:
            print(f"❌ Frontend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend health check error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Health check for FuelOptiMine')
    parser.add_argument('--environment', default='production', help='Environment to check')
    args = parser.parse_args()
    
    if args.environment == 'production':
        backend_url = "https://api.fueloptimine.com"
        frontend_url = "https://app.fueloptimine.com"
    else:
        backend_url = "http://localhost:8000"
        frontend_url = "http://localhost:8501"
    
    print(f"Checking {args.environment} environment health...")
    
    backend_healthy = check_backend_health(backend_url)
    frontend_healthy = check_frontend_health(frontend_url)
    
    if backend_healthy and frontend_healthy:
        print("✅ All services healthy")
        sys.exit(0)
    else:
        print("❌ Some services unhealthy")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Automated Deployment Pipeline

**`.github/workflows/deploy.yml`**

```yaml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.10
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    - name: Run tests
      run: pytest
    - name: Run linting
      run: |
        black --check .
        flake8 .

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    - name: Build and push backend
      uses: docker/build-push-action@v4
      with:
        context: .
        file: docker/backend/Dockerfile
        target: production
        push: true
        tags: ghcr.io/${{ github.repository }}/backend:${{ github.ref_name }}
    - name: Build and push frontend
      uses: docker/build-push-action@v4
      with:
        context: .
        file: docker/frontend/Dockerfile
        push: true
        tags: ghcr.io/${{ github.repository }}/frontend:${{ github.ref_name }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production
    steps:
    - uses: actions/checkout@v3
    - name: Configure kubectl
      uses: azure/k8s-set-context@v3
      with:
        method: kubeconfig
        kubeconfig: ${{ secrets.KUBE_CONFIG }}
    - name: Deploy to Kubernetes
      run: |
        # Update image tags in manifests
        sed -i "s|image: fueloptimine/backend:.*|image: ghcr.io/${{ github.repository }}/backend:${{ github.ref_name }}|" k8s/backend.yaml
        sed -i "s|image: fueloptimine/frontend:.*|image: ghcr.io/${{ github.repository }}/frontend:${{ github.ref_name }}|" k8s/frontend.yaml
        
        # Apply manifests
        kubectl apply -f k8s/
        
        # Wait for rollout
        kubectl rollout status deployment/fueloptimine-backend -n fueloptimine --timeout=600s
        kubectl rollout status deployment/fueloptimine-frontend -n fueloptimine --timeout=600s
    - name: Run health checks
      run: python scripts/health_check.py --environment production
```

This deployment guide provides comprehensive coverage of all deployment scenarios for FuelOptiMine, from local development to production-ready Kubernetes deployments with monitoring, health checks, and automated CI/CD pipelines.