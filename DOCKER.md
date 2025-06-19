# Voice Agent Docker Deployment Guide

This guide covers how to deploy the Voice Agent application using Docker containers for production and development environments.

## 📋 Prerequisites

- **Docker** 20.10+ and **Docker Compose** 2.0+
- **Make** (optional, for using shortcuts)
- At least **4GB RAM** and **2 CPU cores**
- **API Keys**: Google Gemini API key (required)

## 🚀 Quick Start

### 1. Initial Setup

```bash
# Clone the repository
git clone <repository-url>
cd Voice-Agent

# Set up environment
make setup
# OR manually:
cp env.example .env
mkdir -p logs uploads ssl
```

### 2. Configure Environment

Edit the `.env` file with your configuration:

```bash
# Required: Add your Gemini API key
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Optional: Customize passwords
POSTGRES_PASSWORD=your_secure_password
JWT_SECRET=your_jwt_secret_key
```

### 3. Start the Application

**Production:**
```bash
make build && make up
# OR
docker-compose build && docker-compose up -d
```

**Development:**
```bash
make build-dev && make up-dev
# OR
docker-compose -f docker-compose.dev.yml build && docker-compose -f docker-compose.dev.yml up -d
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **Nginx Proxy**: http://localhost:80 (production only)
- **Database**: localhost:5432
- **Redis**: localhost:6379

## 🏗️ Architecture

### Container Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Nginx Proxy   │    │   Frontend      │    │   Backend       │
│   (Port 80/443) │    │   (Port 3000)   │    │   (Port 8000)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
         ┌─────────────────┐    ┌─────────────────┐
         │   PostgreSQL    │    │     Redis       │
         │   (Port 5432)   │    │   (Port 6379)   │
         └─────────────────┘    └─────────────────┘
```

### Services

1. **Frontend** (`voice-agent-frontend`)
   - Next.js application with 3D audio visualization
   - Serves the web interface
   - Built with production optimizations

2. **Backend** (`voice-agent-backend`)
   - Python FastAPI server
   - Handles voice processing and AI interactions
   - WebSocket support for real-time communication

3. **Database** (`voice-agent-db`)
   - PostgreSQL 15 with voice agent schema
   - Stores user data and conversation history

4. **Redis** (`voice-agent-redis`)
   - Caching and session management
   - Message queuing for real-time features

5. **Nginx** (`voice-agent-nginx`) - Production only
   - Reverse proxy and load balancer
   - SSL termination
   - Rate limiting and security

## 🛠️ Development Setup

### Development vs Production

| Feature | Development | Production |
|---------|-------------|------------|
| Hot Reload | ✅ Yes | ❌ No |
| Source Maps | ✅ Yes | ❌ No |
| Volume Mounts | ✅ Code mounted | ❌ Code copied |
| SSL | ❌ HTTP only | ✅ HTTPS ready |
| Optimization | ❌ Debug mode | ✅ Minified |
| Database | Local PostgreSQL | Production-ready |

### Development Commands

```bash
# Start development environment
make up-dev

# Watch logs
make logs-dev

# Access containers
make backend-shell   # Python backend
make frontend-shell  # Node.js frontend
make db-shell       # PostgreSQL

# Restart specific service
docker-compose -f docker-compose.dev.yml restart backend
```

### Frontend Development

The frontend runs in development mode with hot reload:

```bash
# Access the frontend container
make frontend-shell

# Install new dependencies
npm install package-name

# Run specific commands
npm run lint
npm run build
```

### Backend Development

The backend supports hot reload for Python files:

```bash
# Access the backend container
make backend-shell

# Install new Python packages
pip install package-name

# Run tests
python -m pytest

# Check logs
tail -f /app/logs/voice-agent.log
```

## 🚀 Production Deployment

### Environment Configuration

Create a production `.env` file:

```bash
# Production environment
NODE_ENV=production
POSTGRES_PASSWORD=secure_production_password
JWT_SECRET=secure_random_jwt_secret_key

# API Keys
GEMINI_API_KEY=your_production_gemini_key

# SSL (if using HTTPS)
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem

# Performance
MAX_CONCURRENT_CONNECTIONS=500
WORKER_PROCESSES=4
```

### SSL/TLS Setup

For HTTPS in production:

1. **Obtain SSL certificates:**
   ```bash
   # Using Let's Encrypt
   certbot certonly --webroot -w /var/www/html -d yourdomain.com
   
   # Copy certificates
   mkdir -p ssl
   cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ssl/cert.pem
   cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ssl/key.pem
   ```

2. **Update nginx.conf:**
   ```nginx
   # Uncomment SSL configuration lines
   ssl_certificate /etc/nginx/ssl/cert.pem;
   ssl_certificate_key /etc/nginx/ssl/key.pem;
   ```

3. **Restart services:**
   ```bash
   make restart
   ```

### Performance Optimization

**Backend Scaling:**
```yaml
backend:
  deploy:
    replicas: 3
    resources:
      limits:
        cpus: '2'
        memory: 2G
```

**Database Optimization:**
```yaml
postgres:
  environment:
    POSTGRES_SHARED_PRELOAD_LIBRARIES: pg_stat_statements
    POSTGRES_MAX_CONNECTIONS: 200
  command: >
    postgres
    -c shared_buffers=256MB
    -c effective_cache_size=1GB
```

## 📊 Monitoring & Health Checks

### Health Check Endpoints

- **Frontend**: `GET /api/health`
- **Backend**: `GET /health`
- **Database**: `pg_isready` command

### Monitoring Commands

```bash
# Check all services health
make health

# View resource usage
docker stats

# Check logs
make logs

# Monitor specific service
docker-compose logs -f backend
```

### Log Management

Logs are stored in `./logs/` directory:

- `voice-agent.log` - Application logs
- `nginx/access.log` - Web server access logs
- `nginx/error.log` - Web server error logs

## 🔧 Troubleshooting

### Common Issues

**1. Frontend not loading:**
```bash
# Check if frontend container is running
docker-compose ps frontend

# Check logs
docker-compose logs frontend

# Restart frontend
docker-compose restart frontend
```

**2. Backend API errors:**
```bash
# Check backend logs
docker-compose logs backend

# Verify environment variables
docker-compose exec backend env | grep -E '(GEMINI|DATABASE)'

# Test API directly
curl http://localhost:8000/health
```

**3. Database connection issues:**
```bash
# Check database status
docker-compose exec postgres pg_isready -U voice_agent

# Connect to database
make db-shell

# Check database logs
docker-compose logs postgres
```

**4. Audio/WebSocket issues:**
```bash
# Check browser console for WebSocket errors
# Verify CORS settings in backend
# Test WebSocket connection:
curl -H "Upgrade: websocket" http://localhost:8000/ws
```

### Performance Issues

**High memory usage:**
```bash
# Check container resources
docker stats

# Optimize Docker settings
echo '{"log-driver": "json-file", "log-opts": {"max-size": "10m", "max-file": "3"}}' > /etc/docker/daemon.json
```

**Slow startup:**
```bash
# Pre-build images
make build

# Use Docker build cache
docker-compose build --parallel
```

## 🔒 Security

### Production Security Checklist

- [ ] Change default passwords
- [ ] Use strong JWT secrets
- [ ] Enable SSL/TLS
- [ ] Configure firewall rules
- [ ] Regular security updates
- [ ] Backup encryption
- [ ] Rate limiting enabled
- [ ] CORS properly configured

### Security Commands

```bash
# Security scan
make security-scan

# Update base images
docker-compose pull

# Remove unused images
docker system prune -a
```

## 💾 Backup & Recovery

### Database Backup

```bash
# Create backup
make db-backup

# Restore from backup
make db-restore
```

### Full System Backup

```bash
# Backup volumes
docker run --rm -v voice-agent_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data

# Backup configuration
tar czf config_backup.tar.gz .env docker-compose.yml nginx.conf
```

## 🔄 Updates & Maintenance

### Updating the Application

```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
make down
make build
make up

# Or rolling update (zero downtime)
docker-compose up -d --no-deps backend
docker-compose up -d --no-deps frontend
```

### Maintenance Tasks

```bash
# Clean up unused resources
make clean

# Update dependencies
docker-compose pull
docker-compose build --no-cache

# Check for security updates
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock -v $(pwd):/root/.cache/ aquasec/trivy image voice-agent-backend
```

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Next.js Docker Guide](https://nextjs.org/docs/deployment#docker-image)
- [FastAPI Docker Guide](https://fastapi.tiangolo.com/deployment/docker/)
- [PostgreSQL Docker Guide](https://hub.docker.com/_/postgres)

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review container logs: `make logs`
3. Verify your `.env` configuration
4. Check resource usage: `docker stats`
5. Consult the main project README.md 