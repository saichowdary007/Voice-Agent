# Ultra-Fast Voice Agent - Production Deployment Guide

## 🎯 Overview
This guide covers deploying the ultra-fast voice agent optimized for **≤500ms speech-to-response latency** in production environments.

## 📊 Performance Targets

| Component | Target Time | Optimization Strategy |
|-----------|-------------|----------------------|
| Voice Detection | 50-100ms | Aggressive VAD (0.3 sensitivity) |
| Speech Recognition | 100-200ms | Tiny Whisper + streaming |
| LLM Processing | 150-250ms | Skip context + parallel processing |
| TTS Synthesis | 100-200ms | Streaming + fast voice |
| Audio Playback | 50-100ms | Low-latency audio pipeline |
| **Total** | **≤500ms** | **Production Target** |

## 🚀 Quick Deployment

### 1. Backend Deployment

```bash
# Clone and setup
git clone <repository>
cd Voice-Agent

# Configure environment
cp .env.example .env
# Edit .env with your production settings

# Deploy with auto-script
chmod +x deploy-ultra-fast.sh
./deploy-ultra-fast.sh production
```

### 2. Frontend Deployment (Vercel)

```bash
# Deploy to Vercel
cd frontend
npm install -g vercel
vercel --prod

# Set environment variables in Vercel dashboard:
# NEXT_PUBLIC_API_URL=https://your-backend-api.com
```

## 🏗️ Manual Deployment

### Backend (Docker)

#### Option A: Docker Compose (Recommended)
```bash
# Build and start services
docker-compose -f docker-compose.ultra-fast.yml up -d

# View logs
docker-compose -f docker-compose.ultra-fast.yml logs -f

# Health check
curl http://localhost:8000/health
```

#### Option B: Direct Docker
```bash
# Build image
docker build -f Dockerfile.ultra-fast -t ultra-fast-voice-agent .

# Run container
docker run -d \
  --name ultra-fast-voice-agent \
  -p 8000:8000 \
  -e ULTRA_FAST_MODE=true \
  -e ULTRA_FAST_TARGET_LATENCY_MS=500 \
  -e GEMINI_API_KEY=your_key \
  ultra-fast-voice-agent
```

### Frontend (Vercel)

1. **Connect Repository**: Link your GitHub repository to Vercel
2. **Configure Build Settings**:
   - Framework: Next.js
   - Build Command: `npm run build`
   - Output Directory: `.next`
3. **Environment Variables**:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-api.com
   NODE_ENV=production
   ```
4. **Deploy**: Vercel will auto-deploy on push to main branch

## 🔧 Configuration

### Backend Environment Variables

```bash
# Ultra-fast optimizations
ULTRA_FAST_MODE=true
ULTRA_FAST_TARGET_LATENCY_MS=500
ULTRA_FAST_PERFORMANCE_TRACKING=true
ULTRA_FAST_STT_MODEL=tiny
ULTRA_FAST_VAD_SENSITIVITY=0.3
ULTRA_FAST_TTS_STREAMING=true
ULTRA_FAST_TTS_RATE=+25%

# API Keys
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=your_supabase_url (optional)
SUPABASE_KEY=your_supabase_key (optional)

# Production settings
NODE_ENV=production
HOST=0.0.0.0
PORT=8000
```

### Frontend Environment Variables

```bash
# Vercel Environment Variables
NEXT_PUBLIC_API_URL=https://your-backend-domain.com
NODE_ENV=production
```

## 🌐 Infrastructure Options

### Cloud Providers

#### Google Cloud Platform
```bash
# Cloud Run deployment
gcloud run deploy ultra-fast-voice-agent \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 30s \
  --set-env-vars ULTRA_FAST_MODE=true
```

#### AWS
```bash
# ECS with Fargate
aws ecs create-service \
  --cluster ultra-fast-cluster \
  --service-name ultra-fast-voice-agent \
  --task-definition ultra-fast-voice-agent:1 \
  --desired-count 1 \
  --launch-type FARGATE
```

#### Azure
```bash
# Container Instances
az container create \
  --resource-group ultra-fast-rg \
  --name ultra-fast-voice-agent \
  --image ultra-fast-voice-agent:latest \
  --cpu 1 \
  --memory 2 \
  --ports 8000
```

### Self-Hosted

#### Requirements
- **CPU**: 2+ cores (Intel/AMD x64)
- **Memory**: 4GB+ RAM
- **Storage**: 10GB+ SSD
- **Network**: Low-latency connection
- **OS**: Ubuntu 20.04+ or Docker-compatible

#### Setup
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Deploy
./deploy-ultra-fast.sh
```

## 📈 Performance Monitoring

### Built-in Monitoring

#### Health Check Endpoint
```bash
# Basic health
curl http://localhost:8000/health

# Performance stats
curl http://localhost:8000/api/voice/performance
```

#### WebSocket Performance
```javascript
// Frontend performance tracking
const ws = new WebSocket('ws://localhost:8000/ws/voice/session123');
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === 'voice_response') {
    const stats = message.data.performance_stats;
    console.log(`Latency: ${stats.total_latency_ms}ms`);
  }
};
```

### External Monitoring

#### Prometheus Metrics
```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## 🔒 Security Considerations

### SSL/TLS Configuration
```nginx
# nginx.ultra-fast.conf
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    location / {
        proxy_pass http://ultra-fast-voice-backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /ws/ {
        proxy_pass http://ultra-fast-voice-backend:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### API Security
- Use HTTPS/WSS for all communications
- Implement rate limiting (Redis + nginx)
- Validate all input parameters
- Use environment variables for secrets
- Enable CORS only for trusted domains

## 🚨 Troubleshooting

### Common Issues

#### High Latency (>500ms)
```bash
# Check component performance
curl http://localhost:8000/api/voice/performance

# View detailed logs
docker-compose -f docker-compose.ultra-fast.yml logs -f ultra-fast-voice-backend

# Test individual components
python test_ultra_fast_performance.py
```

#### Audio Issues
```bash
# Check audio device access
docker exec -it ultra-fast-voice-backend ls -la /dev/snd/

# Test TTS synthesis
curl -X POST http://localhost:8000/api/voice/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "streaming": true}'
```

#### Memory Issues
```bash
# Monitor container resources
docker stats ultra-fast-voice-backend

# Optimize memory settings
docker update --memory=4g ultra-fast-voice-backend
```

### Performance Optimization

#### TTS Latency Issues
1. **Check Edge TTS**: Ensure stable internet connection
2. **Use Local TTS**: Switch to macOS `say` or Linux `espeak`
3. **Reduce Text Length**: Split long responses into chunks
4. **Pre-warm TTS**: Initialize synthesis engine at startup

#### STT Latency Issues
1. **Aggressive VAD**: Lower `ULTRA_FAST_VAD_SENSITIVITY` to 0.2
2. **Tiny Model**: Ensure using `tiny` Whisper model
3. **Audio Quality**: Use 16kHz mono audio input
4. **Real-time Processing**: Enable streaming transcription

## 📞 Support

### Performance Issues
If experiencing latency >500ms:
1. Run performance test: `python test_ultra_fast_performance.py`
2. Check component health: `curl http://localhost:8000/health`
3. Review optimization settings in `.env`
4. Monitor resource usage: `docker stats`

### Deployment Issues
1. Check prerequisites: Docker, Docker Compose, Node.js
2. Verify environment variables in `.env`
3. Review deployment logs: `docker-compose logs`
4. Test individual endpoints manually

## 🎯 Success Metrics

### Production Readiness Checklist
- [ ] Backend health check returns 200
- [ ] Average latency ≤500ms over 10 requests
- [ ] WebSocket connections stable
- [ ] Frontend deployed and accessible
- [ ] SSL/TLS configured (production)
- [ ] Monitoring setup (optional)
- [ ] Error handling verified
- [ ] Performance tests passing

### Expected Performance
- **Average Latency**: 300-500ms
- **Target Achievement**: 80%+ requests ≤500ms
- **Concurrent Users**: 10+ simultaneous sessions
- **Uptime**: 99.9% availability
- **Error Rate**: <1% failed requests

---

🎉 **Your ultra-fast voice agent is now production-ready with optimized 500ms latency!** 