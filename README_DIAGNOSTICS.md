# WebSocket Voice Pipeline Diagnostics

This comprehensive diagnostic suite provides tools to verify WebSocket connectivity and troubleshoot the voice interaction pipeline.

## Quick Start

### 1. Install Dependencies
```bash
pip install numpy websockets psutil aiohttp
```

### 2. Run Quick Health Check
```bash
python run_diagnostics.py --test quick
```

### 3. Run Comprehensive Diagnostics
```bash
python run_diagnostics.py --test comprehensive
```

### 4. Start Real-time Monitoring
```bash
python run_diagnostics.py --test monitor --duration 300
```

## Diagnostic Tools

### 1. WebSocket Connectivity (`websocket_diagnostics.py`)
- **Basic Connectivity**: Tests WebSocket endpoint accessibility
- **Authentication**: Verifies token-based authentication
- **Latency Measurement**: Measures round-trip times
- **Voice Pipeline**: Tests complete audio processing flow
- **System Resources**: Monitors CPU, memory, and network usage

### 2. Real-time Monitoring (`websocket_monitor.py`)
- **Live Dashboard**: Real-time connection and performance metrics
- **Connection Health**: Tracks connection success/failure rates
- **Response Times**: Monitors message processing latency
- **Pipeline Performance**: Tracks STT, LLM, and TTS processing times
- **Error Tracking**: Categorizes and counts different error types

### 3. Voice Pipeline Testing (`test_voice_pipeline.py`)
- **STT Accuracy**: Tests speech-to-text conversion accuracy
- **LLM Quality**: Evaluates response quality and timing
- **TTS Performance**: Tests text-to-speech synthesis
- **End-to-End**: Complete pipeline testing with quality assessment

## Test Results and Metrics

### Connection Metrics
- **Connection Time**: Time to establish WebSocket connection
- **Handshake Success**: WebSocket handshake completion rate
- **Authentication Time**: Token validation and connection setup
- **Connection Stability**: Connection drop and reconnection rates

### Latency Metrics
- **Round-trip Time**: Client → Server → Client message timing
- **STT Processing**: Audio → Text conversion time
- **LLM Processing**: Text → Response generation time
- **TTS Processing**: Text → Audio synthesis time
- **End-to-End**: Complete voice interaction latency

### Quality Metrics
- **STT Accuracy**: Transcription accuracy vs. expected text
- **Response Relevance**: LLM response quality assessment
- **Audio Quality**: TTS output size and timing analysis
- **Pipeline Success Rate**: Complete interaction success percentage

## Usage Examples

### Quick Health Check
```bash
# Basic connectivity and authentication test
python run_diagnostics.py --test quick --url http://localhost:8080
```

### Comprehensive Testing
```bash
# Full diagnostic suite including voice pipeline
python run_diagnostics.py --test comprehensive --url http://localhost:8080
```

### Real-time Monitoring
```bash
# Monitor for 10 minutes with live dashboard
python run_diagnostics.py --test monitor --duration 600 --url http://localhost:8080
```

### Custom Output Directory
```bash
# Save results to custom directory
python run_diagnostics.py --test comprehensive --output-dir ./my_results
```

## Interpreting Results

### Health Status Indicators

#### Connection Health
- **EXCELLENT**: < 1000ms average connection time
- **GOOD**: 1000-2000ms average connection time  
- **POOR**: > 2000ms average connection time

#### Pipeline Health
- **EXCELLENT**: < 3000ms average end-to-end latency
- **GOOD**: 3000-5000ms average end-to-end latency
- **POOR**: > 5000ms average end-to-end latency

#### Error Rates
- **LOW**: < 5% error rate
- **MODERATE**: 5-15% error rate
- **HIGH**: > 15% error rate

### Performance Targets

| Component | Target Latency | Acceptable Range |
|-----------|----------------|------------------|
| WebSocket Connect | < 500ms | < 1000ms |
| Authentication | < 200ms | < 500ms |
| STT Processing | < 1000ms | < 2000ms |
| LLM Processing | < 2000ms | < 5000ms |
| TTS Processing | < 1000ms | < 2000ms |
| **End-to-End** | **< 3000ms** | **< 5000ms** |

## Troubleshooting Guide

### Common Issues

#### WebSocket Connection Failures
```
❌ WebSocket connection failed: Connection refused
```
**Solutions:**
- Verify server is running on correct port
- Check firewall settings
- Confirm WebSocket endpoint URL

#### Authentication Errors
```
❌ Authentication failed: Invalid token
```
**Solutions:**
- Verify API keys are configured
- Check token expiration
- Confirm authentication endpoint is accessible

#### High Latency
```
⚠️ Average latency: 8000ms (target: <3000ms)
```
**Solutions:**
- Check network connectivity
- Monitor server resource usage
- Verify STT/LLM/TTS service health
- Consider optimizing model sizes

#### Audio Processing Failures
```
❌ STT processing failed: No transcription received
```
**Solutions:**
- Verify STT service is running
- Check audio format compatibility
- Confirm microphone permissions
- Test with known good audio samples

### Performance Optimization

#### Reduce Connection Time
- Use connection pooling
- Implement connection keep-alive
- Optimize authentication flow

#### Improve Pipeline Latency
- Use smaller/faster models (e.g., Whisper tiny)
- Implement streaming processing
- Optimize audio chunk sizes
- Use parallel processing where possible

#### Enhance Reliability
- Implement retry mechanisms
- Add circuit breakers
- Monitor and alert on error rates
- Use health checks and auto-recovery

## Output Files

All diagnostic results are saved as JSON files with timestamps:

- `quick_check_YYYYMMDD_HHMMSS.json` - Quick health check results
- `comprehensive_diagnostics_YYYYMMDD_HHMMSS.json` - Full diagnostic results
- `monitoring_session_YYYYMMDD_HHMMSS.json` - Real-time monitoring data
- `voice_pipeline_test_results_YYYYMMDD_HHMMSS.json` - Voice pipeline test results

### Sample Output Structure
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "test_type": "comprehensive",
  "overall_health": "GOOD",
  "overall_success_rate": 0.85,
  "websocket_diagnostics": {
    "test_summary": {
      "total_tests": 5,
      "passed_tests": 4,
      "failed_tests": 1
    }
  },
  "voice_pipeline_tests": {
    "stt_accuracy": {
      "average_accuracy": 0.92,
      "average_processing_time_ms": 850
    },
    "llm_quality": {
      "average_response_time_ms": 1200,
      "success_rate": 0.95
    },
    "tts_quality": {
      "average_synthesis_time_ms": 600,
      "success_rate": 0.90
    }
  }
}
```

## Integration with CI/CD

### Automated Testing
```bash
# Add to CI pipeline
python run_diagnostics.py --test quick --url $STAGING_URL
if [ $? -eq 0 ]; then
  echo "Health check passed"
else
  echo "Health check failed"
  exit 1
fi
```

### Performance Regression Detection
```bash
# Compare with baseline metrics
python run_diagnostics.py --test comprehensive --url $PRODUCTION_URL
# Parse results and compare with historical data
```

## Support and Troubleshooting

For additional support:

1. **Check Server Logs**: Review backend server logs for errors
2. **Network Analysis**: Use network tools to check connectivity
3. **Resource Monitoring**: Monitor CPU, memory, and disk usage
4. **Service Health**: Verify all dependent services are running

The diagnostic tools provide comprehensive insights into your voice agent's performance and can help identify bottlenecks and issues in the pipeline.