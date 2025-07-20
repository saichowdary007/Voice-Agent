# Implementation Plan

- [x] 1. Fix WebSocket Protocol Alignment
  - Update frontend WebSocket connection to support both "binary" and "stream-audio" protocols
  - Modify backend WebSocket handler to accept multiple sub-protocols instead of rejecting non-"binary" connections
  - Implement proper protocol negotiation logic in websocket_handlers.py
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [ ] 2. Implement Heartbeat System
  - [x] 2.1 Add automatic pong response handling in frontend WebSocket
    - Modify useWebSocket.ts to properly handle ping/pong messages from server
    - Enable automatic pong responses to prevent 1011 timeout errors
    - Add connection health tracking with missed pong detection
    - _Requirements: 1.3, 1.4_

  - [x] 2.2 Configure server-side ping settings
    - Update FastAPI WebSocket configuration to use appropriate ping intervals
    - Add option to disable server pings if client handles heartbeat differently
    - Implement heartbeat acknowledgment system in websocket_handlers.py
    - _Requirements: 1.3, 1.4_

- [ ] 3. Standardize Audio Format Configuration
  - [x] 3.1 Update frontend MediaRecorder configuration
    - Modify VoiceInterface.tsx and AudioVisualizer.tsx to use mono 16kHz audio recording
    - Set MediaRecorder to use "audio/webm;codecs=opus" with 250ms chunk intervals
    - Configure channelCount=1 and sampleRate=16000 in MediaRecorder options
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 3.2 Implement audio format validation
    - Add client-side validation to ensure audio chunks meet format requirements
    - Create utility functions to verify audio chunk size and format compliance
    - Add error handling for unsupported audio formats
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 4. Optimize Audio Processing Pipeline
  - [x] 4.1 Implement streaming ffmpeg processor
    - Modify audio_preprocessor.py to launch ffmpeg once per WebSocket connection
    - Create StreamingAudioProcessor class that maintains persistent ffmpeg process
    - Implement stdin streaming instead of spawning new processes for each blob
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 4.2 Add audio processing performance monitoring
    - Implement CPU usage tracking during audio conversion
    - Add metrics for ffmpeg process lifecycle and performance
    - Create alerts for audio processing bottlenecks that could cause ping timeouts
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 6.2, 6.4_

- [ ] 5. Implement Persistent VAD Management
  - [x] 5.1 Create session-based VAD lifecycle
    - Modify websocket_handlers.py to maintain one VAD instance per WebSocket session
    - Implement WebSocketSessionManager class to track VAD instances by connection
    - Prevent VAD reset on reconnection unless connection is deliberately closed
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 5.2 Enhance VAD speech boundary detection
    - Update VAD processing to handle contiguous audio streams without constant re-initialization
    - Implement proper speech/silence detection with configurable thresholds
    - Add VAD state preservation during temporary connection issues
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 6. Implement Connection Recovery System
  - [x] 6.1 Add exponential backoff reconnection logic
    - Update useWebSocket.ts to implement proper exponential backoff (3s, 5s, 8s, 12s, 20s)
    - Add connection state preservation during reconnection attempts
    - Implement maximum retry limits with user notification
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [x] 6.2 Create connection health monitoring
    - Implement ConnectionHealthMonitor class to track WebSocket connection metrics
    - Add real-time monitoring of connection stability and performance
    - Create diagnostic tools to identify connection issues before they cause failures
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4_

- [ ] 7. Add Comprehensive Error Handling
  - [x] 7.1 Implement WebSocket error classification and recovery
    - Create WebSocketErrorHandler class to categorize different error types (1011, 1008, 1006, etc.)
    - Add specific recovery strategies for each error code
    - Implement graceful degradation when recovery is not possible
    - _Requirements: 5.4, 6.4_

  - [x] 7.2 Add audio processing error recovery
    - Implement error handling for ffmpeg process failures
    - Add recovery mechanisms for VAD processing errors
    - Create fallback strategies when audio processing fails
    - _Requirements: 3.3, 3.4, 4.3, 4.4_

- [ ] 8. Create Performance Monitoring and Diagnostics
  - [x] 8.1 Implement real-time performance metrics
    - Add WebSocket connection latency tracking
    - Implement audio processing performance monitoring
    - Create VAD accuracy and response time metrics
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 8.2 Build diagnostic dashboard and logging
    - Create comprehensive logging for WebSocket events and audio processing
    - Implement diagnostic tools to analyze connection failures and audio issues
    - Add performance alerts for latency and processing bottlenecks
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ] 9. Write Comprehensive Tests
  - [x] 9.1 Create WebSocket protocol and connection tests
    - Write unit tests for protocol negotiation and heartbeat handling
    - Create integration tests for connection recovery and stability
    - Add load tests for concurrent WebSocket connections
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.1, 5.2, 5.3, 5.4_

  - [x] 9.2 Implement audio processing and VAD tests
    - Create unit tests for audio format validation and processing
    - Write integration tests for VAD lifecycle management
    - Add performance tests for audio processing latency
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ] 10. Integration and End-to-End Testing
  - [x] 10.1 Test complete voice interaction pipeline
    - Create end-to-end tests that verify the entire voice processing flow
    - Test connection recovery during active voice sessions
    - Validate that all fixes work together to achieve sub-3-second response times
    - _Requirements: All requirements_

  - [x] 10.2 Performance validation and optimization
    - Run load tests to ensure system stability under concurrent usage
    - Validate that 1011 errors are eliminated and connections remain stable
    - Measure and optimize end-to-end latency to meet performance targets
    - _Requirements: All requirements_