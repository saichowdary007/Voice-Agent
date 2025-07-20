# Requirements Document

## Introduction

The Voice-Agent system is experiencing critical WebSocket connection failures (1011 errors), audio dropouts, and VAD re-initialization loops that prevent reliable voice interaction. Based on forensic analysis comparing the working "loco" frontend with the current Voice-Agent implementation, we need to fix WebSocket contract mismatches, audio processing inefficiencies, and connection stability issues to achieve sub-3-second response times with reliable voice detection.

## Requirements

### Requirement 1: WebSocket Protocol Alignment

**User Story:** As a user, I want stable WebSocket connections without 1011 timeout errors, so that my voice interactions are not interrupted.

#### Acceptance Criteria

1. WHEN the frontend establishes a WebSocket connection THEN the endpoint path and sub-protocol SHALL match between client and server
2. WHEN the WebSocket connection is established THEN the server SHALL accept both "binary" and "stream-audio" sub-protocols
3. WHEN ping/pong heartbeat messages are exchanged THEN the client SHALL respond to server pings to prevent timeout disconnections
4. WHEN the WebSocket connection is active THEN it SHALL remain stable for extended voice sessions without 1011 errors

### Requirement 2: Audio Format Standardization

**User Story:** As a user, I want my voice input to be processed efficiently without causing system overload, so that I get fast responses.

#### Acceptance Criteria

1. WHEN audio is recorded THEN the MediaRecorder SHALL use mono channel audio at 16kHz sample rate
2. WHEN audio chunks are generated THEN they SHALL be 250ms duration to match optimal processing size
3. WHEN audio codec is selected THEN it SHALL use "audio/webm;codecs=opus" format for compatibility
4. WHEN audio is transmitted THEN the chunk size SHALL be approximately 4KB to prevent CPU overload

### Requirement 3: Audio Processing Pipeline Optimization

**User Story:** As a developer, I want efficient audio processing that doesn't block the event loop, so that WebSocket connections remain stable.

#### Acceptance Criteria

1. WHEN audio blobs are processed THEN ffmpeg SHALL be launched once per connection and stream stdin
2. WHEN audio conversion occurs THEN it SHALL not spawn new processes for every blob
3. WHEN audio processing happens THEN CPU usage SHALL not cause ping watchdog timeouts
4. WHEN large audio blobs are received THEN they SHALL be processed without blocking the event loop

### Requirement 4: VAD Lifecycle Management

**User Story:** As a user, I want consistent voice activity detection without resource leaks, so that speech recognition works reliably.

#### Acceptance Criteria

1. WHEN VAD is initialized THEN it SHALL be created once per connection session
2. WHEN WebSocket reconnection occurs THEN VAD SHALL not be reset unnecessarily
3. WHEN speech detection runs THEN VAD SHALL have enough contiguous audio to make accurate decisions
4. WHEN connection closes deliberately THEN VAD resources SHALL be properly cleaned up
5. WHEN VAD processes audio THEN it SHALL distinguish between speech and silence without constant re-initialization

### Requirement 5: Connection Stability and Recovery

**User Story:** As a user, I want automatic connection recovery that doesn't disrupt my voice session, so that I can have uninterrupted conversations.

#### Acceptance Criteria

1. WHEN WebSocket connection drops THEN the system SHALL attempt reconnection with exponential backoff
2. WHEN reconnection occurs THEN existing VAD and audio processing state SHALL be preserved where possible
3. WHEN connection is restored THEN audio streaming SHALL resume without requiring user intervention
4. WHEN multiple reconnection attempts fail THEN the user SHALL be notified with clear error messaging

### Requirement 6: Performance Monitoring and Diagnostics

**User Story:** As a developer, I want comprehensive monitoring of WebSocket and audio performance, so that I can identify and resolve issues quickly.

#### Acceptance Criteria

1. WHEN WebSocket events occur THEN they SHALL be logged with timestamps and connection details
2. WHEN audio processing metrics are collected THEN they SHALL include chunk size, processing time, and CPU usage
3. WHEN VAD state changes THEN transitions SHALL be logged for debugging purposes
4. WHEN performance issues are detected THEN alerts SHALL be generated with actionable information
5. WHEN diagnostics are run THEN they SHALL verify all components of the audio pipeline are functioning correctly