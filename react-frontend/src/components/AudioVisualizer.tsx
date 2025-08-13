import React, { useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';

// Feature flag: when false, use PCM16 streaming (recommended for Deepgram Agent)
const USE_MEDIA_RECORDER_STREAMING = false;

interface AudioVisualizerProps {
  /**
   * When true the audio element (if it exists) will be muted.
   */
  muted?: boolean;
  /**
   * When true, user is speaking and visualizer should animate
   */
  isUserSpeaking?: boolean;
  /**
   * Callback when microphone permission is granted or denied
   */
  onMicrophonePermission?: (granted: boolean) => void;
  /**
   * Callback when voice activity is detected
   */
  onVoiceActivity?: (isActive: boolean) => void;
}

/**
 * A full-screen Three.js powered 3-D audio visualizer rendered inside a React component.
 * The implementation is adapted from a standalone HTML demo the user provided.
 */
const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ 
  muted = false, 
  isUserSpeaking = false,
  onMicrophonePermission,
  onVoiceActivity
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const requestIdRef = useRef<number | null>(null);
  const soundRef = useRef<THREE.Audio | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const microphoneRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const opusStreamingActiveRef = useRef<boolean>(false);
  const finalizeNextOpusChunkRef = useRef<boolean>(false);
  const initializedRef = useRef(false);
  const cleanupRef = useRef<(() => void) | null>(null);

  // Stable callback references
  const stableMicrophonePermission = useCallback((granted: boolean) => {
    onMicrophonePermission?.(granted);
  }, [onMicrophonePermission]);

  const stableVoiceActivity = useCallback((isActive: boolean) => {
    onVoiceActivity?.(isActive);
  }, [onVoiceActivity]);

  useEffect(() => {
    // Prevent multiple initializations
    if (!containerRef.current || initializedRef.current) return;
    
    // Additional check to prevent React strict mode double initialization
    if (containerRef.current.children.length > 0) return;
    
    initializedRef.current = true;

    console.log('🎤 Initializing AudioVisualizer...');

    // Initialize microphone and audio analysis
    const initializeMicrophone = async () => {
      try {
        console.log('Requesting microphone access...');
        
        // Request microphone with optimal settings (browsers ignore sampleRate)
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: false,  // Disable AGC to prevent audio distortion
            channelCount: 1          // Mono audio
          } 
        });
        
        console.log('Microphone access granted');
        microphoneRef.current = stream;
        stableMicrophonePermission(true);

        // Create audio context and check sample rate
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        
        // Resume audio context if it's suspended (required on some browsers)
        if (audioContext.state === 'suspended') {
          await audioContext.resume();
        }
        
        audioContextRef.current = audioContext;
        console.log(`🎵 Audio context sample rate: ${audioContext.sampleRate}Hz`);
        
        // Set up audio processing chain with downsampling if needed
        let finalStream = stream;
        
        if (audioContext.sampleRate !== 16000) {
          try {
            // Load the downsampler worklet
            await audioContext.audioWorklet.addModule('/downsampler.js');
            
            // Create audio source and downsampler
            const source = audioContext.createMediaStreamSource(stream);
            const downsamplerNode = new AudioWorkletNode(audioContext, 'downsampler', {
              processorOptions: { targetSampleRate: 16000 }
            });
            
            // Create destination for resampled audio
            const dest = audioContext.createMediaStreamDestination();
            
            // Connect: source -> downsampler -> destination
            source.connect(downsamplerNode);
            downsamplerNode.connect(dest);
            
            finalStream = dest.stream;
            console.log('✅ Audio downsampler initialized: 48kHz → 16kHz');
          } catch (error) {
            console.warn('⚠️ Failed to initialize downsampler, using original stream:', error);
            finalStream = stream;
          }
        }

        // Initialize MediaRecorder only when opus streaming is enabled
        if (USE_MEDIA_RECORDER_STREAMING) {
          let mediaRecorder: MediaRecorder | null = null;
          try {
            let selectedMime = 'audio/ogg;codecs=opus';
            try {
              const isTypeSupported = (MediaRecorder as any)?.isTypeSupported?.bind(MediaRecorder);
              if (isTypeSupported) {
                if (isTypeSupported('audio/ogg;codecs=opus')) {
                  selectedMime = 'audio/ogg;codecs=opus';
                } else if (isTypeSupported('audio/webm;codecs=opus')) {
                  selectedMime = 'audio/webm;codecs=opus';
                }
              }
            } catch {}
            const mediaRecorderOptions: MediaRecorderOptions = {
              mimeType: selectedMime,
              audioBitsPerSecond: 128000,
            };
            mediaRecorder = new MediaRecorder(finalStream, mediaRecorderOptions);
            console.log('✅ MediaRecorder initialized with:', mediaRecorderOptions);
            (window as any).voiceAgentMediaRecorder = mediaRecorder;
            opusStreamingActiveRef.current = true;

            mediaRecorder.ondataavailable = (event) => {
              if (!USE_MEDIA_RECORDER_STREAMING) return;
              if (event.data.size > 0) {
                console.log(`📦 MediaRecorder chunk: ${event.data.size} bytes`);
                const reader = new FileReader();
                reader.onload = () => {
                  const arrayBuffer = reader.result as ArrayBuffer;
                  const base64Audio = btoa(String.fromCharCode.apply(null, Array.from(new Uint8Array(arrayBuffer))));
                  const blobType = (event.data.type || '').toLowerCase();
                  const detectedFormat = blobType.includes('webm') ? 'webm_opus' : 'ogg_opus';
                  const audioMessage = {
                    type: 'audio_chunk',
                    data: base64Audio,
                    is_final: finalizeNextOpusChunkRef.current === true,
                    format: detectedFormat,
                    sample_rate: audioContext.sampleRate !== 16000 ? 16000 : audioContext.sampleRate,
                    channels: 1,
                    chunk_size: event.data.size,
                    timestamp: Date.now(),
                  };
                  finalizeNextOpusChunkRef.current = false;
                  const ws = (window as any).voiceAgentWebSocket;
                  if (ws && ws.readyState === WebSocket.OPEN) {
                    try { ws.send(JSON.stringify(audioMessage)); } catch {}
                  }
                };
                reader.readAsArrayBuffer(event.data);
              }
            };
            mediaRecorder.onstop = () => console.log('📦 MediaRecorder stopped');
            mediaRecorder.start(250);
            console.log('🎙️ MediaRecorder started with 250ms chunks');
          } catch (error) {
            console.warn('⚠️ MediaRecorder init failed:', error);
            opusStreamingActiveRef.current = false;
          }
        }

        // Create analyser using the existing audio context
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.9; // increase smoothing to reduce jitter
        analyser.minDecibels = -90;
        analyser.maxDecibels = -10;
        analyserRef.current = analyser;

        // Create audio processing chain - use finalStream for analysis too
        const source = audioContext.createMediaStreamSource(finalStream);
        
        // 1. High-pass filter to kill HVAC rumble (120Hz cutoff)
        const highPassFilter = audioContext.createBiquadFilter();
        highPassFilter.type = 'highpass';
        highPassFilter.frequency.value = 120; // Hz
        
        // 2. Gain boost (reduced) to keep RMS in a stable range
        const gainNode = audioContext.createGain();
        gainNode.gain.value = 1.5; // reduce gain to avoid over-driving visualizer
        
        // Connect the processing chain: source -> highpass -> gain -> analyser
        source.connect(highPassFilter);
        highPassFilter.connect(gainNode);
        gainNode.connect(analyser);
        
        // Store gain node reference for potential adjustment
        (window as any).audioGainNode = gainNode;
        
        console.log('Audio analysis setup complete');
        console.log(`🎵 Audio context sample rate: ${audioContext.sampleRate}Hz`);
      } catch (error: any) {
        console.error('Failed to access microphone:', error);
        let errorMessage = 'Failed to access microphone';
        
        if (error.name === 'NotAllowedError') {
          errorMessage = 'Microphone access denied by user';
        } else if (error.name === 'NotFoundError') {
          errorMessage = 'No microphone found';
        } else if (error.name === 'NotSupportedError') {
          errorMessage = 'Microphone not supported by browser';
        } else if (error.name === 'OverconstrainedError') {
          errorMessage = 'Microphone constraints could not be satisfied';
        }
        
        console.error('Microphone error details:', errorMessage);
        stableMicrophonePermission(false);
      }
    };

    // Initialize Three.js scene
    const initializeScene = () => {
      console.log('🎨 Initializing Three.js scene...');
      
      // ——————————————————————————————————————————
      // 1. Setup renderer, scene & camera
      // ——————————————————————————————————————————
      const renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      containerRef.current!.appendChild(renderer.domElement);

      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(
        45,
        window.innerWidth / window.innerHeight,
        0.1,
        1000,
      );
      camera.position.set(0, 0, 8);

      // ——————————————————————————————————————————
      // 2. Uniforms & shaders (ported from user demo)
      // ——————————————————————————————————————————
      const uniforms = {
        u_time: { value: 0.0 },
        u_frequency: { value: 0.0 },
        u_red: { value: 1.0 },
        u_green: { value: 1.0 },
        u_blue: { value: 1.0 },
      } as Record<string, THREE.IUniform>;

      const vertexShader = `
        uniform float u_frequency;
        uniform float u_time;

        vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec4 permute(vec4 x) { return mod289(((x*34.0)+10.0)*x); }
        vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
        vec3 fade(vec3 t) { return t*t*t*(t*(t*6.0-15.0)+10.0); }

        float pnoise(vec3 P, vec3 rep) {
          vec3 Pi0 = mod(floor(P), rep);
          vec3 Pi1 = mod(Pi0 + vec3(1.0), rep);
          Pi0 = mod289(Pi0);
          Pi1 = mod289(Pi1);
          vec3 Pf0 = fract(P);
          vec3 Pf1 = Pf0 - vec3(1.0);
          vec4 ix = vec4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
          vec4 iy = vec4(Pi0.yy, Pi1.yy);
          vec4 iz0 = Pi0.zzzz;
          vec4 iz1 = Pi1.zzzz;
          vec4 ixy = permute(permute(ix) + iy);
          vec4 ixy0 = permute(ixy + iz0);
          vec4 ixy1 = permute(ixy + iz1);
          vec4 gx0 = ixy0 * (1.0 / 7.0);
          vec4 gy0 = fract(floor(gx0) * (1.0 / 7.0)) - 0.5;
          gx0 = fract(gx0);
          vec4 gz0 = vec4(0.5) - abs(gx0) - abs(gy0);
          vec4 sz0 = step(gz0, vec4(0.0));
          gx0 -= sz0 * (step(0.0, gx0) - 0.5);
          gy0 -= sz0 * (step(0.0, gy0) - 0.5);
          vec4 gx1 = ixy1 * (1.0 / 7.0);
          vec4 gy1 = fract(floor(gx1) * (1.0 / 7.0)) - 0.5;
          gx1 = fract(gx1);
          vec4 gz1 = vec4(0.5) - abs(gx1) - abs(gy1);
          vec4 sz1 = step(gz1, vec4(0.0));
          gx1 -= sz1 * (step(0.0, gx1) - 0.5);
          gy1 -= sz1 * (step(0.0, gy1) - 0.5);
          vec3 g000 = vec3(gx0.x,gy0.x,gz0.x);
          vec3 g100 = vec3(gx0.y,gy0.y,gz0.y);
          vec3 g010 = vec3(gx0.z,gy0.z,gz0.z);
          vec3 g110 = vec3(gx0.w,gy0.w,gz0.w);
          vec3 g001 = vec3(gx1.x,gy1.x,gz1.x);
          vec3 g101 = vec3(gx1.y,gy1.y,gz1.y);
          vec3 g011 = vec3(gx1.z,gy1.z,gz1.z);
          vec3 g111 = vec3(gx1.w,gy1.w,gz1.w);
          vec4 norm0 = taylorInvSqrt(vec4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
          g000 *= norm0.x;
          g010 *= norm0.y;
          g100 *= norm0.z;
          g110 *= norm0.w;
          vec4 norm1 = taylorInvSqrt(vec4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
          g001 *= norm1.x;
          g011 *= norm1.y;
          g101 *= norm1.z;
          g111 *= norm1.w;
          float n000 = dot(g000, Pf0);
          float n100 = dot(g100, vec3(Pf1.x, Pf0.yz));
          float n010 = dot(g010, vec3(Pf0.x, Pf1.y, Pf0.z));
          float n110 = dot(g110, vec3(Pf1.xy, Pf0.z));
          float n001 = dot(g001, vec3(Pf0.xy, Pf1.z));
          float n101 = dot(g101, vec3(Pf1.x, Pf0.y, Pf1.z));
          float n011 = dot(g011, vec3(Pf0.x, Pf1.yz));
          float n111 = dot(g111, Pf1);
          vec3 fade_xyz = fade(Pf0);
          vec4 n_z = mix(vec4(n000, n100, n010, n110), vec4(n001, n101, n011, n111), fade_xyz.z);
          vec2 n_yz = mix(n_z.xy, n_z.zw, fade_xyz.y);
          float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
          return 2.2 * n_xyz;
        }

        void main() {
          float noise = 3.0 * pnoise(position + u_time, vec3(10.0));
          float displacement = (u_frequency / 30.0) * (noise / 10.0);
          vec3 newPosition = position + normal * displacement;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
        }
      `;

      const fragmentShader = `
        uniform float u_red;
        uniform float u_blue;
        uniform float u_green;

        void main() {
          gl_FragColor = vec4(vec3(u_red, u_green, u_blue), 1.0);
        }
      `;

      // ——————————————————————————————————————————
      // 3. Geometry & mesh
      // ——————————————————————————————————————————
      const geometry = new THREE.IcosahedronGeometry(1.5, 30);
      const material = new THREE.ShaderMaterial({
        wireframe: true,
        uniforms,
        vertexShader,
        fragmentShader,
      });
      const mesh = new THREE.Mesh(geometry, material);
      scene.add(mesh);

      // ——————————————————————————————————————————
      // 4. Real-time audio analysis
      // ——————————————————————————————————————————
      const listener = new THREE.AudioListener();
      camera.add(listener);
      const sound = new THREE.Audio(listener);
      soundRef.current = sound;

      // Simplified VAD with cleaner logic
      let audioRecordingBuffer: Float32Array[] = [];
      let isCurrentlyRecording = false;
      let silenceFrames = 0;
      let speechFrames = 0;
      let lastRecordingTime = 0;
      
      // Simplified thresholds based on RMS energy
      const SPEECH_START_FRAMES = 3;    // ~60ms of speech to start
      const SILENCE_END_FRAMES = 15;    // ~300ms of silence to stop
      const MIN_RECORDING_MS = 500;     // Minimum recording duration
      const DEBOUNCE_MS = 300;          // Prevent rapid on/off
      
      // Energy thresholds (RMS-based, more reliable than frequency analysis)
      const NOISE_FLOOR = 0.01;         // raise noise floor to ignore ambient noise
      const SPEECH_THRESHOLD = 0.05;    // raise threshold to avoid false positives
      const STRONG_SPEECH_THRESHOLD = 0.12; // raise threshold for strong speech
      
      const getAverageFrequency = (): number => {
        if (!analyserRef.current || muted) return 0;
        
        // Get time domain data for RMS calculation (more reliable than frequency analysis)
        const bufferLength = analyserRef.current.fftSize;
        const timeData = new Float32Array(bufferLength);
        analyserRef.current.getFloatTimeDomainData(timeData);
        
        // Calculate RMS (Root Mean Square) energy
        let rms = 0;
        for (let i = 0; i < timeData.length; i++) {
          rms += timeData[i] * timeData[i];
        }
        rms = Math.sqrt(rms / timeData.length);
        
        // Determine voice activity based on RMS energy
        let isVoiceActive = false;
        
        if (rms > STRONG_SPEECH_THRESHOLD) {
          // Strong speech detected
          speechFrames = Math.min(speechFrames + 2, SPEECH_START_FRAMES * 2);
          silenceFrames = 0;
          isVoiceActive = true;
          // toned down logging to avoid noisy console during active speech
        } else if (rms > SPEECH_THRESHOLD) {
          // Moderate speech detected
          speechFrames = Math.min(speechFrames + 1, SPEECH_START_FRAMES * 2);
          silenceFrames = Math.max(silenceFrames - 1, 0);
          isVoiceActive = speechFrames >= SPEECH_START_FRAMES;
          if (isVoiceActive) {
            console.log(`🎤 Speech detected: RMS=${rms.toFixed(3)}`);
          }
        } else {
          // Silence or noise
          speechFrames = Math.max(speechFrames - 1, 0);
          silenceFrames = Math.min(silenceFrames + 1, SILENCE_END_FRAMES);
          isVoiceActive = speechFrames >= SPEECH_START_FRAMES && silenceFrames < SILENCE_END_FRAMES;
        }
        
        // Handle audio recording and streaming (only if not muted)
        if (!muted) {
          handleAudioRecording(isVoiceActive, rms);
        }
        
        // Call voice activity callback (only report activity if not muted)
        stableVoiceActivity(!muted && isVoiceActive);
        
        // Return scaled value for visualization (convert RMS to 0-255 range)
        return Math.min(rms * 1000, 255); // reduce scaling to soften motion
      };
      
      // Audio recording and streaming logic
      const handleAudioRecording = (isVoiceActive: boolean, rmsLevel: number) => {
        // Only skip PCM pipeline if Opus streaming is active
        if (USE_MEDIA_RECORDER_STREAMING && opusStreamingActiveRef.current) return;
        if (!audioContextRef.current || !microphoneRef.current || muted) return;
        
        const now = Date.now();
        
        // Start recording when voice is detected (with debouncing)
        if (isVoiceActive && !isCurrentlyRecording && (now - lastRecordingTime) > DEBOUNCE_MS) {
          // Additional validation: ensure RMS level is significantly above noise floor
          if (rmsLevel > NOISE_FLOOR * 2) {
            console.log('🎙️ Starting voice recording...', { rmsLevel, noiseFloor: NOISE_FLOOR });
            isCurrentlyRecording = true;
            audioRecordingBuffer = [];
            speechFrames = SPEECH_START_FRAMES; // Reset counters
            silenceFrames = 0;
            lastRecordingTime = now;
          }
        }
        
        // Continue recording while voice is active or during short silence
        if (isCurrentlyRecording) {
          // Capture current audio frame
          captureAudioFrame();
          
          // Stop recording after sufficient silence (with debouncing)
          if (!isVoiceActive && silenceFrames >= SILENCE_END_FRAMES && (now - lastRecordingTime) > MIN_RECORDING_MS) {
            console.log('🎙️ Stopping voice recording...', { silenceFrames, rmsLevel });
            isCurrentlyRecording = false;
            
            // Finalize Opus or PCM path
            const mr: MediaRecorder | undefined = (window as any).voiceAgentMediaRecorder;
            if (USE_MEDIA_RECORDER_STREAMING && opusStreamingActiveRef.current && mr && mr.state === 'recording') {
              finalizeNextOpusChunkRef.current = true;
              try { mr.requestData(); } catch {}
            } else {
              // PCM fallback final send
              if (audioRecordingBuffer.length > 0) {
                sendAudioToBackend(true); // is_final = true
              }
            }
            
            // Reset state
            audioRecordingBuffer = [];
            speechFrames = 0;
            silenceFrames = 0;
            lastRecordingTime = now;
          }
        }
      };
      
      // Capture current audio frame for streaming
      const captureAudioFrame = () => {
        // Only skip PCM pipeline if Opus streaming is active
        if (USE_MEDIA_RECORDER_STREAMING && opusStreamingActiveRef.current) return;
        if (!analyserRef.current || muted) return;
        
        // Get time domain data (actual audio samples)
        const bufferLength = analyserRef.current.fftSize;
        const dataArray = new Float32Array(bufferLength);
        analyserRef.current.getFloatTimeDomainData(dataArray);
        
        // Resample from browser sample rate to 16kHz if needed
        const browserSampleRate = audioContextRef.current?.sampleRate || 44100;
        const targetSampleRate = 16000;
        
        let resampledData: Float32Array;
        if (browserSampleRate !== targetSampleRate) {
          // Simple downsampling by taking every nth sample
          const ratio = browserSampleRate / targetSampleRate;
          const resampledLength = Math.floor(dataArray.length / ratio);
          resampledData = new Float32Array(resampledLength);
          
          for (let i = 0; i < resampledLength; i++) {
            const sourceIndex = Math.floor(i * ratio);
            resampledData[i] = dataArray[sourceIndex];
          }
        } else {
          resampledData = dataArray;
        }
        
        // Store the resampled float data directly (simpler and cleaner)
        audioRecordingBuffer.push(resampledData);
        
        // Ultra-low latency: Send chunks every ~40ms for faster response
        if ((window as any).voiceAgentReady && audioRecordingBuffer.length >= 2) { // ~40ms (2 * ~20ms)
          sendAudioToBackend(false); // is_final = false
        }
      };
      
      // Send audio data to backend via WebSocket
      const sendAudioToBackend = (isFinal: boolean) => {
        // Only skip PCM pipeline if Opus streaming is active
        if (USE_MEDIA_RECORDER_STREAMING && opusStreamingActiveRef.current) return;
        if (audioRecordingBuffer.length === 0) return;
        
        try {
          if (!(window as any).voiceAgentReady) {
            // Hold until SettingsApplied
            return;
          }
          // Concatenate all buffered audio frames
          const totalLength = audioRecordingBuffer.reduce((sum, arr) => sum + arr.length, 0);
          const combinedAudio = new Float32Array(totalLength);
          let offset = 0;
          
          for (const frame of audioRecordingBuffer) {
            combinedAudio.set(frame, offset);
            offset += frame.length;
          }
          
          // Convert Float32 to Int16 PCM for backend processing
          const pcmData = new Int16Array(combinedAudio.length);
          for (let i = 0; i < combinedAudio.length; i++) {
            // Convert from [-1, 1] to [-32768, 32767] with proper clamping
            const sample = Math.max(-1, Math.min(1, combinedAudio[i]));
            pcmData[i] = Math.round(sample * 32767);
          }
          
          // Send raw binary PCM directly for lowest latency and zero base64 overhead
          const ws = (window as any).voiceAgentWebSocket;
          if (ws && ws.readyState === WebSocket.OPEN) {
            try {
              console.log(`🎵 Sending PCM chunk: ${pcmData.byteLength} bytes, final: ${isFinal}`);
              ws.send(pcmData.buffer);
            } catch (error) {
              console.error('❌ Failed to send audio message:', error);
            }
          } else {
            console.warn('⚠️ WebSocket not connected, discarding audio chunk', {
              wsExists: !!ws,
              readyState: ws?.readyState,
              expectedState: WebSocket.OPEN
            });
          }
          
          // Clear the buffer for non-final chunks (keep minimal overlap)
          if (!isFinal) {
            audioRecordingBuffer = audioRecordingBuffer.slice(-2); // Keep ~40ms overlap for continuity
          }
          
        } catch (error) {
          console.error('❌ Failed to send audio to backend:', error);
        }
      };

      // ——————————————————————————————————————————
      // 5. Animation loop & resize
      // ——————————————————————————————————————————
      const clock = new THREE.Clock();

      const animate = () => {
        uniforms.u_time.value = clock.getElapsedTime();
        uniforms.u_frequency.value = getAverageFrequency();

        // Color changes based on voice activity
        const frequency = uniforms.u_frequency.value;
        
        if (frequency > 60) { // Use a simple threshold for visualization
          // Active voice - colorful
          uniforms.u_red.value = 0.5 + Math.sin(clock.getElapsedTime() * 2) * 0.5;
          uniforms.u_green.value = 0.5 + Math.sin(clock.getElapsedTime() * 2 + Math.PI / 3) * 0.5;
          uniforms.u_blue.value = 0.5 + Math.sin(clock.getElapsedTime() * 2 + 2 * Math.PI / 3) * 0.5;
        } else {
          // No voice - subtle blue/purple
          uniforms.u_red.value = 0.3;
          uniforms.u_green.value = 0.2;
          uniforms.u_blue.value = 0.8;
        }

        // optional slow rotation for subtle dynamism
        mesh.rotation.y += 0.002;

        renderer.render(scene, camera);
        requestIdRef.current = requestAnimationFrame(animate);
      };
      animate();

      const handleResize = () => {
        const { innerWidth, innerHeight } = window;
        camera.aspect = innerWidth / innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(innerWidth, innerHeight);
      };
      window.addEventListener('resize', handleResize);

      // Return cleanup function
      return () => {
        console.log('🧹 Cleaning up Three.js scene...');
        if (requestIdRef.current) cancelAnimationFrame(requestIdRef.current);
        window.removeEventListener('resize', handleResize);
        geometry.dispose();
        material.dispose();
        renderer.dispose();
        if (containerRef.current && renderer.domElement.parentNode === containerRef.current) {
          containerRef.current.removeChild(renderer.domElement);
        }
      };
    };

    // Initialize both microphone and scene
    const init = async () => {
      await initializeMicrophone();
      cleanupRef.current = initializeScene();
    };

    init();

    // Cleanup on unmount
    return () => {
      console.log('🧹 Cleaning up AudioVisualizer...');
      initializedRef.current = false;
      
      // Stop microphone
      if (microphoneRef.current) {
        microphoneRef.current.getTracks().forEach(track => track.stop());
      }
      
      // Close audio context
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
      
      // Cleanup Three.js scene
      if (cleanupRef.current) {
        cleanupRef.current();
      }
    };
  }, []); // Empty dependency array - only run once on mount

  // Handle muted prop changes separately
  useEffect(() => {
    if (!soundRef.current) return;
    soundRef.current.setVolume(muted ? 0 : 1);
  }, [muted]);

  return <div ref={containerRef} className="absolute inset-0" />;
};

export default AudioVisualizer; 