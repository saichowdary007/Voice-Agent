import { AudioConfig, AudioConstraints, MediaDeviceInfo } from './types';

/**
 * Audio utility class for handling microphone input and audio processing
 */
export class AudioManager {
  private mediaStream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private mediaRecorder: MediaRecorder | null = null;
  private processor: ScriptProcessorNode | null = null;
  private isRecording = false;
  private onAudioData: ((data: ArrayBuffer) => void) | null = null;
  private onAudioLevel: ((level: number) => void) | null = null;
  private config: AudioConfig;

  constructor(config: AudioConfig) {
    this.config = config;
  }

  /**
   * Initialize audio system and request microphone permissions
   */
  async initialize(): Promise<void> {
    try {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      console.error('Failed to initialize audio:', error);
      throw error;
    }
  }

  /**
   * Start recording audio and processing
   */
  async startRecording(
    onAudioData: (data: ArrayBuffer) => void,
    onAudioLevel?: (level: number) => void
  ): Promise<void> {
    if (!this.mediaStream || !this.audioContext) {
      throw new Error('Audio not initialized');
    }

    if (this.isRecording) {
      return;
    }

    try {
      this.onAudioData = onAudioData;
      this.onAudioLevel = onAudioLevel || null;

      // Create MediaRecorder for Opus encoding
      const options: MediaRecorderOptions = {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 32000, // 32kbps for voice
      };

      this.mediaRecorder = new MediaRecorder(this.mediaStream, options);

      // Handle data available
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && this.onAudioData) {
          event.data.arrayBuffer().then((buffer) => {
            this.onAudioData!(buffer);
          });
        }
      };

      // Start recording with specified time slice
      this.mediaRecorder.start(this.config.frameDuration);

      // Setup audio level monitoring
      if (onAudioLevel) {
        this.setupAudioLevelMonitoring();
      }

      this.isRecording = true;
      console.log('Started recording audio');
    } catch (error) {
      console.error('Failed to start recording:', error);
      throw new Error(`Recording failed: ${error}`);
    }
  }

  /**
   * Stop recording audio
   */
  stopRecording(): void {
    if (!this.isRecording) {
      return;
    }

    try {
      if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
        this.mediaRecorder.stop();
      }

      if (this.processor) {
        this.processor.disconnect();
        this.processor = null;
      }

      this.isRecording = false;
      this.onAudioData = null;
      this.onAudioLevel = null;

      console.log('Stopped recording audio');
    } catch (error) {
      console.error('Error stopping recording:', error);
    }
  }

  /**
   * Setup audio level monitoring for visual feedback
   */
  private setupAudioLevelMonitoring(): void {
    if (!this.audioContext || !this.mediaStream) {
      return;
    }

    try {
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      const analyser = this.audioContext.createAnalyser();

      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.8;

      source.connect(analyser);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      const updateLevel = () => {
        if (!this.isRecording) {
          return;
        }

        analyser.getByteFrequencyData(dataArray);

        // Calculate RMS level
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          sum += dataArray[i] * dataArray[i];
        }
        const rms = Math.sqrt(sum / dataArray.length);
        const level = rms / 255; // Normalize to 0-1

        if (this.onAudioLevel) {
          this.onAudioLevel(level);
        }

        requestAnimationFrame(updateLevel);
      };

      updateLevel();
    } catch (error) {
      console.error('Failed to setup audio level monitoring:', error);
    }
  }

  /**
   * Get available audio input devices
   */
  async getAudioDevices(): Promise<MediaDeviceInfo[]> {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices
        .filter(device => device.kind === 'audioinput')
        .map(device => ({
          deviceId: device.deviceId,
          label: device.label || `Microphone ${device.deviceId.slice(0, 8)}`,
          kind: device.kind as 'audioinput',
        }));
    } catch (error) {
      console.error('Failed to get audio devices:', error);
      return [];
    }
  }

  /**
   * Switch to a different audio input device
   */
  async switchAudioDevice(deviceId: string): Promise<void> {
    try {
      // Stop current recording
      const wasRecording = this.isRecording;
      if (wasRecording) {
        this.stopRecording();
      }

      // Release current stream
      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach(track => track.stop());
      }

      // Get new stream with specified device
      const constraints: MediaStreamConstraints = {
        audio: {
          deviceId: { exact: deviceId },
          sampleRate: this.config.sampleRate,
          channelCount: this.config.channelCount,
          echoCancellation: this.config.enableEchoCancellation,
          noiseSuppression: this.config.enableNoiseSuppression,
          autoGainControl: this.config.enableAutoGainControl,
        },
        video: false,
      };

      this.mediaStream = await navigator.mediaDevices.getUserMedia(constraints);

      // Restart recording if it was active
      if (wasRecording && this.onAudioData) {
        await this.startRecording(this.onAudioData, this.onAudioLevel || undefined);
      }

      console.log(`Switched to audio device: ${deviceId}`);
    } catch (error) {
      console.error('Failed to switch audio device:', error);
      throw new Error(`Device switch failed: ${error}`);
    }
  }

  /**
   * Check if audio recording is supported
   */
  static isRecordingSupported(): boolean {
    return !!(
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === 'function' &&
      window.MediaRecorder &&
      MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    );
  }

  /**
   * Get recording state
   */
  getRecordingState(): boolean {
    return this.isRecording;
  }

  /**
   * Clean up resources
   */
  cleanup(): void {
    this.stopRecording();

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }

    console.log('Audio manager cleaned up');
  }

  getAudioContext(): AudioContext | null {
    return this.audioContext;
  }

  getMediaStream(): MediaStream | null {
    return this.mediaStream;
  }
}

/**
 * Audio playback manager for TTS audio
 */
export class AudioPlayer {
  private audioContext: AudioContext | null = null;
  private currentSource: AudioBufferSourceNode | null = null;
  private isPlaying = false;
  private onPlaybackEnd: (() => void) | null = null;

  constructor(audioContext?: AudioContext) {
    // Initialize AudioContext if not provided
    if (audioContext) {
      this.audioContext = audioContext;
    } else {
      this.initializeAudioContext();
    }
  }

  /**
   * Initialize audio context
   */
  private initializeAudioContext(): void {
    try {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioContextClass) {
        this.audioContext = new AudioContextClass();
        console.log('AudioPlayer: AudioContext initialized');
      } else {
        throw new Error('AudioContext not supported in this browser');
      }
    } catch (error) {
      console.error('Failed to initialize AudioContext:', error);
    }
  }

  /**
   * Ensure audio context is ready
   */
  private async ensureAudioContextReady(): Promise<void> {
    if (!this.audioContext) {
      this.initializeAudioContext();
    }

    if (this.audioContext && this.audioContext.state === 'suspended') {
      try {
        await this.audioContext.resume();
        console.log('AudioPlayer: AudioContext resumed');
      } catch (error) {
        console.error('Failed to resume AudioContext:', error);
        throw error;
      }
    }
  }

  /**
   * Play audio buffer (handles MP3 and other formats)
   */
  async playAudio(audioData: ArrayBuffer, onEnd?: () => void): Promise<void> {
    try {
      // Ensure audio context is ready
      await this.ensureAudioContextReady();

      if (!this.audioContext) {
        throw new Error('Audio context not available');
      }

      // Stop current playback
      this.stopAudio();

      this.onPlaybackEnd = onEnd || null;

      console.log('AudioPlayer: Decoding audio data...', audioData.byteLength, 'bytes');

      // Decode audio data (supports MP3, WAV, etc.)
      const audioBuffer = await this.audioContext.decodeAudioData(audioData.slice(0));

      console.log('AudioPlayer: Audio decoded successfully', 
        `${audioBuffer.duration.toFixed(2)}s`, 
        `${audioBuffer.sampleRate}Hz`,
        `${audioBuffer.numberOfChannels} channels`
      );

      // Create source
      this.currentSource = this.audioContext.createBufferSource();
      this.currentSource.buffer = audioBuffer;
      this.currentSource.connect(this.audioContext.destination);

      // Handle end of playback
      this.currentSource.onended = () => {
        console.log('AudioPlayer: Playback ended');
        this.isPlaying = false;
        this.currentSource = null;
        if (this.onPlaybackEnd) {
          this.onPlaybackEnd();
        }
      };

      // Start playback
      this.currentSource.start();
      this.isPlaying = true;

      console.log('AudioPlayer: Started audio playback');
    } catch (error) {
      console.error('AudioPlayer: Failed to play audio:', error);
      this.isPlaying = false;
      this.currentSource = null;
      
      // Call onEnd callback even on error
      if (this.onPlaybackEnd) {
        this.onPlaybackEnd();
      }
      
      throw new Error(`Audio playback failed: ${error}`);
    }
  }

  /**
   * Stop current audio playback
   */
  stopAudio(): void {
    if (this.currentSource && this.isPlaying) {
      try {
        this.currentSource.stop();
        this.currentSource = null;
        this.isPlaying = false;
        console.log('AudioPlayer: Stopped audio playback');
      } catch (error) {
        console.error('AudioPlayer: Error stopping audio:', error);
      }
    }
  }

  /**
   * Get playback state
   */
  getPlaybackState(): boolean {
    return this.isPlaying;
  }

  /**
   * Get audio context
   */
  getAudioContext(): AudioContext | null {
    return this.audioContext;
  }

  /**
   * Clean up resources
   */
  cleanup(): void {
    this.stopAudio();

    // Note: Don't close AudioContext as it might be shared
    // Only close if we created it ourselves
    console.log('AudioPlayer: Cleaned up');
  }
}

/**
 * Utility functions for audio processing
 */
export class AudioUtils {
  /**
   * Convert Float32Array to Int16Array for processing
   */
  static float32ToInt16(float32Array: Float32Array): Int16Array {
    const int16Array = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const sample = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
    }
    return int16Array;
  }

  /**
   * Calculate RMS level of audio data
   */
  static calculateRMS(audioData: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < audioData.length; i++) {
      sum += audioData[i] * audioData[i];
    }
    return Math.sqrt(sum / audioData.length);
  }

  /**
   * Apply simple noise gate to audio data
   */
  static applyNoiseGate(audioData: Float32Array, threshold: number = 0.01): Float32Array {
    const output = new Float32Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
      const level = Math.abs(audioData[i]);
      output[i] = level > threshold ? audioData[i] : 0;
    }
    return output;
  }

  /**
   * Check if browser supports required audio features
   */
  static checkAudioSupport(): {
    getUserMedia: boolean;
    mediaRecorder: boolean;
    opus: boolean;
    audioContext: boolean;
  } {
    return {
      getUserMedia: !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia),
      mediaRecorder: !!window.MediaRecorder,
      opus: !!(window.MediaRecorder && MediaRecorder.isTypeSupported('audio/webm;codecs=opus')),
      audioContext: !!(window.AudioContext || (window as any).webkitAudioContext),
    };
  }
} 