/**
 * Audio processing and playback utilities
 */

export class AudioPlayer {
  private audioContext: AudioContext | null = null;
  private queue: AudioBuffer[] = [];
  private isPlaying: boolean = false;
  private gainNode: GainNode | null = null;
  private currentSource: AudioBufferSourceNode | null = null;
  private lastError: Error | null = null;
  private consecutiveErrors: number = 0;
  private maxConsecutiveErrors: number = 5;

  constructor() {
    try {
      // Create audio context with fallbacks for different browsers
      const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
      this.audioContext = new AudioContext();
      
      // Create gain node for volume control
      this.gainNode = this.audioContext.createGain();
      this.gainNode.gain.value = 1.0; // Default volume
      this.gainNode.connect(this.audioContext.destination);
      
      console.log('AudioPlayer initialized with sample rate:', this.audioContext.sampleRate);
    } catch (error) {
      console.error('Failed to initialize AudioPlayer:', error);
      this.lastError = error instanceof Error ? error : new Error(String(error));
    }
  }

  /**
   * Set the output volume (0.0 to 1.0)
   */
  setVolume(volume: number): void {
    if (this.gainNode) {
      // Clamp volume between 0 and 1
      const safeVolume = Math.max(0, Math.min(1, volume));
      this.gainNode.gain.value = safeVolume;
    }
  }

  /**
   * Get player status
   */
  getStatus(): { isPlaying: boolean, queueLength: number, hasError: boolean, lastError?: string } {
    return {
      isPlaying: this.isPlaying,
      queueLength: this.queue.length,
      hasError: this.lastError !== null,
      lastError: this.lastError?.message
    };
  }

  /**
   * Play a raw ArrayBuffer of audio data
   */
  async playAudioData(audioData: ArrayBuffer): Promise<boolean> {
    if (!this.audioContext || !this.gainNode) {
      console.error('AudioContext or GainNode not initialized');
      return false;
    }

    try {
      // Resume context if it's suspended (autoplay policy)
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      // Safety check for invalid data
      if (!audioData || audioData.byteLength === 0) {
        console.warn('Empty audio data received, skipping playback');
        return false;
      }

      // Decode the audio data
      const audioBuffer = await this.audioContext.decodeAudioData(audioData);
      
      // Validate buffer has actual data
      if (audioBuffer.length === 0 || audioBuffer.duration === 0) {
        console.warn('Decoded audio buffer is empty, skipping playback');
        return false;
      }
      
      // Add to queue and start playing if not already
      this.queue.push(audioBuffer);
      
      if (!this.isPlaying) {
        this.playNextInQueue();
      }
      
      // Reset consecutive errors on success
      this.consecutiveErrors = 0;
      this.lastError = null;
      return true;
    } catch (error) {
      this.consecutiveErrors++;
      this.lastError = error instanceof Error ? error : new Error(String(error));
      console.error(`Error playing audio data (attempt ${this.consecutiveErrors}/${this.maxConsecutiveErrors}):`, error);
      
      // If too many consecutive errors, try to recover the audio context
      if (this.consecutiveErrors >= this.maxConsecutiveErrors) {
        console.warn('Too many consecutive playback errors, attempting to recover audio context');
        this._attemptContextRecovery();
      }
      
      return false;
    }
  }

  /**
   * Play the next item in the queue
   */
  private playNextInQueue(): void {
    if (!this.audioContext || !this.gainNode || this.queue.length === 0) {
      this.isPlaying = false;
      return;
    }

    try {
      const audioBuffer = this.queue.shift();
      if (!audioBuffer) return;

      this.isPlaying = true;
      
      // Create a new source for this buffer
      this.currentSource = this.audioContext.createBufferSource();
      this.currentSource.buffer = audioBuffer;
      this.currentSource.connect(this.gainNode);
      
      // When playback ends, play the next item or set isPlaying to false
      this.currentSource.onended = () => {
        // Clean up the current source
        if (this.currentSource) {
          this.currentSource.disconnect();
          this.currentSource = null;
        }
        
        if (this.queue.length > 0) {
          // Play the next item
          this.playNextInQueue();
        } else {
          // No more items to play
          this.isPlaying = false;
        }
      };
      
      // Add error handler for the source
      this.currentSource.onerror = (err) => {
        console.error('AudioBufferSourceNode error:', err);
        // Cleanup and move to next item
        if (this.currentSource) {
          this.currentSource.disconnect();
          this.currentSource = null;
        }
        this.isPlaying = false;
        
        // Try to continue with next item if available
        if (this.queue.length > 0) {
          setTimeout(() => this.playNextInQueue(), 100);
        }
      };
      
      // Start playback
      this.currentSource.start(0);
    } catch (error) {
      console.error('Error in playNextInQueue:', error);
      this.isPlaying = false;
      this.lastError = error instanceof Error ? error : new Error(String(error));
      
      // Try to continue with next item if available
      if (this.queue.length > 0) {
        setTimeout(() => this.playNextInQueue(), 500);
      }
    }
  }

  /**
   * Attempt to recover the audio context after errors
   */
  private async _attemptContextRecovery(): Promise<void> {
    try {
      // Cleanup existing resources first
      this.stop();
      
      if (this.gainNode) {
        this.gainNode.disconnect();
        this.gainNode = null;
      }
      
      if (this.audioContext) {
        await this.audioContext.close().catch(e => console.warn('Error closing audio context:', e));
        this.audioContext = null;
      }
      
      // Recreate audio context
      const AudioContext = window.AudioContext || (window as any).webkitAudioContext;
      this.audioContext = new AudioContext();
      
      // Recreate gain node
      this.gainNode = this.audioContext.createGain();
      this.gainNode.gain.value = 1.0;
      this.gainNode.connect(this.audioContext.destination);
      
      // Reset error counter
      this.consecutiveErrors = 0;
      console.log('AudioPlayer context recovered successfully');
    } catch (error) {
      console.error('Failed to recover audio context:', error);
    }
  }

  /**
   * Stop all audio playback and clear the queue
   */
  stop(): void {
    if (this.currentSource) {
      try {
        this.currentSource.stop();
        this.currentSource.disconnect();
      } catch (e) {
        // Ignore errors from stopping already stopped sources
      }
      this.currentSource = null;
    }
    
    this.queue = [];
    this.isPlaying = false;
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.stop();
    
    if (this.gainNode) {
      this.gainNode.disconnect();
      this.gainNode = null;
    }
    
    if (this.audioContext) {
      this.audioContext.close().catch(e => console.error('Error closing AudioContext:', e));
      this.audioContext = null;
    }
  }
} 