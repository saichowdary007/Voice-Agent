/**
 * Audio processing and playback utilities
 */

export class AudioPlayer {
  private audioContext: AudioContext | null = null;
  private queue: AudioBuffer[] = [];
  private isPlaying: boolean = false;
  private gainNode: GainNode | null = null;
  private currentSource: AudioBufferSourceNode | null = null;

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
   * Play a raw ArrayBuffer of audio data
   */
  async playAudioData(audioData: ArrayBuffer): Promise<void> {
    if (!this.audioContext) {
      console.error('AudioContext not initialized');
      return;
    }

    try {
      // Resume context if it's suspended (autoplay policy)
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      // Decode the audio data
      const audioBuffer = await this.audioContext.decodeAudioData(audioData);
      
      // Add to queue and start playing if not already
      this.queue.push(audioBuffer);
      
      if (!this.isPlaying) {
        this.playNextInQueue();
      }
    } catch (error) {
      console.error('Error playing audio data:', error);
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
      
      // Start playback
      this.currentSource.start(0);
    } catch (error) {
      console.error('Error in playNextInQueue:', error);
      this.isPlaying = false;
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