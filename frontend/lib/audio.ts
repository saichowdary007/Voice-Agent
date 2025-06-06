// Audio Player for TTS audio playback
export class AudioPlayer {
  private audioContext: AudioContext | null = null;
  private gainNode: GainNode | null = null;
  private isPlaying: boolean = false;
  private currentSource: AudioBufferSourceNode | null = null;

  constructor() {
    this.initializeAudioContext();
  }

  private async initializeAudioContext() {
    try {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      this.gainNode = this.audioContext.createGain();
      this.gainNode.connect(this.audioContext.destination);
    } catch (error) {
      console.error('Failed to initialize audio context:', error);
    }
  }

  async playAudioData(audioData: ArrayBuffer): Promise<void> {
    if (!this.audioContext || !this.gainNode) {
      console.error('Audio context not initialized');
      return;
    }

    try {
      // Resume audio context if suspended
      if (this.audioContext.state === 'suspended') {
        await this.audioContext.resume();
      }

      // Stop any currently playing audio
      this.stop();

      // Check if audio data is valid
      if (!audioData || audioData.byteLength === 0) {
        console.warn('Empty audio data received, skipping playback');
        return;
      }

      // Decode audio data
      let audioBuffer;
      try {
        audioBuffer = await this.audioContext.decodeAudioData(audioData);
      } catch (decodeError) {
        console.error('Failed to decode audio data:', decodeError);
        return;
      }
      
      // Create source node
      this.currentSource = this.audioContext.createBufferSource();
      this.currentSource.buffer = audioBuffer;
      this.currentSource.connect(this.gainNode);

      // Set up event handlers
      this.currentSource.onended = () => {
        this.isPlaying = false;
        this.currentSource = null;
      };

      // Start playback
      this.currentSource.start(0);
      this.isPlaying = true;

    } catch (error) {
      const errorMessage = error && (error as any).message ? (error as any).message : 'Unknown error';
      console.error('Failed to play audio:', errorMessage);
      this.isPlaying = false;
    }
  }

  stop(): void {
    if (this.currentSource) {
      try {
        this.currentSource.stop();
      } catch (error) {
        // Ignore errors from stopping already stopped sources
      }
      this.currentSource = null;
    }
    this.isPlaying = false;
  }

  setVolume(volume: number): void {
    if (this.gainNode) {
      this.gainNode.gain.value = Math.max(0, Math.min(1, volume));
    }
  }

  getIsPlaying(): boolean {
    return this.isPlaying;
  }

  async resume(): Promise<void> {
    if (this.audioContext && this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
  }

  dispose(): void {
    this.stop();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    this.gainNode = null;
  }
} 