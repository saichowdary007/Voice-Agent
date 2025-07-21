// AudioWorklet processor for downsampling audio from 48kHz to 16kHz
class DownsamplerProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    
    const { targetSampleRate = 16000 } = options.processorOptions || {};
    this.targetSampleRate = targetSampleRate;
    this.inputSampleRate = sampleRate; // AudioContext sample rate (usually 48kHz)
    
    // Calculate decimation ratio
    this.decimationRatio = this.inputSampleRate / this.targetSampleRate;
    
    // Simple low-pass filter coefficients for anti-aliasing
    // Butterworth 4th order at Nyquist/2 (4kHz for 16kHz target)
    this.filterBuffer = new Float32Array(8);
    this.filterIndex = 0;
    
    // Output buffer for accumulating downsampled data
    this.outputBuffer = [];
    this.sampleCounter = 0;
    
    console.log(`🎵 Downsampler initialized: ${this.inputSampleRate}Hz → ${this.targetSampleRate}Hz (ratio: ${this.decimationRatio})`);
  }
  
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    
    if (input.length === 0) return true;
    
    const inputChannel = input[0]; // Mono input
    const outputChannel = output[0];
    
    // Process each input sample
    for (let i = 0; i < inputChannel.length; i++) {
      // Simple anti-aliasing filter (moving average)
      this.filterBuffer[this.filterIndex] = inputChannel[i];
      this.filterIndex = (this.filterIndex + 1) % this.filterBuffer.length;
      
      const filtered = this.filterBuffer.reduce((sum, val) => sum + val, 0) / this.filterBuffer.length;
      
      // Decimation - only keep every Nth sample
      if (this.sampleCounter % Math.round(this.decimationRatio) === 0) {
        this.outputBuffer.push(filtered);
        
        // Send batch when we have enough samples (every ~10ms worth)
        if (this.outputBuffer.length >= Math.floor(this.targetSampleRate * 0.01)) {
          this.port.postMessage({
            type: 'audio',
            samples: new Float32Array(this.outputBuffer),
            sampleRate: this.targetSampleRate
          });
          this.outputBuffer = [];
        }
      }
      
      this.sampleCounter++;
    }
    
    // Pass through original audio for monitoring
    if (outputChannel) {
      outputChannel.set(inputChannel);
    }
    
    return true;
  }
}

registerProcessor('downsampler', DownsamplerProcessor);