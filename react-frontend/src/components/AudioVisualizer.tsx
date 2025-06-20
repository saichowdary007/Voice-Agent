import React, { useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';

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
    initializedRef.current = true;

    console.log('🎤 Initializing AudioVisualizer...');

    // Initialize microphone and audio analysis
    const initializeMicrophone = async () => {
      try {
        console.log('Requesting microphone access...');
        
        // Request microphone permission
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            sampleRate: 44100
          } 
        });
        
        console.log('Microphone access granted');
        microphoneRef.current = stream;
        stableMicrophonePermission(true);

        // Create audio context and analyser
        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        
        // Resume audio context if it's suspended (required on some browsers)
        if (audioContext.state === 'suspended') {
          await audioContext.resume();
        }
        
        audioContextRef.current = audioContext;
        
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        analyser.smoothingTimeConstant = 0.8;
        analyser.minDecibels = -90;
        analyser.maxDecibels = -10;
        analyserRef.current = analyser;

        // Connect microphone to analyser
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        
        console.log('Audio analysis setup complete');
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

      // Real audio analysis function
      const getAverageFrequency = (): number => {
        if (!analyserRef.current || muted) return 0;
        
        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyserRef.current.getByteFrequencyData(dataArray);
        
        // Calculate average frequency for lower frequencies (more sensitive to voice)
        const voiceRange = Math.floor(bufferLength * 0.3); // Focus on lower 30% of frequency range
        const sum = dataArray.slice(0, voiceRange).reduce((acc, value) => acc + value, 0);
        const average = sum / voiceRange;
        
        // Voice activity detection with more sensitive threshold
        const voiceThreshold = 15; // Lowered threshold for better sensitivity
        const isVoiceActive = average > voiceThreshold;
        
        // Add some logging for debugging (reduced frequency)
        if (isVoiceActive && Math.random() < 0.01) { // Log very occasionally when voice is detected
          console.log('🎙️ Voice detected - average frequency:', average);
        }
        
        // Call voice activity callback
        stableVoiceActivity(isVoiceActive);
        
        return average;
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
        if (frequency > 15) { // Match the voice threshold
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