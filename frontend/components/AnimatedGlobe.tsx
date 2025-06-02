'use client';

import React, { useEffect, useRef } from 'react';

interface AnimatedGlobeProps {
  isConnected?: boolean;
  isSpeaking?: boolean;
  isListening?: boolean;
  audioLevel?: number;
  isLoading?: boolean;
  className?: string;
}

export const AnimatedGlobe: React.FC<AnimatedGlobeProps> = ({
  isConnected = false,
  isSpeaking = false,
  isListening = false,
  audioLevel = 0,
  isLoading = false,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const timeRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const size = 200;
    canvas.width = size;
    canvas.height = size;

    const centerX = size / 2;
    const centerY = size / 2;
    const baseRadius = size * 0.3;

    const animate = () => {
      timeRef.current += 0.02;
      
      // Clear canvas
      ctx.clearRect(0, 0, size, size);

      // Calculate dynamic radius based on audio level and states
      let radiusMultiplier = 1;
      if (isListening) {
        radiusMultiplier += audioLevel * 0.3;
      }
      if (isSpeaking) {
        radiusMultiplier += Math.sin(timeRef.current * 8) * 0.1;
      }

      const currentRadius = baseRadius * radiusMultiplier;

      // Outer glow effect
      const glowGradient = ctx.createRadialGradient(
        centerX, centerY, 0,
        centerX, centerY, currentRadius * 1.5
      );
      
      if (isLoading) {
        glowGradient.addColorStop(0, 'rgba(147, 197, 253, 0.8)'); // blue
        glowGradient.addColorStop(1, 'rgba(147, 197, 253, 0)');
      } else if (isSpeaking) {
        glowGradient.addColorStop(0, 'rgba(168, 85, 247, 0.8)'); // purple
        glowGradient.addColorStop(1, 'rgba(168, 85, 247, 0)');
      } else if (isListening) {
        glowGradient.addColorStop(0, 'rgba(34, 197, 94, 0.8)'); // green
        glowGradient.addColorStop(1, 'rgba(34, 197, 94, 0)');
      } else if (isConnected) {
        glowGradient.addColorStop(0, 'rgba(59, 130, 246, 0.6)'); // blue
        glowGradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
      } else {
        glowGradient.addColorStop(0, 'rgba(156, 163, 175, 0.4)'); // gray
        glowGradient.addColorStop(1, 'rgba(156, 163, 175, 0)');
      }

      ctx.fillStyle = glowGradient;
      ctx.fillRect(0, 0, size, size);

      // Main sphere
      const sphereGradient = ctx.createRadialGradient(
        centerX - currentRadius * 0.3, centerY - currentRadius * 0.3, 0,
        centerX, centerY, currentRadius
      );

      if (isLoading) {
        sphereGradient.addColorStop(0, 'rgba(191, 219, 254, 0.9)');
        sphereGradient.addColorStop(0.7, 'rgba(59, 130, 246, 0.7)');
        sphereGradient.addColorStop(1, 'rgba(29, 78, 216, 0.5)');
      } else if (isSpeaking) {
        sphereGradient.addColorStop(0, 'rgba(221, 214, 254, 0.9)');
        sphereGradient.addColorStop(0.7, 'rgba(168, 85, 247, 0.7)');
        sphereGradient.addColorStop(1, 'rgba(107, 33, 168, 0.5)');
      } else if (isListening) {
        sphereGradient.addColorStop(0, 'rgba(187, 247, 208, 0.9)');
        sphereGradient.addColorStop(0.7, 'rgba(34, 197, 94, 0.7)');
        sphereGradient.addColorStop(1, 'rgba(21, 128, 61, 0.5)');
      } else if (isConnected) {
        sphereGradient.addColorStop(0, 'rgba(191, 219, 254, 0.9)');
        sphereGradient.addColorStop(0.7, 'rgba(59, 130, 246, 0.7)');
        sphereGradient.addColorStop(1, 'rgba(29, 78, 216, 0.5)');
      } else {
        sphereGradient.addColorStop(0, 'rgba(243, 244, 246, 0.7)');
        sphereGradient.addColorStop(0.7, 'rgba(156, 163, 175, 0.5)');
        sphereGradient.addColorStop(1, 'rgba(75, 85, 99, 0.3)');
      }

      ctx.beginPath();
      ctx.arc(centerX, centerY, currentRadius, 0, Math.PI * 2);
      ctx.fillStyle = sphereGradient;
      ctx.fill();

      // Animated particles/waves around the sphere
      if (isConnected || isLoading) {
        const particleCount = isListening ? 12 : 8;
        const particleDistance = currentRadius + 30;
        
        for (let i = 0; i < particleCount; i++) {
          const angle = (i / particleCount) * Math.PI * 2 + timeRef.current;
          const x = centerX + Math.cos(angle) * particleDistance;
          const y = centerY + Math.sin(angle) * particleDistance;
          
          const opacity = Math.sin(timeRef.current * 2 + i) * 0.3 + 0.4;
          const particleRadius = isListening ? 3 + audioLevel * 5 : 2;
          
          ctx.beginPath();
          ctx.arc(x, y, particleRadius, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;
          ctx.fill();
        }
      }

      // Audio level visualization (concentric rings)
      if (isListening && audioLevel > 0.1) {
        const ringCount = 3;
        for (let i = 0; i < ringCount; i++) {
          const ringRadius = currentRadius + (i + 1) * 15 * audioLevel;
          const opacity = (1 - i / ringCount) * audioLevel * 0.5;
          
          ctx.beginPath();
          ctx.arc(centerX, centerY, ringRadius, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(34, 197, 94, ${opacity})`;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }

      // Loading spinner
      if (isLoading) {
        const spinnerRadius = currentRadius + 40;
        const spinnerAngle = timeRef.current * 4;
        
        ctx.beginPath();
        ctx.arc(centerX, centerY, spinnerRadius, spinnerAngle, spinnerAngle + Math.PI);
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.stroke();
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isConnected, isSpeaking, isListening, audioLevel, isLoading]);

  return (
    <div className={`relative flex items-center justify-center ${className}`}>
      <canvas
        ref={canvasRef}
        className="drop-shadow-2xl"
        style={{
          filter: 'drop-shadow(0 0 20px rgba(0, 0, 0, 0.3))',
        }}
      />
      
      {/* State indicator text */}
      <div className="absolute bottom-[-40px] text-center">
        <div className="text-white/60 text-sm font-medium">
          {isLoading ? 'Initializing...' :
           isSpeaking ? 'AI Speaking' :
           isListening ? 'Listening...' :
           isConnected ? 'Ready' :
           'Disconnected'}
        </div>
      </div>
    </div>
  );
}; 