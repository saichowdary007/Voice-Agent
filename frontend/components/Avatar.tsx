'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

interface AvatarProps {
  isVisible: boolean;
  isSpeaking: boolean;
  lipSyncData?: number[];
  emotion?: 'neutral' | 'happy' | 'thinking' | 'speaking';
  className?: string;
}

export const Avatar: React.FC<AvatarProps> = ({
  isVisible,
  isSpeaking,
  lipSyncData = [],
  emotion = 'neutral',
  className = '',
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene>();
  const rendererRef = useRef<THREE.WebGLRenderer>();
  const avatarRef = useRef<THREE.Group>();
  const animationIdRef = useRef<number>();

  // Initialize Three.js scene
  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      mountRef.current.clientWidth / mountRef.current.clientHeight,
      0.1,
      1000
    );
    camera.position.z = 5;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
    renderer.setClearColor(0x000000, 0); // Transparent background
    mountRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // Create simple avatar
    const avatarGroup = new THREE.Group();
    avatarRef.current = avatarGroup;

    // Head
    const headGeometry = new THREE.SphereGeometry(1, 32, 32);
    const headMaterial = new THREE.MeshLambertMaterial({ color: 0xffdbac });
    const head = new THREE.Mesh(headGeometry, headMaterial);
    avatarGroup.add(head);

    // Eyes
    const eyeGeometry = new THREE.SphereGeometry(0.1, 16, 16);
    const eyeMaterial = new THREE.MeshLambertMaterial({ color: 0x000000 });
    
    const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    leftEye.position.set(-0.3, 0.2, 0.8);
    avatarGroup.add(leftEye);

    const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
    rightEye.position.set(0.3, 0.2, 0.8);
    avatarGroup.add(rightEye);

    // Mouth
    const mouthGeometry = new THREE.RingGeometry(0.1, 0.2, 16);
    const mouthMaterial = new THREE.MeshLambertMaterial({ 
      color: 0x000000,
      side: THREE.DoubleSide 
    });
    const mouth = new THREE.Mesh(mouthGeometry, mouthMaterial);
    mouth.position.set(0, -0.3, 0.8);
    mouth.name = 'mouth';
    avatarGroup.add(mouth);

    scene.add(avatarGroup);

    // Animation loop
    const animate = () => {
      animationIdRef.current = requestAnimationFrame(animate);

      // Subtle head movement when thinking
      if (emotion === 'thinking') {
        avatarGroup.rotation.y = Math.sin(Date.now() * 0.001) * 0.1;
      }

      // Mouth animation when speaking
      if (isSpeaking) {
        const mouth = avatarGroup.getObjectByName('mouth') as THREE.Mesh;
        if (mouth) {
          const scale = 1 + Math.sin(Date.now() * 0.01) * 0.3;
          mouth.scale.set(scale, scale, 1);
        }
      }

      // Emotion-based expressions
      switch (emotion) {
        case 'happy':
          // Slight upward rotation for smile
          avatarGroup.rotation.z = Math.sin(Date.now() * 0.002) * 0.05;
          break;
        case 'speaking':
          // More dynamic movement when speaking
          avatarGroup.rotation.x = Math.sin(Date.now() * 0.003) * 0.1;
          break;
        case 'neutral':
        default:
          // Return to neutral position
          avatarGroup.rotation.x *= 0.95;
          avatarGroup.rotation.y *= 0.95;
          avatarGroup.rotation.z *= 0.95;
          break;
      }

      renderer.render(scene, camera);
    };

    animate();

    // Handle resize
    const handleResize = () => {
      if (mountRef.current && rendererRef.current) {
        const width = mountRef.current.clientWidth;
        const height = mountRef.current.clientHeight;
        
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        rendererRef.current.setSize(width, height);
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      
      if (animationIdRef.current) {
        cancelAnimationFrame(animationIdRef.current);
      }
      
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      
      renderer.dispose();
    };
  }, []);

  // Update animations based on props
  useEffect(() => {
    if (!avatarRef.current) return;

    const mouth = avatarRef.current.getObjectByName('mouth') as THREE.Mesh;
    if (mouth) {
      if (isSpeaking) {
        // Animate mouth based on lip sync data or default animation
        if (lipSyncData.length > 0) {
          const currentFrame = Math.floor((Date.now() / 50) % lipSyncData.length);
          const intensity = lipSyncData[currentFrame] || 0;
          mouth.scale.set(1 + intensity, 1 + intensity, 1);
        }
      } else {
        // Return mouth to normal size
        mouth.scale.set(1, 1, 1);
      }
    }
  }, [isSpeaking, lipSyncData]);

  if (!isVisible) {
    return null;
  }

  return (
    <div className={`avatar-container ${className}`}>
      <div
        ref={mountRef}
        className="w-48 h-48 mx-auto rounded-full border-4 border-gray-200 shadow-lg overflow-hidden bg-gradient-to-b from-blue-50 to-blue-100"
        style={{ minHeight: '192px' }}
      />
      
      {/* Avatar Status */}
      <div className="text-center mt-2">
        <div className="flex justify-center items-center space-x-2 text-sm text-gray-600">
          <span className={`w-2 h-2 rounded-full ${
            isSpeaking ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
          }`} />
          <span>
            {emotion === 'thinking' ? 'Thinking...' :
             emotion === 'speaking' || isSpeaking ? 'Speaking' :
             emotion === 'happy' ? 'Happy' :
             'Ready'}
          </span>
        </div>
      </div>
    </div>
  );
}; 