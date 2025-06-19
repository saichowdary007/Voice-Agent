import React, { useState, useRef, useEffect } from 'react';
import { SpeakerWaveIcon, SpeakerXMarkIcon } from '@heroicons/react/24/solid';

const SpeechOutput = ({ text, disabled = false }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isSupported, setIsSupported] = useState(true);
  const utteranceRef = useRef(null);

  useEffect(() => {
    // Check for Web Speech Synthesis API support
    if (!window.speechSynthesis) {
      setIsSupported(false);
      return;
    }

    return () => {
      // Cleanup on unmount
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  const startSpeaking = () => {
    if (!isSupported || !text || isPlaying || disabled) {
      return;
    }

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    // Create new utterance
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = navigator.language || 'en-US';
    utterance.volume = 1;
    utterance.rate = 1;
    utterance.pitch = 1;

    // Find appropriate voice
    const voices = window.speechSynthesis.getVoices();
    let voice = voices.find(v => v.lang === utterance.lang);
    if (!voice) {
      voice = voices.find(v => v.lang.startsWith('en'));
    }
    if (voice) {
      utterance.voice = voice;
    }

    // Event handlers
    utterance.onstart = () => {
      setIsPlaying(true);
    };

    utterance.onend = () => {
      setIsPlaying(false);
    };

    utterance.onerror = () => {
      setIsPlaying(false);
    };

    utteranceRef.current = utterance;
    window.speechSynthesis.speak(utterance);
  };

  const stopSpeaking = () => {
    if (window.speechSynthesis && isPlaying) {
      window.speechSynthesis.cancel();
      setIsPlaying(false);
    }
  };

  const handleClick = () => {
    if (isPlaying) {
      stopSpeaking();
    } else {
      startSpeaking();
    }
  };

  if (!isSupported) {
    return (
      <button
        disabled
        className="p-2 rounded-full bg-gray-300 text-gray-500 cursor-not-allowed"
        title="Speech synthesis not supported"
      >
        <SpeakerXMarkIcon className="h-5 w-5" />
      </button>
    );
  }

  if (!text) {
    return null;
  }

  return (
    <button
      onClick={handleClick}
      disabled={disabled}
      className={`p-2 rounded-full transition-all duration-200 ${
        isPlaying
          ? 'bg-green-500 text-white animate-pulse shadow-lg'
          : disabled
          ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
          : 'bg-green-500 text-white hover:bg-green-600 shadow-md hover:shadow-lg'
      }`}
      title={isPlaying ? 'Stop speaking' : 'Read aloud'}
    >
      {isPlaying ? (
        <SpeakerXMarkIcon className="h-5 w-5" />
      ) : (
        <SpeakerWaveIcon className="h-5 w-5" />
      )}
    </button>
  );
};

export default SpeechOutput; 