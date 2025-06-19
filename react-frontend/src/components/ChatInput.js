import React, { useState } from 'react';
import { PaperAirplaneIcon } from '@heroicons/react/24/solid';
import SpeechInput from './SpeechInput';

const ChatInput = ({ onSendMessage, disabled = false }) => {
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
      setError('');
    }
  };

  const handleSpeechResult = (transcript) => {
    setMessage(prev => prev + (prev ? ' ' : '') + transcript);
    setError('');
  };

  const handleSpeechError = (errorMessage) => {
    setError(errorMessage);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="border-t bg-white p-4">
      {error && (
        <div className="mb-3 p-2 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      )}
      
      <form onSubmit={handleSubmit} className="flex items-end space-x-2">
        <div className="flex-1">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message or use voice input..."
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows="2"
            disabled={disabled}
          />
        </div>
        
        <div className="flex space-x-2">
          <SpeechInput
            onResult={handleSpeechResult}
            onError={handleSpeechError}
            disabled={disabled}
          />
          
          <button
            type="submit"
            disabled={!message.trim() || disabled}
            className="p-2 rounded-full bg-blue-500 text-white hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors duration-200"
          >
            <PaperAirplaneIcon className="h-5 w-5" />
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInput; 