import React from 'react';
import SpeechOutput from './SpeechOutput';

const ChatMessage = ({ message, isUser, timestamp }) => {
  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
        isUser 
          ? 'bg-blue-500 text-white' 
          : 'bg-gray-200 text-gray-800'
      }`}>
        <div className="flex items-start justify-between">
          <p className="text-sm flex-1">{message}</p>
          {!isUser && (
            <div className="ml-2 flex-shrink-0">
              <SpeechOutput text={message} />
            </div>
          )}
        </div>
        {timestamp && (
          <p className={`text-xs mt-1 ${
            isUser ? 'text-blue-200' : 'text-gray-500'
          }`}>
            {formatTime(timestamp)}
          </p>
        )}
      </div>
    </div>
  );
};

export default ChatMessage; 