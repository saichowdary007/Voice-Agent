'use client';

import React, { useEffect, useRef } from 'react';
import { TranscriptDisplayProps } from '../lib/types';

export const TranscriptDisplay: React.FC<TranscriptDisplayProps> = ({
  partialText,
  finalTexts,
  aiResponse,
  isAiResponding,
  maxLines = 10,
  className = '',
  transparent = false,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new content is added
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [partialText, finalTexts, aiResponse]);

  if (transparent) {
    return (
      <div className={`transcript-display-transparent ${className}`}>
        {/* Transparent overlay mode - minimal UI */}
        <div className="space-y-2 max-h-40 overflow-hidden">
          {/* Current partial text */}
          {partialText && (
            <div className="text-center">
              <div className="inline-block bg-black/30 backdrop-blur-sm text-white/90 text-sm px-4 py-2 rounded-full max-w-md">
                <span className="opacity-75">{partialText}</span>
                <span className="ml-2 inline-block w-1 h-4 bg-white/60 animate-pulse"></span>
              </div>
            </div>
          )}

          {/* AI Response */}
          {aiResponse && (
            <div className="text-center">
              <div className="inline-block bg-purple-500/20 backdrop-blur-sm text-white/90 text-sm px-4 py-2 rounded-full max-w-md">
                <span className="text-purple-200">AI:</span> {aiResponse}
                {isAiResponding && (
                  <span className="ml-2 flex inline-flex space-x-1">
                    <span className="w-1 h-1 bg-purple-300 rounded-full animate-bounce"></span>
                    <span className="w-1 h-1 bg-purple-300 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></span>
                    <span className="w-1 h-1 bg-purple-300 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Show last final text if no current interaction */}
          {!partialText && !aiResponse && finalTexts.length > 0 && (
            <div className="text-center">
              <div className="inline-block bg-black/20 backdrop-blur-sm text-white/70 text-xs px-3 py-1 rounded-full max-w-md">
                Last: {finalTexts[finalTexts.length - 1]}
              </div>
            </div>
          )}

          {/* Empty state */}
          {!partialText && !aiResponse && finalTexts.length === 0 && (
            <div className="text-center">
              <div className="inline-block bg-black/20 backdrop-blur-sm text-white/50 text-xs px-3 py-1 rounded-full">
                Start speaking...
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={`transcript-display ${className}`}>
      <div className="bg-white rounded-lg border shadow-sm">
        {/* Header */}
        <div className="px-4 py-2 border-b bg-gray-50 rounded-t-lg">
          <h3 className="text-sm font-medium text-gray-700 flex items-center">
            <span className="mr-2">💬</span>
            Live Transcript
            {isAiResponding && (
              <span className="ml-2 text-xs text-blue-600 animate-pulse">
                AI responding...
              </span>
            )}
          </h3>
        </div>

        {/* Transcript Content */}
        <div
          ref={scrollRef}
          className="p-4 h-64 overflow-y-auto"
          style={{
            maxHeight: `${maxLines * 1.5}rem`,
          }}
        >
          {/* No conversation yet */}
          {finalTexts.length === 0 && !partialText && !aiResponse && (
            <div className="text-center text-gray-400 mt-8">
              <div className="text-2xl mb-2">🎯</div>
              <p>Start speaking to begin your conversation</p>
              <p className="text-xs mt-1">Your voice will appear here in real-time</p>
            </div>
          )}

          {/* Conversation History */}
          <div className="space-y-3">
            {finalTexts.map((text, index) => (
              <div key={index} className="space-y-2">
                {/* User Message */}
                <div className="flex justify-end">
                  <div className="max-w-xs lg:max-w-md bg-blue-500 text-white rounded-lg px-3 py-2">
                    <p className="text-sm">{text}</p>
                    <span className="text-xs opacity-75">You</span>
                  </div>
                </div>

                {/* AI Response (if available) */}
                {index < finalTexts.length - 1 && (
                  <div className="flex justify-start">
                    <div className="max-w-xs lg:max-w-md bg-gray-100 text-gray-800 rounded-lg px-3 py-2">
                      <p className="text-sm">AI response would go here</p>
                      <span className="text-xs text-gray-500">Assistant</span>
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Current Partial User Input */}
            {partialText && (
              <div className="flex justify-end">
                <div className="max-w-xs lg:max-w-md bg-blue-400 text-white rounded-lg px-3 py-2 opacity-75">
                  <p className="text-sm">{partialText}</p>
                  <span className="text-xs opacity-75 flex items-center">
                    You
                    <span className="ml-1 w-1 h-1 bg-white rounded-full animate-pulse"></span>
                  </span>
                </div>
              </div>
            )}

            {/* Current AI Response */}
            {aiResponse && (
              <div className="flex justify-start">
                <div className="max-w-xs lg:max-w-md bg-gray-100 text-gray-800 rounded-lg px-3 py-2">
                  <p className="text-sm whitespace-pre-wrap">{aiResponse}</p>
                  <span className="text-xs text-gray-500 flex items-center">
                    Assistant
                    {isAiResponding && (
                      <span className="ml-1 flex space-x-1">
                        <span className="w-1 h-1 bg-gray-400 rounded-full animate-bounce"></span>
                        <span className="w-1 h-1 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></span>
                        <span className="w-1 h-1 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                      </span>
                    )}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Status Bar */}
        <div className="px-4 py-2 border-t bg-gray-50 rounded-b-lg">
          <div className="flex justify-between items-center text-xs text-gray-500">
            <span>
              {finalTexts.length} message{finalTexts.length !== 1 ? 's' : ''}
            </span>
            <div className="flex items-center space-x-4">
              {partialText && (
                <span className="text-blue-600 flex items-center">
                  <span className="w-2 h-2 bg-blue-600 rounded-full mr-1 animate-pulse"></span>
                  Listening...
                </span>
              )}
              {isAiResponding && (
                <span className="text-green-600 flex items-center">
                  <span className="w-2 h-2 bg-green-600 rounded-full mr-1 animate-pulse"></span>
                  Thinking...
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 