import React, { useState, useEffect, useRef } from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { useWebSocket } from './hooks/useWebSocket';
import Login from './components/Login';
import chatService from './services/chatService';
import { ShimmerButton } from './components/magicui/shimmer-button';
import { RippleButton } from './components/magicui/ripple-button';
import AnimatedGradientText from './components/magicui/animated-gradient-text';
import OrbitingCircles from './components/magicui/orbiting-circles';
import { ChatMessage, VoiceSettings } from './types';
import { cn, formatTime, generateId } from './lib/utils';

interface ChatInterfaceProps {
  className?: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ className }) => {
  const { user, logout, isAuthenticated } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [voiceSettings, setVoiceSettings] = useState<VoiceSettings>({
    language: 'en',
    voice: 'default',
    speed: 1.0,
    pitch: 1.0,
  });
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const { connectionStatus, isConnected, error: wsError } = useWebSocket({
    onMessage: (message) => {
      console.log('WebSocket message received:', message);
      // Handle real-time voice messages here
    },
  }, isAuthenticated);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!inputText.trim() || isLoading) {
      return;
    }

    const userMessage: ChatMessage = {
      id: generateId(),
      text: inputText.trim(),
      isUser: true,
      timestamp: new Date(),
      language: voiceSettings.language,
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await chatService.sendMessage({
        text: userMessage.text,
        language: voiceSettings.language,
      });

      const assistantMessage: ChatMessage = {
        id: generateId(),
        text: response.response,
        isUser: false,
        timestamp: new Date(),
        language: response.language || voiceSettings.language,
        audioUrl: response.audio_url,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err: any) {
      setError(err.message);
      console.error('Chat error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e as any);
    }
  };

  const ConnectionIndicator = () => {
    const getStatusColor = () => {
      switch (connectionStatus) {
        case 'connected': return 'bg-green-500';
        case 'connecting': return 'bg-yellow-500 animate-pulse';
        case 'disconnected': return 'bg-gray-500';
        case 'error': return 'bg-red-500';
        default: return 'bg-gray-500';
      }
    };

    const getStatusText = () => {
      switch (connectionStatus) {
        case 'connected': return 'Connected';
        case 'connecting': return 'Connecting...';
        case 'disconnected': return 'Disconnected';
        case 'error': return 'Connection Error';
        default: return 'Unknown';
      }
    };

    return (
      <div className="flex items-center space-x-2">
        <div className={cn("w-2 h-2 rounded-full", getStatusColor())} />
        <span className="text-xs text-gray-400">{getStatusText()}</span>
      </div>
    );
  };

  return (
    <div className={cn("min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900", className)}>
      {/* Background Effects */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/3 left-1/5 w-64 h-64 bg-purple-500/10 rounded-full filter blur-xl animate-pulse-slow" />
        <div className="absolute bottom-1/3 right-1/5 w-80 h-80 bg-blue-500/10 rounded-full filter blur-xl animate-pulse-slow delay-1000" />
      </div>

      {/* Header */}
      <header className="relative z-10 bg-black/20 backdrop-blur-lg border-b border-white/10">
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          {/* Logo with Orbiting Circles */}
          <div className="relative w-16 h-16">
            <div className="absolute inset-0 flex items-center justify-center">
              <AnimatedGradientText>
                <span className="text-sm font-bold">🎤 VA</span>
              </AnimatedGradientText>
            </div>
            <OrbitingCircles
              className="h-3 w-3 border-none bg-transparent"
              duration={15}
              delay={0}
              radius={25}
            >
              <div className="h-2 w-2 rounded-full bg-gradient-to-r from-purple-400 to-pink-400" />
            </OrbitingCircles>
            <OrbitingCircles
              className="h-2 w-2 border-none bg-transparent"
              duration={15}
              delay={7.5}
              radius={25}
              reverse
            >
              <div className="h-1.5 w-1.5 rounded-full bg-gradient-to-r from-blue-400 to-cyan-400" />
            </OrbitingCircles>
          </div>

          <div className="flex items-center space-x-4">
            <ConnectionIndicator />
            <div className="text-sm text-gray-300">
              {user?.email && (
                <span className="hidden sm:inline">
                  {user.email.startsWith('guest') ? 'Guest User' : user.email}
                </span>
              )}
            </div>
            <RippleButton
              onClick={logout}
              className="px-4 py-2 bg-white/10 hover:bg-white/20 text-white text-sm border border-white/20"
              rippleColor="rgba(255, 255, 255, 0.3)"
            >
              Logout
            </RippleButton>
          </div>
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="relative z-10 flex flex-col h-[calc(100vh-80px)]">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.length === 0 ? (
              <div className="text-center py-12">
                <div className="mb-6">
                  <div className="w-20 h-20 mx-auto bg-gradient-to-r from-purple-400 to-pink-400 rounded-full flex items-center justify-center text-3xl">
                    🎤
                  </div>
                </div>
                <AnimatedGradientText>
                  <span className="text-xl font-semibold">Welcome to Voice Agent</span>
                </AnimatedGradientText>
                <p className="text-gray-400 mt-4 max-w-md mx-auto">
                  Start a conversation with your AI assistant. Type a message below or use voice commands.
                </p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex",
                    message.isUser ? "justify-end" : "justify-start"
                  )}
                >
                  <div
                    className={cn(
                      "max-w-xs sm:max-w-md lg:max-w-lg xl:max-w-xl rounded-xl px-4 py-3 shadow-lg",
                      message.isUser
                        ? "bg-gradient-to-r from-purple-600 to-pink-600 text-white"
                        : "bg-white/10 backdrop-blur-lg text-gray-100 border border-white/20"
                    )}
                  >
                    <p className="text-sm leading-relaxed">{message.text}</p>
                    <div className="flex items-center justify-between mt-2">
                      <span className="text-xs opacity-70">
                        {formatTime(message.timestamp)}
                      </span>
                      {message.audioUrl && (
                        <button
                          className="text-xs bg-white/20 hover:bg-white/30 px-2 py-1 rounded transition-colors"
                          onClick={() => {
                            const audio = new Audio(message.audioUrl);
                            audio.play().catch(console.error);
                          }}
                        >
                          🔊 Play
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white/10 backdrop-blur-lg rounded-xl px-4 py-3 border border-white/20">
                  <div className="flex items-center space-x-2">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
                    </div>
                    <span className="text-sm text-gray-300">AI is thinking...</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Error Banner */}
        {(error || wsError) && (
          <div className="px-4 py-3 bg-red-500/20 border-t border-red-500/30">
            <div className="max-w-4xl mx-auto">
              <p className="text-red-200 text-sm">
                {error || wsError}
              </p>
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="bg-black/20 backdrop-blur-lg border-t border-white/10 px-4 py-4">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={handleSendMessage} className="flex items-end space-x-4">
              <div className="flex-1">
                <textarea
                  ref={inputRef}
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Type your message... (Press Enter to send, Shift+Enter for new line)"
                  className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all resize-none"
                  rows={1}
                  style={{
                    minHeight: '44px',
                    maxHeight: '120px',
                    height: 'auto',
                  }}
                  disabled={isLoading}
                />
              </div>
              
              <ShimmerButton
                type="submit"
                disabled={!inputText.trim() || isLoading}
                className="px-6 py-3 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
                shimmerColor="#ffffff"
                shimmerDuration="2s"
              >
                {isLoading ? (
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                ) : (
                  'Send'
                )}
              </ShimmerButton>
            </form>
            
            {/* Voice Settings (collapsed for now) */}
            <div className="mt-3 flex items-center justify-center space-x-4 text-xs text-gray-400">
              <span>Language: {voiceSettings.language.toUpperCase()}</span>
              <span>•</span>
              <span>WebSocket: {isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
};

const AppContent: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 relative">
            <OrbitingCircles
              className="h-4 w-4 border-none bg-transparent"
              duration={10}
              delay={0}
              radius={30}
            >
              <div className="h-3 w-3 rounded-full bg-gradient-to-r from-purple-400 to-pink-400" />
            </OrbitingCircles>
            <OrbitingCircles
              className="h-3 w-3 border-none bg-transparent"
              duration={10}
              delay={5}
              radius={30}
              reverse
            >
              <div className="h-2 w-2 rounded-full bg-gradient-to-r from-blue-400 to-cyan-400" />
            </OrbitingCircles>
          </div>
          <AnimatedGradientText>
            <span className="text-lg font-semibold">Loading Voice Agent...</span>
          </AnimatedGradientText>
        </div>
      </div>
    );
  }

  return isAuthenticated ? <ChatInterface /> : <Login />;
};

export default App; 