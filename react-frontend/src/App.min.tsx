import React from 'react';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import Login from './components/Login';
import VoiceInterface from './components/VoiceInterface';
import AgentPage from './pages/Agent';

const App: React.FC = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
};

const AppContent: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuth();
  if (isLoading) return null;
  if (typeof window !== 'undefined' && window.location && window.location.pathname === '/agent') {
    return <AgentPage />;
  }
  return isAuthenticated ? <VoiceInterface /> : <Login />;
};

export default App;


