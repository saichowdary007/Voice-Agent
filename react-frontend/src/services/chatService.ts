import axiosInstance from './axiosInstance';
import { ChatRequest, ChatResponse } from '../types';

class ChatService {
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      const response = await axiosInstance.post<ChatResponse>('/api/chat', request);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Failed to send message');
    }
  }

  async getLanguages(): Promise<string[]> {
    try {
      const response = await axiosInstance.get<{ languages: string[] }>('/api/languages');
      return response.data.languages;
    } catch (error: any) {
      console.warn('Failed to fetch languages, using defaults');
      return ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh'];
    }
  }

  async getVoices(language?: string): Promise<string[]> {
    try {
      const params = language ? { language } : {};
      const response = await axiosInstance.get<{ voices: string[] }>('/api/voices', { params });
      return response.data.voices;
    } catch (error: any) {
      console.warn('Failed to fetch voices, using defaults');
      return ['default'];
    }
  }
}

export default new ChatService(); 