import axiosInstance from './axiosInstance';
import { AuthResponse, User } from '../types';

interface LoginCredentials {
  email: string;
  password: string;
}

interface SignupCredentials extends LoginCredentials {}

class AuthService {
  async signup(credentials: SignupCredentials): Promise<AuthResponse> {
    try {
      const response = await axiosInstance.post<AuthResponse>('/api/auth/signup', credentials);
      this.setAuthData(response.data);
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 422) {
        const detail = error.response.data?.detail;
        if (Array.isArray(detail)) {
          const emailError = detail.find((err: any) => err.loc?.includes('email'));
          if (emailError) {
            throw new Error('Please enter a valid email address (e.g., user@example.com)');
          }
        }
        throw new Error('Please check your input and try again');
      }
      throw new Error(error.response?.data?.detail || 'Signup failed');
    }
  }

  async signin(credentials: LoginCredentials): Promise<AuthResponse> {
    try {
      const response = await axiosInstance.post<AuthResponse>('/api/auth/signin', credentials);
      this.setAuthData(response.data);
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 422) {
        const detail = error.response.data?.detail;
        if (Array.isArray(detail)) {
          const emailError = detail.find((err: any) => err.loc?.includes('email'));
          if (emailError) {
            throw new Error('Please enter a valid email address (e.g., user@example.com)');
          }
        }
        throw new Error('Please check your input and try again');
      }
      throw new Error(error.response?.data?.detail || 'Login failed');
    }
  }

  async loginAsGuest(): Promise<AuthResponse> {
    try {
      const guestEmail = `guest${Math.floor(Math.random() * 10000)}@example.com`;
      const response = await axiosInstance.post<AuthResponse>('/api/auth/debug-signin', {
        email: guestEmail,
        password: 'dummy',
      });
      this.setAuthData(response.data);
      return response.data;
    } catch (error: any) {
      throw new Error(error.response?.data?.detail || 'Guest login failed');
    }
  }

  private setAuthData(authData: AuthResponse): void {
    localStorage.setItem('access_token', authData.access_token);
    if (authData.refresh_token) {
      localStorage.setItem('refresh_token', authData.refresh_token);
    }
    localStorage.setItem('user', JSON.stringify(authData.user));
  }

  logout(): void {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
  }

  getCurrentUser(): User | null {
    try {
      const userStr = localStorage.getItem('user');
      return userStr ? JSON.parse(userStr) : null;
    } catch {
      return null;
    }
  }

  getAccessToken(): string | null {
    return localStorage.getItem('access_token');
  }

  isAuthenticated(): boolean {
    return !!this.getAccessToken();
  }
}

export default new AuthService(); 