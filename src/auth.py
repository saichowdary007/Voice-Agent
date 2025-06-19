"""
Handles user authentication using Supabase Auth.
"""
import os
import hashlib
from typing import Optional, NamedTuple
from datetime import datetime, timedelta

# Mock classes for demo mode
class MockUser:
    def __init__(self, email: str):
        self.id = f"user_{hash(email) % 10000}"
        self.email = email
        self.created_at = datetime.utcnow().isoformat()

class MockSession:
    def __init__(self, user: MockUser):
        self.user = user
        self.access_token = f"demo_token_{user.id}"
        self.refresh_token = f"demo_refresh_{user.id}"
        self.expires_at = datetime.utcnow() + timedelta(hours=1)

def _hash_password(password: str) -> str:
    """Hash password for demo mode storage."""
    return hashlib.sha256(password.encode()).hexdigest()

def _should_use_demo_mode() -> bool:
    """Determine if we should use demo mode."""
    try:
        from src.config import SUPABASE_URL, SUPABASE_KEY, USE_SUPABASE_ENV
        
        # If USE_SUPABASE is explicitly set to false, use demo mode
        if not USE_SUPABASE_ENV:
            return True
            
        # If no Supabase credentials, use demo mode
        if not SUPABASE_URL or not SUPABASE_KEY:
            return True
            
        # Check for placeholder values that indicate demo setup
        placeholder_indicators = [
            'demo', 'localhost', 'your_', 'demo_', 'placeholder', 'example'
        ]
        
        url_lower = str(SUPABASE_URL).lower()
        key_lower = str(SUPABASE_KEY).lower()
        
        for indicator in placeholder_indicators:
            if indicator in url_lower or indicator in key_lower:
                return True
        
        # Check for specific demo patterns
        if (SUPABASE_URL.startswith('https://demo') or 
            SUPABASE_URL.startswith('demo_mode') or
            len(SUPABASE_KEY) < 50):  # Real Supabase keys are typically much longer
            return True
            
        return False
    except ImportError:
        return True

class AuthManager:
    """
    Manages user sign-up, sign-in, and session state with Supabase.
    Includes demo mode for testing without real Supabase credentials.
    """
    def __init__(self):
        self.demo_mode = _should_use_demo_mode()
        
        if self.demo_mode:
            print("⚠️ Running in DEMO mode - not using real Supabase")
            self.demo_users = {}  # Store demo users in memory
        else:
            from supabase import create_client, Client
            from src.config import SUPABASE_URL, SUPABASE_KEY
            
            if not SUPABASE_URL or not SUPABASE_KEY:
                raise ValueError("Supabase URL and Key must be set in the .env file.")
            self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            self.auth = self.client.auth

    def sign_up(self, email: str, password: str) -> Optional[MockSession]:
        """
        Signs up a new user.

        Returns:
            The user session object on success, None on failure.
        """
        if self.demo_mode:
            try:
                if email in self.demo_users:
                    print(f"❌ Demo sign-up failed: User {email} already exists")
                    return None
                
                # Create demo user with hashed password
                user = MockUser(email)
                self.demo_users[email] = {
                    'user': user,
                    'password_hash': _hash_password(password)  # Hash passwords in demo mode
                }
                
                session = MockSession(user)
                print(f"✅ Demo sign-up successful for {email}")
                return session
            except Exception as e:
                print(f"❌ Demo sign-up failed: {e}")
                return None
        else:
            try:
                credentials = {"email": email, "password": password}
                print(f"Attempting to sign up with: {credentials}")
                response = self.auth.sign_up(credentials)
                print("✅ Sign-up successful! Please check your email for verification.")
                return response.session
            except Exception as e:
                print(f"❌ Sign-up failed: {e}")
                return None

    def sign_in(self, email: str, password: str) -> Optional[MockSession]:
        """
        Signs in an existing user.

        Returns:
            The user session object on success, None on failure.
        """
        if self.demo_mode:
            try:
                if email not in self.demo_users:
                    # Auto-create demo user for convenience
                    print(f"🔧 Auto-creating demo user for {email}")
                    user = MockUser(email)
                    self.demo_users[email] = {
                        'user': user,
                        'password_hash': _hash_password(password)
                    }
                
                stored_user = self.demo_users[email]
                if stored_user['password_hash'] == _hash_password(password):
                    session = MockSession(stored_user['user'])
                    print(f"✅ Demo login successful! Welcome back, {email}")
                    return session
                else:
                    print(f"❌ Demo login failed: Invalid password for {email}")
                    return None
            except Exception as e:
                print(f"❌ Demo login failed: {e}")
                return None
        else:
            try:
                response = self.auth.sign_in_with_password({"email": email, "password": password})
                print(f"✅ Login successful! Welcome back, {response.user.email}.")
                return response.session
            except Exception as e:
                print(f"❌ Login failed: {e}")
                return None

    def sign_out(self):
        """Signs out the current user."""
        if self.demo_mode:
            try:
                print("👋 Demo user signed out successfully.")
            except Exception as e:
                print(f"❌ Demo sign-out failed: {e}")
        else:
            try:
                self.auth.sign_out()
                print("👋 You have been successfully signed out.")
            except Exception as e:
                print(f"❌ Sign-out failed: {e}")

    def refresh_session(self, refresh_token: str):
        """Refresh user session using the provided refresh token.

        Returns:
            A new session object on success, or None if the refresh token is invalid.
        """
        if self.demo_mode:
            try:
                # Expect refresh tokens in the format "demo_refresh_<user_id>"
                prefix = "demo_refresh_"
                if not refresh_token.startswith(prefix):
                    print("❌ Demo refresh failed: Invalid token format")
                    return None
                user_id = refresh_token[len(prefix):]
                # Find the user with the matching ID
                for user_record in self.demo_users.values():
                    user = user_record.get("user")
                    if user and user.id == user_id:
                        new_session = MockSession(user)
                        print(f"✅ Demo session refreshed for {user.email}")
                        return new_session
                print("❌ Demo refresh failed: User not found")
                return None
            except Exception as e:
                print(f"❌ Demo refresh failed: {e}")
                return None
        else:
            try:
                # Supabase Python client provides refresh_session method
                response = self.auth.refresh_session(refresh_token)
                return response.session if response else None
            except Exception as e:
                print(f"❌ Session refresh failed: {e}")
                return None

    def verify_token(self, token: str):
        """Verify a JWT (or demo) access token and return the corresponding user object.

        In demo mode we expect tokens that start with the prefix ``demo_token_`` followed
        by the user id. For Supabase we delegate to the Auth client. Returns ``None`` if
        the token is invalid or expired.
        
        SECURITY FIX: Demo tokens are only accepted in demo mode.
        """
        if not token:
            return None

        # Check for demo tokens - ONLY accept in demo mode for security
        if token.startswith("demo_token_"):
            if not self.demo_mode:
                # Security fix: reject demo tokens in production mode
                print("❌ Demo tokens not allowed in production mode")
                return None
                
            user_id = token[len("demo_token_"):]
            
            if self.demo_mode:
                # Look up the user in the in-memory store
                for user_record in getattr(self, "demo_users", {}).values():
                    user = user_record.get("user")
                    if user and user.id == user_id:
                        return user
                return None
            else:
                # Create a temporary mock user for debug authentication
                # This allows debug endpoints to work even with real Supabase config
                try:
                    email = f"debug-user-{user_id}@example.com"
                    mock_user = MockUser(email)
                    mock_user.id = user_id  # Override with the token's user ID
                    return mock_user
                except Exception:
                    return None
        
        # Handle real Supabase tokens
        if not self.demo_mode:
            try:
                # Supabase python client exposes get_user when a valid access token is supplied
                response = self.auth.get_user(token)
                return response.user if response and response.user else None
            except Exception as e:
                print(f"❌ Token verification failed: {e}")
                return None
        
        return None 