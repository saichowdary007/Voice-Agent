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
    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode or _should_use_demo_mode()
        
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
            if email in self.demo_users:
                raise Exception(f"User {email} already exists")
            
            user = MockUser(email)
            self.demo_users[email] = {
                'user': user,
                'password_hash': _hash_password(password)
            }
            
            session = MockSession(user)
            return session
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
            if email not in self.demo_users:
                raise Exception(f"User {email} not found")
            
            stored_user = self.demo_users[email]
            if stored_user['password_hash'] != _hash_password(password):
                raise Exception(f"Invalid password for {email}")
            
            session = MockSession(stored_user['user'])
            return session
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
        
        SECURITY FIX: Demo tokens are only accepted in demo mode or DEBUG_MODE.
        """
        if not token:
            raise ValueError("Token cannot be empty")

        # Check for demo tokens - accept in demo mode OR debug mode
        if token.startswith("demo_token_"):
            from src.config import DEBUG_MODE
            
            if not self.demo_mode and not DEBUG_MODE:
                raise Exception("Demo tokens not allowed in production mode")
                
            user_id = token[len("demo_token_"):]
            
            if self.demo_mode:
                for user_record in getattr(self, "demo_users", {}).values():
                    user = user_record.get("user")
                    if user and user.id == user_id:
                        return user
                raise Exception("Invalid demo token")
            else:
                try:
                    email = f"debug-user-{user_id}@example.com"
                    mock_user = MockUser(email)
                    mock_user.id = user_id
                    return mock_user
                except Exception:
                    raise Exception("Invalid demo token")
        
        # Handle real Supabase tokens
        if not self.demo_mode:
            try:
                # Supabase python client exposes get_user when a valid access token is supplied
                response = self.auth.get_user(token)
                return response.user if response and response.user else None
            except Exception as e:
                print(f"❌ Token verification failed: {e}")
                # Development fallback: accept any well-formed JWT when DEBUG_MODE is enabled
                try:
                    from src.config import DEBUG_MODE
                    if DEBUG_MODE:
                        from jose import jwt
                        try:
                            claims = jwt.get_unverified_claims(token)
                            sub = claims.get("sub", "debug-user")
                            email = claims.get("email", f"{sub}@example.com")
                            mock_user = MockUser(email)
                            mock_user.id = sub
                            print(f"🔓 DEBUG_MODE active – bypassing Supabase verification for user {email}")
                            return mock_user
                        except Exception as parse_err:
                            print(f"⚠️ Failed to parse JWT in DEBUG_MODE fallback: {parse_err}")
                            # If JWT parsing fails, still try to create a mock user from the token
                            mock_user = MockUser("debug-user@example.com")
                            mock_user.id = "debug-user"
                            print(f"🔓 DEBUG_MODE fallback – created mock user for unparseable token")
                            return mock_user
                except Exception:
                    # If DEBUG_MODE import fails or disabled, just fall through
                    pass
                return None
        
        # DEBUG_MODE fallback when in demo mode (or general fallback)
        try:
            from src.config import DEBUG_MODE
            if DEBUG_MODE:
                from jose import jwt
                try:
                    claims = jwt.get_unverified_claims(token)
                    sub = claims.get("sub", "debug-user")
                    email = claims.get("email", f"{sub}@example.com")
                    mock_user = MockUser(email)
                    mock_user.id = sub
                    print(f"🔓 DEBUG_MODE fallback – accepting JWT for user {email}")
                    return mock_user
                except Exception as parse_err:
                    print(f"⚠️ Failed to parse JWT in DEBUG_MODE fallback (demo_mode): {parse_err}")
        except Exception:
            pass

        raise ValueError("Invalid token")