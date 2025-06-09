#!/usr/bin/env python3
"""
Setup script to create the required table in Supabase for the Voice Agent.
Run this script once to set up your database.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def setup_supabase():
    """Test Supabase connection and provide setup instructions."""
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        return False
    
    try:
        # Connect to Supabase
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Connected to Supabase successfully")
        
        # Test if table exists by trying to insert and delete a test record
        try:
            test_data = {
                'session_id': 'test_session',
                'role': 'user', 
                'text': 'Test message for setup',
                'embedding': '[0.1, 0.2, 0.3]'
            }
            
            insert_result = supabase.table('conversation_history').insert(test_data).execute()
            print("✅ Table exists and insert test successful")
            
            # Clean up test data
            supabase.table('conversation_history').delete().eq('session_id', 'test_session').execute()
            print("✅ Test cleanup successful")
            
            print("\n🎉 Supabase is ready to use!")
            print("Your Voice Agent can now store conversation history in Supabase.")
            return True
            
        except Exception as table_error:
            print(f"❌ Table doesn't exist or has wrong structure: {table_error}")
            print("\n📋 MANUAL SETUP REQUIRED:")
            print("Please follow these steps to set up your Supabase database:")
            print("\n1. Go to your Supabase dashboard: https://app.supabase.com")
            print("2. Select your project")
            print("3. Navigate to 'SQL Editor' in the left sidebar")
            print("4. Click 'New Query'")
            print("5. Copy and paste this SQL query:")
            
            print("""
-- Create the conversation_history table
CREATE TABLE IF NOT EXISTS conversation_history (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    text TEXT NOT NULL,
    embedding TEXT, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_conversation_session_id ON conversation_history(session_id);
CREATE INDEX IF NOT EXISTS idx_conversation_created_at ON conversation_history(created_at);

-- Enable Row Level Security (optional but recommended)
ALTER TABLE conversation_history ENABLE ROW LEVEL SECURITY;

-- Create a policy to allow all operations (adjust as needed for your security requirements)
CREATE POLICY "Allow all operations on conversation_history" ON conversation_history FOR ALL USING (true);
            """)
            
            print("\n6. Click 'Run' to execute the query")
            print("7. After successful execution, run this script again to test the connection")
            
            return False
        
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {e}")
        print("Please check your SUPABASE_URL and SUPABASE_KEY in the .env file")
        return False

if __name__ == "__main__":
    print("🚀 Setting up Supabase for Voice Agent...")
    success = setup_supabase()
    
    if success:
        print("\n✅ Setup complete! You can now run your Voice Agent with Supabase.")
    else:
        print("\n⚠️  Please complete the manual setup steps above and run this script again.") 