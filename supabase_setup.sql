-- =================================================================
--  Supabase Vector Search Setup for the Voice Agent (v2)
-- =================================================================
--  This script completely resets and sets up the necessary table
--  and function for enabling conversation memory.
--
--  Instructions:
--  1. Navigate to the "SQL Editor" in your Supabase project dashboard.
--  2. Click "New query".
--  3. Copy and paste the entire content of this file into the editor.
--  4. Click "RUN".
--
--  This will delete any existing conversation history.
-- =================================================================

-- 1. Enable the pgvector extension
--    (pgvector is already enabled on new Supabase projects)
create extension if not exists vector;

-- 2. Drop existing objects to ensure a clean slate.
--    The CASCADE will also drop the 'match_conversations' function if it depends on the table.
DROP TABLE IF EXISTS public.conversation_history CASCADE;
DROP FUNCTION IF EXISTS public.match_conversations;

-- 3. Create the conversation history table
--    This table will store messages, roles, and their vector embeddings.
create table public.conversation_history (
  id bigserial primary key,
  session_id text not null,
  role text not null,
  text text not null,
  embedding vector(384), -- Matches 'all-MiniLM-L6-v2' embedding dimension
  created_at timestamptz default now()
);

-- 4. Create the RPC function for vector similarity search
--    This function is called by the application to find relevant
--    past messages based on the current user input.
create or replace function public.match_conversations (
  query_embedding vector(384),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  content text,
  role text,
  created_at timestamptz,
  similarity float
)
language sql stable
as $$
  select
    ch.id,
    ch.text as content,
    ch.role,
    ch.created_at,
    1 - (ch.embedding <=> query_embedding) as similarity
  from public.conversation_history as ch
  where 1 - (ch.embedding <=> query_embedding) > match_threshold
  order by similarity desc
  limit match_count;
$$;

-- 5. Create a vector index for performance
--    As your conversation history grows, this index will make the
--    similarity search much faster.
create index if not exists conversation_history_embedding_idx on public.conversation_history using ivfflat (embedding vector_l2_ops) with (lists = 100);

-- 6. Create User Profile Table for long-term memory
create table if not exists public.user_profile (
  id bigserial primary key,
  user_id text not null, -- e.g., 'sai_main_session' for now
  key text not null,
  value text not null,
  created_at timestamptz default now(),
  unique(user_id, key) -- ensures we don't have duplicate keys for a user
);

-- 7. Create RPC function to upsert user profile facts
-- This allows adding/updating facts in a single database call.
create or replace function public.upsert_user_profile(p_user_id text, p_key text, p_value text)
returns void as $$
begin
  insert into public.user_profile(user_id, key, value)
  values (p_user_id, p_key, p_value)
  on conflict (user_id, key) do update
  set value = p_value;
end;
$$ language plpgsql; 