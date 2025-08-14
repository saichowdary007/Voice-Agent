---
inclusion: always
---

# Voice Agent Development Guidelines

## Architecture Patterns

### Backend (FastAPI + Python)
- **Async-first**: All I/O operations use `async/await` patterns
- **Dependency injection**: Services injected via FastAPI dependencies
- **Provider pattern**: LLM providers abstracted behind common interface in `src/llm_providers.py`
- **WebSocket proxy**: Direct proxy to Deepgram Voice Agent API for real-time voice
- **Configuration management**: Centralized in `src/config.py` with environment variables

### Frontend (React + TypeScript)
- **Component composition**: Small, focused components with single responsibility
- **Custom hooks**: Business logic extracted to reusable hooks (e.g., `useWebSocket.ts`)
- **Context API**: Global state management for auth and WebSocket connections
- **Strong typing**: TypeScript throughout with shared types in `src/types/`
- **Real-time communication**: WebSocket connections for voice agent interaction

## Code Style & Conventions

### Python Backend
- **Naming**: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- **Imports**: Standard library → Third-party → Local imports (with blank lines between groups)
- **Error handling**: Use FastAPI's HTTPException for API errors, proper async exception handling
- **Type hints**: Required for all function signatures and class attributes

### TypeScript Frontend
- **Naming**: `PascalCase.tsx` for components, `camelCase.ts` for hooks/services, `camelCase` for interfaces
- **Components**: Functional components with TypeScript interfaces for props
- **State management**: useState/useEffect for local state, Context for global state
- **Event handling**: Proper cleanup in useEffect hooks, especially for WebSocket connections

## Voice Agent Specific Rules

### Real-time Performance
- **Target latency**: ~500ms for voice-to-voice interactions
- **WebSocket handling**: Maintain persistent connections, handle reconnection gracefully
- **Audio processing**: Use Deepgram's browser agent for client-side audio handling
- **Error recovery**: Graceful degradation when voice services are unavailable

### Deepgram Integration
- **Voice Agent API**: Primary interface for STT/TTS/VAD functionality
- **Configuration**: Settings in `src/deepgram_settings.py` and `src/agent/` modules
- **Event handling**: Proper routing of Deepgram events through `agent_event_router.py`
- **Connection management**: Use `deepgram_agent_manager.py` for lifecycle management

### LLM Provider Support
- **Multi-provider**: Support OpenAI, Anthropic, Google Gemini, and Groq
- **Provider abstraction**: Common interface in `src/llm.py`, implementations in `src/llm_providers.py`
- **Configuration**: Provider selection via environment variables
- **Error handling**: Fallback strategies when primary provider fails

## Security & Authentication

### Supabase Integration
- **JWT tokens**: Use Supabase Auth for user authentication
- **Row-level security**: Implement RLS policies for user data isolation
- **Environment variables**: Never commit API keys, use `.env` files
- **Optional fallback**: Support local PostgreSQL when Supabase unavailable

### WebSocket Security
- **Authentication**: Verify JWT tokens on WebSocket connection
- **Rate limiting**: Implement appropriate limits for voice interactions
- **Input validation**: Sanitize all user inputs before processing

## Development Workflow

### Environment Setup
- **Docker first**: Use `docker-compose.yml` for consistent development environment
- **Environment files**: Copy `env-template.txt` to `.env` and configure required keys
- **Minimum requirements**: `DEEPGRAM_API_KEY` and at least one LLM provider key

### Testing Strategy
- **Unit tests**: Focus on business logic and provider integrations
- **Integration tests**: Test WebSocket connections and voice agent flows
- **Manual testing**: Use provided test scripts (`test_*.py`) for voice functionality
- **Error scenarios**: Test network failures, API timeouts, and service unavailability

### File Organization
- **Backend**: Keep related functionality in focused modules under `src/`
- **Frontend**: Group by feature (components, hooks, services, types)
- **Configuration**: Centralize environment-dependent settings
- **Tests**: Co-locate test files with source code when possible

## Performance Considerations

### Backend Optimization
- **Async operations**: Never block the event loop with synchronous I/O
- **Connection pooling**: Reuse database and HTTP connections
- **Memory management**: Proper cleanup of WebSocket connections and audio streams
- **Caching**: Use Redis for session data and frequently accessed information

### Frontend Optimization
- **Bundle size**: Lazy load components and libraries when possible
- **Audio handling**: Efficient audio buffer management for real-time processing
- **Re-renders**: Optimize React components to prevent unnecessary re-renders
- **WebSocket efficiency**: Batch messages when appropriate, handle backpressure

## Error Handling Patterns

### Backend Errors
- **HTTP exceptions**: Use FastAPI's HTTPException with appropriate status codes
- **Async errors**: Proper exception handling in async contexts
- **Service failures**: Graceful degradation when external services fail
- **Logging**: Structured logging for debugging and monitoring

### Frontend Errors
- **Error boundaries**: Catch and handle React component errors
- **Network failures**: Retry logic for API calls and WebSocket reconnection
- **User feedback**: Clear error messages and recovery instructions
- **State consistency**: Ensure UI state remains consistent during errors