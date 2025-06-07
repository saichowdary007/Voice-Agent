# Voice Agent Frontend

This Next.js application provides the web interface for the Voice Agent. It uses **Magic UI** components for an enhanced design.

## Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend expects the backend WebSocket server to be available at `ws://localhost:8003/ws` by default. You can change this using the `NEXT_PUBLIC_WS_URL` environment variable.
