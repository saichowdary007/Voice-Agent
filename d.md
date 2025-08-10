Deepgram Voice Agent – End‑to‑End Integration

This guide shows how to wire Deepgram’s Voice Agent API v1 into a real-time voice app—browser mic → WebSocket → Deepgram → (optional) function calls → streamed TTS back to the user—and how to adapt the same agent for telephony (Twilio). All examples are production‑oriented and easy to copy‑paste.

TL;DR (happy path): connect to wss://agent.deepgram.com/v1/agent/converse, authenticate, immediately send a Settings message, then stream mic audio up and play TTS bytes down. Handle server events like Welcome, SettingsApplied, ConversationText, UserStartedSpeaking, AgentAudioDone, and (if enabled) FunctionCallRequest → reply with FunctionCallResponse.

⸻

0) Prereqs & choices
	•	A Deepgram API key (Project → API Keys). Store it server‑side.
	•	Runtime: Node 18+ (for server/proxy & function calling) and any modern browser (for mic & playback).
	•	Optional: a PSTN number (Twilio) for phone calls.
	•	Optional: your own LLM or a managed one (OpenAI, Anthropic, Google, Groq, etc.).

Audio defaults that “just work”
	•	Browser: input linear16 @ 24 kHz mono; output linear16 @ 24 kHz.
	•	Twilio: input/output mulaw @ 8 kHz mono with container none.

You can change encodings/sample‑rates—just keep input/output settings consistent with your capture and playback chain.

⸻

1) Architecture at a glance

[Browser Mic] ⇄ WebSocket ⇄ [Your Node Server] ⇄ WebSocket ⇄ [Deepgram Voice Agent]
                             ▲                                │
                             └── HTTP (function calls) ◀──────┘

(Telephony)  Twilio <—WSS stream—> [Your Node Server] <—WSS—> Deepgram

Why a small Node server?
	•	Hide your API key and mint short‑lived auth.
	•	Normalize audio and add WAV headers for browser playback when needed.
	•	Terminate Twilio call streams.

⸻

2) Connect & authenticate

Deepgram Voice Agent WebSocket endpoint:

wss://agent.deepgram.com/v1/agent/converse

Auth options
	•	HTTP header: Authorization: Token <YOUR_DEEPGRAM_API_KEY> (preferred from your server)
	•	WebSocket subprotocol (when headers are awkward): subprotocols: ["token", DEEPGRAM_API_KEY]

Don’t ship your API key to the browser. Use a server to connect on behalf of the client or to issue a short‑lived token.

⸻

3) The very first message: Settings

Send Settings immediately after the WebSocket opens—before any audio. It defines audio IO, the LLM (“think”), STT (“listen”), and TTS (“speak”), plus optional greeting and context.

Minimal Settings (browser mic)

{
  "type": "Settings",
  "audio": {
    "input":  { "encoding": "linear16", "sample_rate": 24000 },
    "output": { "encoding": "linear16", "sample_rate": 24000, "container": "none" }
  },
  "agent": {
    "language": "en",
    "listen": { "provider": { "type": "deepgram", "model": "nova-3" } },
    "think":  { "provider": { "type": "open_ai", "model": "gpt-4o-mini", "temperature": 0.7 } },
    "speak":  { "provider": { "type": "deepgram", "model": "aura-2-thalia-en" } },
    "greeting": "Hi! How can I help today?"
  }
}

Minimal Settings (Twilio)

{
  "type": "Settings",
  "audio": {
    "input":  { "encoding": "mulaw", "sample_rate": 8000 },
    "output": { "encoding": "mulaw", "sample_rate": 8000, "container": "none" }
  },
  "agent": {
    "language": "en",
    "listen": { "provider": { "type": "deepgram", "model": "nova-3" } },
    "think":  { "provider": { "type": "open_ai", "model": "gpt-4o-mini" } },
    "speak":  { "provider": { "type": "deepgram", "model": "aura-2-thalia-en" } },
    "greeting": "Hello! How can I help you today?"
  }
}

Useful toggles
	•	flags.history: true|false (include/exclude function call & chat history)
	•	agent.context.messages: […] (seed prior conversation or function results)
	•	agent.listen.provider.smart_format: true (prettier transcripts for UIs)
	•	agent.think.context_length: 15000 | "max" (only when using a custom think endpoint)

⸻

4) Client → server messages you’ll use
	•	Settings: required first message (see above)
	•	UpdatePrompt: change system prompt mid‑call

{ "type": "UpdatePrompt", "prompt": "You are a concise, friendly assistant." }


	•	UpdateSpeak: live‑switch TTS voice

{ "type": "UpdateSpeak", "model": "aura-2-zeus-en" }


	•	InjectUserMessage: pretend the user said some text (testing/chat UI)

{ "type": "InjectUserMessage", "content": "Please reset my password." }


	•	InjectAgentMessage: force the agent to say something right now

{ "type": "InjectAgentMessage", "message": "One moment while I check that." }


	•	FunctionCallResponse: send back function results when client_side = true

{ "type": "FunctionCallResponse", "id": "fc_123", "name": "get_weather", "content": "{\"temp_c\": 21}" }


	•	KeepAlive: only if you’re silent for > ~8s and not streaming mic audio

{ "type": "KeepAlive" }



⸻

5) Server → client events to handle
	•	Welcome → contains request_id (good for logging/trace)
	•	SettingsApplied → only start audio after you see this
	•	ConversationText → { role: "user"|"assistant", content } for your transcript UI
	•	UserStartedSpeaking → stop/clear any TTS playback (barge‑in)
	•	AgentThinking → show a subtle “thinking…” indicator
	•	AgentAudioDone → last audio frame for the current reply was sent
	•	AgentErrors / AgentWarnings → log and display
	•	FunctionCallRequest → run the function(s) if client_side: true, then reply with FunctionCallResponse

⸻

6) Browser mic → Deepgram (TypeScript)

Streams Float32 from WebAudio → 16‑bit PCM frames (linear16) to the server; plays TTS by prepending a WAV header to each chunk so browsers accept it.

// client/app.ts
const WS_URL = "/ws"; // your Node proxy (don’t call Deepgram from the browser)

let ws: WebSocket;
let ctx: AudioContext;
let processor: ScriptProcessorNode;

async function start() {
  ctx = new AudioContext({ sampleRate: 24000 });
  const media = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: 24000,
      echoCancellation: true,
      noiseSuppression: false,
    },
  });

  const source = ctx.createMediaStreamSource(media);
  processor = ctx.createScriptProcessor(2048, 1, 1);
  source.connect(processor);
  processor.connect(ctx.destination); // required in some browsers

  ws = new WebSocket(WS_URL);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    // Send Settings immediately
    ws.send(
      JSON.stringify({
        type: "Settings",
        audio: {
          input: { encoding: "linear16", sample_rate: 24000 },
          output: { encoding: "linear16", sample_rate: 24000, container: "none" },
        },
        agent: {
          language: "en",
          listen: { provider: { type: "deepgram", model: "nova-3" } },
          think: { provider: { type: "open_ai", model: "gpt-4o-mini" } },
          speak: { provider: { type: "deepgram", model: "aura-2-thalia-en" } },
          greeting: "Hi! How can I help?",
        },
      })
    );
  };

  // Capture mic frames and send as 16‑bit PCM
  processor.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    const pcm = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    ws.send(pcm.buffer);
  };

  // Play TTS by adding a WAV header to each raw PCM chunk
  ws.onmessage = async (ev) => {
    if (typeof ev.data === "string") {
      const msg = JSON.parse(ev.data);
      if (msg.type === "ConversationText") {
        addTranscript(msg.role, msg.content);
      }
      if (msg.type === "UserStartedSpeaking") {
        // stop/flush any queued audio
        flushPlayback();
      }
      return;
    }

    // Binary audio, container: none → prepend minimal WAV header
    const audioBytes: ArrayBuffer = ev.data;
    const wav = withWavHeader(audioBytes, 24000, 1, 16);
    await playWav(wav);
  };
}

function withWavHeader(pcmBytes: ArrayBuffer, sampleRate: number, channels: number, bitsPerSample: number) {
  const byteRate = (sampleRate * channels * bitsPerSample) / 8;
  const blockAlign = (channels * bitsPerSample) / 8;
  const dataSize = pcmBytes.byteLength;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // RIFF/WAVE header
  writeStr(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeStr(view, 8, "WAVE");
  writeStr(view, 12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true);  // format = PCM
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);
  writeStr(view, 36, "data");
  view.setUint32(40, dataSize, true);

  new Uint8Array(buffer, 44).set(new Uint8Array(pcmBytes));
  return buffer;
}

function writeStr(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

let audioQueue: ArrayBuffer[] = [];
function flushPlayback() { audioQueue = []; }

async function playWav(wav: ArrayBuffer) {
  audioQueue.push(wav);
  if (audioQueue.length > 4) audioQueue.shift();
  const blob = new Blob([wav], { type: "audio/wav" });
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  await audio.play();
}

function addTranscript(role: string, text: string) {
  const el = document.querySelector("#transcript")!;
  const row = document.createElement("div");
  row.textContent = `${role}: ${text}`;
  el.appendChild(row);
}

start().catch(console.error);

Notes
	•	We prepend a minimal WAV header per chunk so Chrome will play raw PCM. If you choose an encoded output (e.g., mp3), you don’t need this header, but latency may increase.
	•	Echo cancellation is on to reduce agent self‑triggering; you can also implement VAD to gate upstream audio.

⸻

7) Node server (proxy + function calling)

Proxies client WS ↔ Deepgram, injects auth header, forwards binary audio both ways, handles events, and executes client‑side function calls when requested.

// server/index.ts
import http from "http";
import express from "express";
import { WebSocketServer, WebSocket } from "ws";
import fetch from "node-fetch";

const PORT = process.env.PORT || 3000;
const DG_KEY = process.env.DEEPGRAM_API_KEY!;

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: "/ws" });

wss.on("connection", (client) => {
  // Connect upstream to Deepgram with Authorization header
  const upstream = new WebSocket("wss://agent.deepgram.com/v1/agent/converse", {
    headers: { Authorization: `Token ${DG_KEY}` },
  });

  // Pipe messages up
  client.on("message", (data, isBinary) => {
    if (upstream.readyState === WebSocket.OPEN) upstream.send(data, { binary: isBinary });
  });

  // Pipe messages down
  upstream.on("message", (data, isBinary) => {
    // Optional: inspect JSON events for FunctionCallRequest and handle client_side true
    if (!isBinary) {
      const text = data.toString();
      try {
        const msg = JSON.parse(text);
        if (msg.type === "FunctionCallRequest" && Array.isArray(msg.functions)) {
          for (const f of msg.functions) {
            if (f.client_side) {
              handleFunction(f)
                .then((content) => {
                  upstream.send(
                    JSON.stringify({
                      type: "FunctionCallResponse",
                      id: f.id,
                      name: f.name,
                      content,
                    })
                  );
                })
                .catch((e) => console.error("Function error", e));
            }
          }
        }
      } catch {}
    }
    if (client.readyState === WebSocket.OPEN) client.send(data, { binary: isBinary });
  });

  const closeBoth = () => {
    try { upstream.close(); } catch {}
    try { client.close(); } catch {}
  };
  client.on("close", closeBoth);
  upstream.on("close", closeBoth);
  client.on("error", closeBoth);
  upstream.on("error", closeBoth);
});

server.listen(PORT, () => console.log(`Server listening on :${PORT}`));

// Example function router
async function handleFunction(f: { name: string; arguments: string }) {
  switch (f.name) {
    case "get_weather": {
      const { location } = JSON.parse(f.arguments || "{}");
      // Replace with your data source
      const result = { location, temperature_c: 21, condition: "Sunny" };
      return JSON.stringify(result);
    }
    default:
      return JSON.stringify({ error: `Unknown function ${f.name}` });
  }
}

Tips
	•	Only allow your own origin(s) to hit /ws.
	•	If you prefer a BYO LLM endpoint, set agent.think.endpoint.url and headers in the Settings message; model IDs are flexible for BYO.

⸻

8) Defining functions for the LLM to call (in Settings)

You make functions discoverable to the LLM by adding JSON‑Schema‑like definitions under agent.think.functions. The agent will request calls via FunctionCallRequest with client_side set based on how you configured an endpoint.

{
  "type": "Settings",
  "agent": {
    "think": {
      "provider": { "type": "open_ai", "model": "gpt-4o-mini" },
      "prompt": "You are a helpful support agent.",
      "functions": [
        {
          "name": "get_weather",
          "description": "Get current weather by location",
          "parameters": {
            "type": "object",
            "properties": { "location": { "type": "string" } },
            "required": ["location"]
          },
          // If you include an endpoint, the server will call it (client_side=false)
          // Omit endpoint to handle it in your app (client_side=true)
          // "endpoint": { "url": "https://api.example.com/weather", "method": "post", "headers": { "authorization": "Bearer {{token}}" } }
        }
      ]
    }
  }
}

Flow
	1.	Server sends FunctionCallRequest (with functions: [{ id, name, arguments, client_side }]).
	2.	If client_side: true, your app runs it and replies FunctionCallResponse with a stringified result (often JSON).
	3.	The agent continues the conversation using the result.

⸻

9) Twilio telephony integration (8 kHz µ‑law)

TwiML Bin (replace the url with your WSS server’s /twilio endpoint):

<Response>
  <Say language="en">This call may be monitored or recorded.</Say>
  <Connect>
    <Stream url="wss://YOUR-NGROK-OR-DOMAIN/twilio" />
  </Connect>
</Response>

Server sketch (key parts):

# twilio_server.py (Python example)
import asyncio, base64, json, os, websockets

DG_KEY = os.environ["DEEPGRAM_API_KEY"]

async def sts_connect():
    return await websockets.connect(
        "wss://agent.deepgram.com/v1/agent/converse",
        subprotocols=["token", DG_KEY],
    )

async def twilio_handler(twilio_ws):
    audio_q = asyncio.Queue()
    streamsid_q = asyncio.Queue()

    async with sts_connect() as dg:
        settings = {
            "type": "Settings",
            "audio": {
                "input": {"encoding": "mulaw", "sample_rate": 8000},
                "output": {"encoding": "mulaw", "sample_rate": 8000, "container": "none"},
            },
            "agent": {
                "language": "en",
                "listen": {"provider": {"type": "deepgram", "model": "nova-3"}},
                "think":  {"provider": {"type": "open_ai", "model": "gpt-4o-mini"}},
                "speak":  {"provider": {"type": "deepgram", "model": "aura-2-thalia-en"}},
                "greeting": "Hello! How can I help you today?",
            },
        }
        await dg.send(json.dumps(settings))

        async def to_deepgram():
            while True:
                chunk = await audio_q.get()
                await dg.send(chunk)

        async def from_deepgram():
            sid = await streamsid_q.get()
            async for message in dg:
                if isinstance(message, str):
                    evt = json.loads(message)
                    if evt.get("type") == "UserStartedSpeaking":
                        await twilio_ws.send(json.dumps({"event":"clear", "streamSid": sid}))
                    continue
                payload = base64.b64encode(message).decode("ascii")
                await twilio_ws.send(json.dumps({
                    "event": "media",
                    "streamSid": sid,
                    "media": {"payload": payload},
                }))

        async def from_twilio():
            buf = bytearray()
            async for raw in twilio_ws:
                data = json.loads(raw)
                if data.get("event") == "start":
                    streamSid = data["start"]["streamSid"]
                    streamsid_q.put_nowait(streamSid)
                elif data.get("event") == "media":
                    # Twilio sends 160 bytes / 20ms; batch a few frames
                    b = base64.b64decode(data["media"]["payload"])  # 8k mu‑law
                    buf += b
                    if len(buf) >= 160 * 20:  # ~0.4s
                        await audio_q.put(bytes(buf))
                        buf.clear()

        await asyncio.gather(to_deepgram(), from_deepgram(), from_twilio())

Gotchas
	•	Twilio streams 20 ms frames (160 bytes each at 8 kHz µ‑law). Batching reduces WS overhead.
	•	Always clear buffered TTS when you see UserStartedSpeaking (barge‑in).

⸻

10) Choosing TTS providers & voices
	•	Deepgram Aura: set agent.speak.provider.type = "deepgram" and choose a model (e.g., aura-2-thalia-en).
	•	OpenAI: type = "open_ai", provide model, voice, and an endpoint with auth headers.
	•	ElevenLabs, Cartesia, AWS Polly: set provider‑specific fields; for Polly also include credentials and (often) an explicit engine.
	•	You can pass multiple speak providers (an array) to define fallbacks.

⸻

11) Bring your own LLM (BYO)

If your endpoint speaks an OpenAI‑compatible (chat‑completions) protocol, set:

"think": {
  "provider": { "type": "open_ai", "model": "your-model-id" },
  "endpoint": { "url": "https://your.domain/v1/chat/completions", "headers": { "authorization": "Bearer {{token}}" } },
  "prompt": "You are…"
}

For Anthropic/Google/Groq use the corresponding provider.type and include endpoint when required. context_length is adjustable only when using a custom think.endpoint.

⸻

12) Security & production checklist
	•	Never expose your Deepgram API key to the browser.
	•	Prefer short‑lived tokens from a secure backend; proxy the WS.
	•	Throttle/meter: enforce per‑user limits on concurrent sessions.
	•	Validate FunctionCallRequest names/arguments before executing.
	•	Log request_id (from Welcome) and all AgentErrors/Warnings.
	•	Graceful shutdown: close both sockets on either side closing.

⸻

13) Troubleshooting
	•	“Static” or screeching playback → output encoding/sample‑rate mismatch. Ensure your Settings match what you play. For browsers, if using raw PCM, prepend a WAV header to each chunk (see code above).
	•	No audio in Chrome → you’re feeding raw PCM without a container. Add the WAV header helper or choose an encoded output (e.g., MP3/Opus) if latency permits.
	•	Agent talks to itself → enable echo cancellation, or gate mic with VAD; clear playback on UserStartedSpeaking.
	•	Silence / disconnects → if you’re not streaming any mic audio for > ~8s, send {"type":"KeepAlive"} periodically.
	•	Prompt/voice changes not sticking → wait for PromptUpdated / SpeakUpdated before assuming changes are active.
	•	403/401 → missing/incorrect Authorization header (server side) or wrong subprotocol usage.

⸻

14) Quick test plan
	1.	Start your Node server (DEEPGRAM_API_KEY=... pnpm tsx server/index.ts).
	2.	Open the browser app and watch for Welcome then SettingsApplied.
	3.	Speak; confirm ConversationText (user) then an assistant reply appears and audio plays.
	4.	Talk over the agent; verify UserStartedSpeaking stops playback immediately.
	5.	Trigger a function (say “What’s the weather in Paris?”) and verify a FunctionCallRequest arrives and your FunctionCallResponse unblocks the conversation.

⸻

15) Appendix – Message cheat sheet

Client → Server
	•	Settings
	•	UpdatePrompt
	•	UpdateSpeak
	•	InjectUserMessage
	•	InjectAgentMessage
	•	FunctionCallResponse
	•	KeepAlive

Server → Client
	•	Welcome
	•	SettingsApplied
	•	ConversationText
	•	UserStartedSpeaking
	•	AgentThinking
	•	FunctionCallRequest
	•	PromptUpdated
	•	SpeakUpdated
	•	AgentAudioDone
	•	AgentErrors
	•	AgentWarnings

⸻

Happy shipping! If you want this split into a ready‑to‑run repo (/client, /server, /twilio), ping me and I’ll scaffold it with scripts and Docker.