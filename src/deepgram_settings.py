import os
from typing import Any, Dict, Optional


def build_settings(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    e = env or os.environ
    language = e.get("DG_AGENT_LANGUAGE", "en")
    stt_model = e.get("DG_AGENT_STT_MODEL", e.get("DEEPGRAM_STT_MODEL", "nova-3"))
    tts_model = e.get("DG_AGENT_TTS_MODEL", "aura-2-thalia-en")
    llm_provider = e.get("DG_AGENT_LLM_PROVIDER", "open_ai")
    llm_model = e.get("DG_AGENT_LLM_MODEL", "gpt-4o-mini")
    temperature = float(e.get("DG_AGENT_TEMPERATURE", "0.7"))
    greeting = e.get("DG_AGENT_GREETING", "Hello! How can I help you today?")

    return {
        "type": "Settings",
        "audio": {
            "input": {"encoding": "linear16", "sample_rate": 24000},
            "output": {"encoding": "linear16", "sample_rate": 24000, "container": "wav"},
        },
        "agent": {
            "language": language,
            "listen": {
                "provider": {
                    "type": "deepgram",
                    "model": stt_model,
                    "smart_format": True,
                }
            },
            "think": {
                "provider": {
                    "type": llm_provider,
                    "model": llm_model,
                    "temperature": temperature,
                }
            },
            "speak": {"provider": {"type": "deepgram", "model": tts_model}},
            "greeting": greeting,
        },
    }


