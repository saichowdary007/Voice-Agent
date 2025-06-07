import pytest
from backend import audio_stream_pb2


def test_audio_chunk_serialization():
    chunk = audio_stream_pb2.AudioChunk(data=b"hello")
    serialized = chunk.SerializeToString()
    parsed = audio_stream_pb2.AudioChunk()
    parsed.ParseFromString(serialized)
    assert parsed.data == b"hello"


def test_recognition_result_fields():
    result = audio_stream_pb2.RecognitionResult(text="hi", is_final=True)
    assert result.text == "hi"
    assert result.is_final is True
