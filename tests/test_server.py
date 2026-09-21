"""
Server tests. They use a fake Whisper model, so no models or GPU are needed:
    pip install -r requirements-dev.txt
    python -m pytest tests -q
"""
import io
import json
import math
import os
import struct
import sys
import threading
import types
import wave
import http.client

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import server_faster_whisper as srv  # noqa: E402

ORIGIN = "http://localhost:3000"


def make_wav(seconds=6, sr=16000):
    """Two alternating tones so the clip has some structure."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(int(seconds * sr)):
            freq = 120 if (i // sr) % 2 == 0 else 220
            frames += struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / sr)))
        wf.writeframes(bytes(frames))
    return buf.getvalue()


class FakeModel:
    calls = []

    def transcribe(self, path, **kwargs):
        FakeModel.calls.append(kwargs)
        seg = lambda s, e, t: types.SimpleNamespace(
            start=s, end=e, text=f" {t} ", avg_logprob=-0.2,
            words=[types.SimpleNamespace(word=t, start=s, end=e, probability=0.9)],
        )
        segments = iter([seg(0.0, 2.0, "hello"), seg(2.0, 4.0, "world"), seg(4.0, 4.3, "ok")])
        info = types.SimpleNamespace(language="en", language_probability=0.99, duration=6.0)
        return segments, info


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "TRANSCRIPTS_DIR", str(tmp_path / "transcripts"))
    monkeypatch.setattr(srv, "get_whisper_model", lambda *a, **k: FakeModel())
    monkeypatch.setattr(srv, "resolve_whisper_model", lambda name: name if name != "nope" else (_ for _ in ()).throw(srv.ModelNotFound("Whisper model 'nope' is not in the local Hugging Face cache")))
    monkeypatch.setattr(srv, "diarizer_type", "acoustic-multifeature")
    FakeModel.calls.clear()
    httpd = srv.make_server(0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], tmp_path
    httpd.shutdown()
    httpd.server_close()


def request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    data = resp.read()
    return resp, data


def test_health_is_well_formed_and_leaks_no_paths(server):
    port, _ = server
    resp, data = request(port, "GET", "/health", headers={"Origin": ORIGIN})
    assert resp.status == 200                      # status line parsed => headers are in the right order
    assert resp.getheader("Access-Control-Allow-Origin") == ORIGIN
    payload = json.loads(data)
    assert payload["status"] == "online"
    text = data.decode()
    assert "admin_fdr" not in text and "hf_cache_dir" not in payload and "model_path" not in payload


def test_foreign_origin_is_rejected(server):
    port, _ = server
    resp, _ = request(port, "GET", "/health", headers={"Origin": "http://evil.example"})
    assert resp.status == 403
    assert resp.getheader("Access-Control-Allow-Origin") is None
    resp, _ = request(port, "POST", "/transcribe", body=make_wav(), headers={"Origin": "http://evil.example"})
    assert resp.status == 403


def test_dns_rebinding_host_is_rejected(server):
    port, _ = server
    resp, _ = request(port, "GET", "/health", headers={"Host": "evil.example:8000"})
    assert resp.status == 403


def test_transcribe_returns_segments_speakers_and_real_confidence(server):
    port, tmp_path = server
    resp, data = request(
        port, "POST", "/transcribe?model=small&language=ru&filename=a%20b.mp4",
        body=make_wav(), headers={"Origin": ORIGIN, "Content-Type": "audio/wav"},
    )
    assert resp.status == 200, data
    payload = json.loads(data)
    assert payload["total_segments"] == 3
    assert [s["text"] for s in payload["segments"]] == ["hello", "world", "ok"]
    assert all(s["speakerName"].startswith("Speaker 00") for s in payload["segments"])
    assert all("Male" not in s["speakerName"] and "Female" not in s["speakerName"] for s in payload["segments"])
    assert payload["segments"][0]["confidence"] == pytest.approx(math.exp(-0.2), abs=1e-3)
    assert all("confidence" not in spk for spk in payload["speakers"].values())
    assert FakeModel.calls[0]["language"] == "ru"          # language is now honoured
    assert (tmp_path / "transcripts" / "transcript_a b.md").exists()


def test_auto_language_means_none(server):
    port, _ = server
    request(port, "POST", "/transcribe?language=auto", body=make_wav(), headers={"Content-Type": "audio/wav"})
    assert FakeModel.calls[0]["language"] is None


def test_unknown_model_returns_404_json(server):
    port, _ = server
    resp, data = request(port, "POST", "/transcribe?model=nope", body=make_wav())
    assert resp.status == 404
    assert "not in the local" in json.loads(data)["error"]


def test_upload_limit(server, monkeypatch):
    port, _ = server
    monkeypatch.setattr(srv, "MAX_UPLOAD_BYTES", 1000)
    resp, _ = request(port, "POST", "/transcribe", body=make_wav())
    assert resp.status == 413


def test_empty_body(server):
    port, _ = server
    resp, _ = request(port, "POST", "/transcribe", body=b"")
    assert resp.status == 400


def test_save_markdown_cannot_escape_transcripts_dir(server):
    port, tmp_path = server
    body = json.dumps({"filename": "../../evil", "content": "x"})
    resp, data = request(port, "POST", "/api/save-markdown", body=body,
                         headers={"Origin": ORIGIN, "Content-Type": "application/json"})
    assert resp.status == 200
    saved = json.loads(data)["fullPath"]
    assert os.path.dirname(saved) == os.path.realpath(str(tmp_path / "transcripts"))
    assert not (tmp_path / "evil.md").exists()


def test_diarization_short_segments_inherit_neighbour():
    import numpy as np
    sr = 16000
    samples = np.zeros(sr * 6, dtype=np.float32)
    segs = [{"start": 0.0, "end": 2.0, "text": "a", "words": []},
            {"start": 2.0, "end": 2.2, "text": "b", "words": []}]     # too short to embed
    out, meta = srv.perform_speaker_diarization(samples, sr, segs)
    assert out[0]["speakerId"] == out[1]["speakerId"]
    assert list(meta) == ["speaker_001"]


# --- speaker clustering -------------------------------------------------------

def _synthetic_embeddings(n_speakers, per_speaker, noise, dim=192, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_speakers, dim))
    X, truth = [], []
    for s in range(n_speakers):
        for _ in range(per_speaker):
            v = centers[s] + rng.normal(scale=noise, size=dim)
            X.append(v / np.linalg.norm(v))
            truth.append(s)
    return X, truth


def _purity(labels, truth):
    from collections import Counter
    total = 0
    for lab in set(labels):
        members = [t for l, t in zip(labels, truth) if l == lab]
        total += Counter(members).most_common(1)[0][1]
    return total / len(labels)


@pytest.mark.parametrize("true_k", [2, 3])
def test_cluster_recovers_speaker_count_from_noisy_embeddings(true_k):
    X, truth = _synthetic_embeddings(true_k, 80, noise=1.0)
    labels = srv.cluster_speakers(X, [2.0] * len(X), max_speakers=3)
    assert len(set(labels)) == true_k
    assert _purity(labels, truth) > 0.9


def test_cluster_single_speaker_stays_single():
    X, _ = _synthetic_embeddings(1, 150, noise=1.0)
    assert set(srv.cluster_speakers(X, [2.0] * len(X), max_speakers=3)) == {0}


def test_cluster_never_exceeds_max_speakers():
    # 12 real speakers (the old threshold-based code produced dozens); the cap must hold.
    X, _ = _synthetic_embeddings(12, 40, noise=0.6)
    for cap in (1, 2, 3, 5):
        labels = srv.cluster_speakers(X, [1.5] * len(X), max_speakers=cap)
        assert len(set(labels)) <= cap


def test_cluster_exact_mode_and_outlier_merge():
    X, _ = _synthetic_embeddings(2, 60, noise=1.0)
    assert len(set(srv.cluster_speakers(X, [2.0] * len(X), max_speakers=2, exact=True))) == 2

    # Two real speakers plus 3 short outliers that must not become a third speaker.
    X, _ = _synthetic_embeddings(3, 60, noise=1.0, seed=1)
    X = X[:120] + X[120:123]                     # 2 big clusters + 3 stray points
    dur = [3.0] * 120 + [1.0] * 3
    labels = srv.cluster_speakers(X, dur, max_speakers=3)
    assert len(set(labels)) == 2


def test_long_recording_with_many_segments_is_capped(server, monkeypatch):
    """End-to-end: hundreds of segments must produce <= max_speakers speakers."""
    import numpy as np
    port, _ = server
    rng = np.random.default_rng(3)

    class ManySegs(FakeModel):
        def transcribe(self, path, **kw):
            mk = lambda i: types.SimpleNamespace(start=i * 0.5, end=i * 0.5 + 0.45, text=f"w{i}", avg_logprob=-0.3, words=[])
            return iter([mk(i) for i in range(11)]), types.SimpleNamespace(language="en", language_probability=1.0, duration=6.0)

    monkeypatch.setattr(srv, "get_whisper_model", lambda *a, **k: ManySegs())
    # random embeddings => "45 speakers" under the old threshold logic
    monkeypatch.setattr(srv, "embed_segment", lambda chunk, sr: (rng.normal(size=192) / 14.0, 120.0) if len(chunk) > 0 else (None, 0.0))
    resp, data = request(port, "POST", "/transcribe?max_speakers=3", body=make_wav(), headers={"Content-Type": "audio/wav"})
    assert resp.status == 200
    assert len(json.loads(data)["speakers"]) <= 3


def test_client_disconnect_does_not_raise(server, capsys):
    """A browser that hangs up mid-response must not produce a traceback / bogus 500."""
    port, _ = server
    import socket
    s = socket.create_connection(("127.0.0.1", port))
    body = make_wav()
    s.sendall(b"POST /transcribe HTTP/1.1\r\nHost: localhost\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
    s.close()                                     # hang up immediately
    import time; time.sleep(1.0)
    resp, _ = request(port, "GET", "/health")     # server still alive
    assert resp.status == 200
