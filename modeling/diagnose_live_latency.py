"""
DIAGNOSTIC ONLY -- measures real end-to-end latency through the ACTUAL
running production backend (uvicorn + FastAPI + the real /ws/mic
handler, unmodified inference/decision code), using the temporary
?diag=1 instrumentation added to app/main.py (see
outputs/reports/live_stream_latency.md for what that adds and why).

No physical microphone or speakers exist in this sandboxed environment,
so this cannot literally "speak into a mic." What it CAN do, and does,
is emulate exactly what the browser's ScriptProcessorNode does:
  - read a real recording (human or synthetic)
  - upsample it once to a realistic browser AudioContext rate (48kHz)
  - send it over the SAME WebSocket protocol, in the SAME 4096-sample
    chunks, paced at the SAME real wall-clock interval a live callback
    would use (4096/48000s per chunk) -- not sped up, not batched
  - prepend the same 8-byte capture-timestamp header the real browser
    diag build would send
  - receive the real JSON responses from the real backend and measure
    the real timestamps the real server recorded

This isolates transport/backend timing questions (capture->send,
network+queue, backend processing, response->receive) from anything
that requires actual room acoustics or hardware -- exactly the
distinction asked for (A-G bottleneck categories).
"""
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.audio_utils import TARGET_SR

AUDIO_DIR = Path(__file__).resolve().parents[1] / "files" / "audio"
REPORT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

WS_URL = "ws://127.0.0.1:8000/ws/mic?sr=48000&diag=1"
MIC_SR = 48000
CHUNK = 4096  # matches app/main.py: audioCtx.createScriptProcessor(4096, 1, 1)
CHUNK_SECONDS = CHUNK / MIC_SR

TEST_FILES = [
    ("call_0294f969f98b", "synthetic (AI voice)"),
    ("call_0e1e2f29bfdc", "human"),
]


def load_mic_emulated(anon_id, seconds=12.0):
    y, sr = sf.read(AUDIO_DIR / f"{anon_id}.wav", always_2d=True)
    y = y[:, 0].astype(np.float64)
    assert sr == TARGET_SR
    n = int(seconds * TARGET_SR)
    y = y[:n]
    return librosa.resample(y, orig_sr=TARGET_SR, target_sr=MIC_SR, res_type="soxr_hq")


def pack_chunk(capture_ts_ms: float, samples: np.ndarray) -> bytes:
    header = np.array([capture_ts_ms], dtype="<f8").tobytes()
    body = samples.astype(np.float32).tobytes()
    return header + body


async def run_one(anon_id, label, seconds=12.0):
    y_mic = load_mic_emulated(anon_id, seconds=seconds)
    n_chunks = len(y_mic) // CHUNK
    print(f"\n=== {anon_id} ({label}) -- {seconds:.0f}s emulated mic stream, "
          f"{n_chunks} chunks @ {CHUNK_SECONDS*1000:.1f}ms/chunk ===")

    samples_diag = []
    responses = []
    t_stream_start = time.time()
    t_first_result = None

    async with websockets.connect(WS_URL, max_size=None) as ws:

        async def receiver():
            nonlocal t_first_result
            try:
                async for raw in ws:
                    t_browser_receive = time.time() * 1000.0
                    d = json.loads(raw)
                    if d.get("type") == "error":
                        print("  [server error]", d.get("error"))
                        continue
                    responses.append(d)
                    if t_first_result is None and not d.get("silence"):
                        t_first_result = time.time() - t_stream_start
                    diag = d.get("_diag")
                    if diag:
                        samples_diag.append({
                            "capture_to_backend": diag["t_backend_receive"] - diag["capture_ts"],
                            "backend_processing": diag["t_response_send"] - diag["t_backend_receive"],
                            "backend_to_client": t_browser_receive - diag["t_response_send"],
                            "end_to_end": t_browser_receive - diag["capture_ts"],
                            "server_total_ms": d.get("total_ms"),
                        })
            except websockets.exceptions.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(receiver())

        # Pace sends at real wall-clock intervals, exactly like a live
        # ScriptProcessorNode callback firing every CHUNK_SECONDS.
        next_send = time.time()
        for i in range(n_chunks):
            chunk = y_mic[i * CHUNK:(i + 1) * CHUNK]
            capture_ts = time.time() * 1000.0
            await ws.send(pack_chunk(capture_ts, chunk))
            next_send += CHUNK_SECONDS
            sleep_for = next_send - time.time()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

        await asyncio.sleep(2.0)  # drain any trailing in-flight responses
        recv_task.cancel()

    n_results = len(responses)
    n_silence = sum(1 for r in responses if r.get("silence"))
    scores = [r["raw"] for r in responses if r.get("raw") is not None]
    labels_final = responses[-1]["label"] if responses else None

    def pct(key, p):
        vals = sorted(s[key] for s in samples_diag)
        if not vals:
            return None
        idx = min(int(len(vals) * p), len(vals) - 1)
        return vals[idx]

    print(f"  chunks sent: {n_chunks}  results received: {n_results}  (silence: {n_silence})")
    print(f"  time to first non-silent result: {t_first_result:.2f}s" if t_first_result else "  no non-silent result received")
    print(f"  scores seen: n={len(scores)} mean={np.mean(scores):.3f} max={np.max(scores):.3f}" if scores else "  no scores")
    print(f"  final label: {labels_final}")
    for key in ["capture_to_backend", "backend_processing", "backend_to_client", "end_to_end"]:
        med = pct(key, 0.5)
        p95 = pct(key, 0.95)
        if med is not None:
            print(f"  {key:22s}: median={med:7.2f}ms  p95={p95:7.2f}ms")

    return {
        "anon_id": anon_id, "label": label,
        "n_chunks": n_chunks, "n_results": n_results, "n_silence": n_silence,
        "time_to_first_result_s": t_first_result,
        "scores": scores, "final_label": labels_final,
        "latency_samples": samples_diag,
        "timeline": [{"raw": r.get("raw"), "smoothed": r.get("smoothed"),
                      "label": r.get("label"), "silence": r.get("silence")}
                     for r in responses],
    }


async def main():
    all_results = {}
    for anon_id, label in TEST_FILES:
        r = await run_one(anon_id, label, seconds=22.0)
        all_results[anon_id] = r

    with open(REPORT_DIR / "_live_latency_dump.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDumped full results to {REPORT_DIR / '_live_latency_dump.json'}")


if __name__ == "__main__":
    asyncio.run(main())
