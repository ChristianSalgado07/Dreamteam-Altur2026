"""
Live-demo server for the AI-voice-likelihood CNN prototype.

Microphone (the only demo path): captured IN THE BROWSER via
getUserMedia + Web Audio, streamed to /ws/mic as raw PCM frames over a
WebSocket, and scored by the SAME RollingDetector (preprocessing + model
+ smoothing/decision layer, unchanged from app/live_infer.py /
modeling/train_cnn.py) used everywhere else in this project -- no
separate inference implementation. The server never touches the OS
microphone and never requests mic access itself -- only a browser tab
does, and only after the user clicks "START DETECTION". Each WebSocket
connection gets its own fresh RollingDetector; closing the connection
(STOP button, tab close, error) ends it -- nothing lingers server-side.

(A file-upload/"DEMO MODE" fallback used to live here too; removed --
`python -m app.live_infer --file <path.wav>` and the modeling/diagnose_*
scripts already cover running a recording through this same detector.)

IMPORTANT (see outputs/reports/cnn_training_report.md): internal
validation ROC-AUC is ~1.00 but the existing external-validation set is
~0.53 (chance level). This is a live PROTOTYPE that analyzes acoustic
patterns in real time -- it is NOT a scientifically validated universal
AI-voice detector, and the UI says so.

Run with:
  uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/
"""
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from app.live_infer import RollingDetector, default_model_path
from app.model import load_model

app = FastAPI(title="AI Voice Likelihood - Live Demo")


# ---------------------------------------------------------------------
# Microphone mode -- browser-captured audio over a WebSocket. The server
# process itself never opens a microphone and nothing runs at startup;
# a connection only exists while a browser tab has explicitly started
# one (after the user clicks START DETECTION and grants permission).
# ---------------------------------------------------------------------
@app.websocket("/ws/mic")
async def ws_mic(websocket: WebSocket):
    await websocket.accept()

    model_path = default_model_path()
    if not Path(model_path).exists():
        await websocket.send_json({"type": "error", "error": f"Model not found at {model_path}."})
        await websocket.close()
        return

    try:
        sr = int(websocket.query_params.get("sr", "48000"))
    except ValueError:
        sr = 48000

    # TEMPORARY LATENCY DIAGNOSTIC (outputs/reports/live_stream_latency.md) --
    # opt-in via ?diag=1, does not affect the default wire format/behavior at
    # all. When on, each incoming binary message is expected to carry an
    # 8-byte little-endian float64 browser capture timestamp (epoch ms,
    # matching JS Date.now()) prepended to the raw float32 PCM samples, and
    # each outgoing result JSON gets an extra "_diag" block. Safe to delete
    # this whole diag branch (and the matching browser-side block) once the
    # latency investigation is done -- nothing else depends on it.
    diag = websocket.query_params.get("diag") == "1"

    model = load_model(model_path)
    detector = RollingDetector(model)  # fresh per connection -- nothing persists across STOP/START
    last_capture_ts = None  # most recent embedded browser capture timestamp seen (diag mode only)

    try:
        while True:
            data = await websocket.receive_bytes()
            t_backend_receive = time.time() * 1000.0
            if diag:
                last_capture_ts = np.frombuffer(data[:8], dtype="<f8")[0].item()
                samples = np.frombuffer(data[8:], dtype=np.float32)
            else:
                samples = np.frombuffer(data, dtype=np.float32)
            detector.push(samples, sr)
            if detector.ready():
                result = detector.step()
                payload = {
                    "type": "result",
                    "raw": result["raw"], "smoothed": result["smoothed"],
                    "label": result["label"], "silence": result["silence"],
                    "preprocess_ms": result["preprocess_ms"], "inference_ms": result["inference_ms"],
                    "total_ms": result["total_ms"], "error": result["error"],
                    "report": detector.get_report(),
                }
                if diag:
                    t_response_send = time.time() * 1000.0
                    payload["_diag"] = {
                        "capture_ts": last_capture_ts,
                        "t_backend_receive": t_backend_receive,
                        "t_response_send": t_response_send,
                    }
                await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass  # normal: browser called ws.close() from the STOP button, or the tab closed
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "error": f"{type(e).__name__}: {e}"})
        except Exception:
            pass
    # No cleanup needed server-side: nothing was opened on this process
    # (no thread, no OS mic handle) -- the browser owns and releases the
    # actual microphone (MediaStream) on its side.


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html><html><head><title>AI Voice Likelihood - Live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
/* Palette matches the Altur voice-deepfake-detector frontend (main branch:
   tailwind.config.js / src/index.css) -- ink/surface/line + Inter, red=synthetic,
   emerald=human, amber=uncertain (this app's 3-way verdict has no equivalent
   there, so amber -- already this app's own existing "Uncertain" color --
   was kept as the closest match rather than inventing a new hue). */
:root{
  --ink:#0a0e1a; --surface:#151b2e; --line:#1f2937; --line-soft:#1a2233;
  --text:#f9fafb; --muted:#94a3b8; --muted2:#64748b; --muted3:#475569;
  --blue:#60a5fa; --blue-strong:#3b82f6; --violet:#a78bfa; --violet-strong:#7c3aed;
  --ai:#ef4444; --human:#10b981; --uncertain:#f59e0b; --silence:#475569;
}
*{box-sizing:border-box}
body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;background:var(--ink);color:var(--text);
     display:flex;flex-direction:column;align-items:center;padding:0 0 3rem;margin:0;min-height:100vh}

/* ---- header (mirrors src/components/Header.tsx) ---- */
.topbar{width:100%;border-bottom:1px solid var(--line);background:#0d111e;position:sticky;top:0;z-index:5}
.topbar-inner{max-width:1000px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;
               padding:1rem 1.5rem;gap:1rem;flex-wrap:wrap}
.brand{display:flex;align-items:center;gap:0.75rem}
.brand-icon{width:2.5rem;height:2.5rem;border-radius:0.75rem;background:rgba(96,165,250,0.1);
            border:1px solid rgba(96,165,250,0.25);color:var(--blue);display:grid;place-items:center;flex-shrink:0}
.brand-title-row{display:flex;align-items:center;gap:0.5rem}
.brand-title{font-size:1.05rem;font-weight:800;color:#fff;letter-spacing:-0.01em}
.brand-badge{background:rgba(96,165,250,0.1);color:#93c5fd;font-size:0.6rem;font-weight:700;
             letter-spacing:0.18em;text-transform:uppercase;padding:0.15rem 0.4rem;border-radius:0.25rem}
.brand-sub{color:var(--muted2);font-size:0.75rem;margin-top:0.1rem}
.status-row{display:flex;align-items:center;gap:0.5rem;border:1px solid var(--line);background:rgba(2,6,23,0.4);
            border-radius:999px;padding:0.45rem 0.85rem;font-size:0.75rem;font-weight:500;color:var(--muted)}
.dot{width:0.55rem;height:0.55rem;border-radius:50%;background:var(--muted3);display:inline-block}
.dot.live{background:var(--human);box-shadow:0 0 10px var(--human);animation:pulse 1.4s infinite}
.dot.err{background:var(--ai);box-shadow:0 0 10px var(--ai)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
#micTestBanner{width:100%;text-align:center;font-size:0.75rem;font-weight:600;color:var(--blue);
               background:rgba(96,165,250,0.08);border-bottom:1px solid rgba(96,165,250,0.2);padding:0.5rem}

/* ---- shared layout ---- */
.page{width:100%;max-width:640px;padding:2rem 1.25rem 0;display:flex;flex-direction:column;align-items:center}
.eyebrow{color:var(--muted2);font-size:0.65rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase}
.card{width:100%;border:1px solid var(--line);background:rgba(21,27,46,0.8);border-radius:1rem;
      box-shadow:0 18px 50px rgba(0,0,0,0.18);padding:1.5rem;margin-bottom:1.25rem}
.card-head{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:1.1rem}
.card-title{font-size:1.05rem;font-weight:700;color:#fff;margin-top:0.15rem}

/* ---- controls ---- */
.controls{display:flex;gap:0.75rem;margin-bottom:0.5rem;justify-content:center}
#startBtn,#stopBtn{font-size:0.9rem;font-weight:700;padding:0.8rem 1.8rem;border-radius:0.75rem;border:none;
                    cursor:pointer;transition:all 0.2s;letter-spacing:0.01em}
#startBtn{background:linear-gradient(90deg,var(--blue-strong),var(--violet-strong));color:#fff;
          box-shadow:0 0 24px rgba(59,130,246,0.25)}
#startBtn:hover{box-shadow:0 0 32px rgba(59,130,246,0.45);transform:translateY(-1px)}
#stopBtn{background:rgba(239,68,68,0.1);color:#fca5a5;border:1px solid rgba(239,68,68,0.35)}
#stopBtn:hover{background:rgba(239,68,68,0.18)}

/* ---- gauge ---- */
.gauge-wrap{position:relative;margin:0 auto;max-width:320px}
.gauge-svg{width:100%;height:auto;overflow:visible}
.gauge-track{fill:none;stroke:var(--line);stroke-width:18;stroke-linecap:round}
.gauge-arc{fill:none;stroke-width:18;stroke-linecap:round;pathLength:100;
           transition:stroke-dasharray 0.4s ease,stroke 0.4s ease}
.gauge-tick{fill:var(--muted3);font-size:8px;text-anchor:middle}
#gaugeNeedle{stroke-width:2.5;stroke-linecap:round;transition:transform 0.5s cubic-bezier(.34,1.4,.64,1),stroke 0.4s ease}
#gaugeHub{transition:fill 0.4s ease}
.gauge-readout{position:absolute;left:0;right:0;bottom:0.1rem;text-align:center;pointer-events:none}
#score{font-size:2.75rem;font-weight:800;letter-spacing:-0.02em;font-variant-numeric:tabular-nums;line-height:1}
.gauge-label{margin-top:0.2rem;font-size:0.65rem;font-weight:600;letter-spacing:0.15em;text-transform:uppercase;color:var(--muted)}

/* ---- verdict badge (mirrors ui.tsx Badge) ---- */
.verdict-row{display:flex;justify-content:center;margin-top:0.85rem}
#label{display:inline-flex;align-items:center;gap:0.4rem;font-size:0.8rem;font-weight:700;letter-spacing:0.03em;
       padding:0.45rem 1rem;border-radius:999px;border:1px solid var(--violet);color:var(--violet);
       background:rgba(167,139,250,0.1);transition:all 0.3s}
#label::before{content:'';width:0.4rem;height:0.4rem;border-radius:50%;background:currentColor}
#rawline{color:var(--muted2);font-size:0.75rem;margin-top:0.9rem;font-variant-numeric:tabular-nums;text-align:center}
#stateNote{color:var(--violet);font-size:0.78rem;max-width:32rem;text-align:center;min-height:1.1em;margin-top:0.5rem}

canvas{width:100%;max-width:100%;height:110px;background:rgba(2,6,23,0.4);border:1px solid var(--line);
       border-radius:0.75rem;margin-top:1.25rem;display:block}
#latency{color:var(--muted2);font-size:0.72rem;margin-top:0.6rem;text-align:center}
#err{color:#fca5a5;margin-top:0.6rem;text-align:center;font-size:0.82rem}

.disclaimer{max-width:36rem;font-size:0.72rem;color:var(--muted3);text-align:center;
            padding-top:0.25rem;margin-top:0.25rem;line-height:1.6}

/* ---- report ---- */
.report-overall{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;
                 background:rgba(2,6,23,0.4);border:1px solid var(--line);border-radius:0.75rem;padding:1rem 1.25rem}
.report-overall .big{font-size:1.5rem;font-weight:800;letter-spacing:-0.01em}
.report-overall .strength{font-size:0.75rem;padding:0.25rem 0.65rem;border-radius:999px;font-weight:700}
.report-bar{width:100%;height:0.6rem;background:rgba(2,6,23,0.6);border-radius:999px;overflow:hidden;margin-top:0.9rem}
.report-bar-fill{height:100%;background:linear-gradient(90deg,var(--human),var(--uncertain),var(--ai));border-radius:999px}
.count-row{display:flex;gap:1.5rem;margin-top:0.9rem;font-size:0.78rem;color:var(--muted);flex-wrap:wrap}
.count-row b{color:var(--text);font-size:1.15rem;display:block;font-weight:700}
.report h2:not(.card-title){font-size:0.65rem;color:var(--muted2);letter-spacing:0.18em;text-transform:uppercase;
           font-weight:700;margin:1.5rem 0 0.6rem}
.flags-list{list-style:none;padding:0;margin:0;font-size:0.85rem;line-height:1.9}
.flags-list li.yes{color:#6ee7b7}
.flags-list li.no{color:var(--muted3)}
table.acoustic{width:100%;border-collapse:collapse;font-size:0.76rem;margin-top:0.5rem}
table.acoustic th{text-align:left;color:var(--muted2);font-weight:600;padding:0.4rem 0.5rem;border-bottom:1px solid var(--line)}
table.acoustic td{padding:0.4rem 0.5rem;border-bottom:1px solid var(--line-soft);color:#cbd5e1}
table.acoustic td.tag{font-size:0.7rem;color:var(--blue)}
.window-list{font-size:0.8rem;font-variant-numeric:tabular-nums;line-height:1.8;color:var(--muted)}
.window-list .w-ai{color:var(--ai)}
.window-list .w-human{color:var(--human)}
.timeline-track{display:flex;width:100%;height:1.4rem;border-radius:0.4rem;overflow:hidden;
                 background:rgba(2,6,23,0.6);margin-top:0.5rem}
.timeline-seg{height:100%}
.timeline-seg.ai{background:var(--ai)}
.timeline-seg.human{background:var(--human)}
.timeline-seg.uncertain{background:var(--uncertain)}
.timeline-seg.silence{background:var(--line)}
.timeline-seg.stabilizing{background:var(--violet-strong)}
.timeline-seg.speaker-change-marker{width:3px;background:var(--violet);flex-shrink:0}
.timeline-labels{display:flex;justify-content:space-between;font-size:0.68rem;color:var(--muted3);margin-top:0.35rem}
.report-empty{color:var(--muted3);font-size:0.85rem;font-style:italic}
</style></head><body>
<div class="topbar">
  <div class="topbar-inner">
    <div class="brand">
      <div class="brand-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/><path d="m9 12 2 2 4-4"/></svg>
      </div>
      <div>
        <div class="brand-title-row">
          <span class="brand-title">Live AI Voice Detector</span>
          <span class="brand-badge">Real-Time</span>
        </div>
        <p class="brand-sub">Analyzing acoustic patterns as they happen</p>
      </div>
    </div>
    <div class="status-row"><span id="dot" class="dot"></span><span id="micstatus">READY</span></div>
  </div>
  <div id="micTestBanner" style="display:none"></div>
</div>

<div class="page">

  <div class="card" style="text-align:center">
    <div class="controls">
      <button id="startBtn" onclick="startDetection()">Start Detection</button>
      <button id="stopBtn" onclick="stopDetection()" style="display:none">Stop Detection</button>
    </div>

    <div class="gauge-wrap">
      <svg viewBox="0 0 240 145" class="gauge-svg">
        <path class="gauge-track" d="M 20 120 A 100 100 0 0 1 220 120"/>
        <path id="gaugeArc" class="gauge-arc" d="M 20 120 A 100 100 0 0 1 220 120" pathLength="100" stroke-dasharray="0 100"/>
        <text class="gauge-tick" x="20" y="138">0</text>
        <text class="gauge-tick" x="120" y="30">50</text>
        <text class="gauge-tick" x="220" y="138">100</text>
        <line id="gaugeNeedle" x1="120" y1="120" x2="120" y2="43"/>
        <circle cx="120" cy="120" r="6" id="gaugeHub"/><circle cx="120" cy="120" r="2.5" fill="var(--ink)"/>
      </svg>
      <div class="gauge-readout">
        <div id="score">--</div>
        <div class="gauge-label">AI Likelihood</div>
      </div>
    </div>
    <div class="verdict-row"><div id="label">Analysing</div></div>
    <div id="stateNote"></div>
    <div id="rawline">raw: -- &nbsp;|&nbsp; smoothed: --</div>
    <canvas id="history" width="600" height="120"></canvas>
    <div id="latency">-- ms/window</div>
    <div id="err"></div>
  </div>

  <div class="disclaimer">
    Prototype trained on this dataset; external validation is currently limited.
    AI likelihood represents the model's score, not a guaranteed determination.
    This report shows model evidence gathered so far, not proof of synthetic generation.
  </div>

  <div class="card report" id="reportSection">
    <div class="card-head"><div><p class="eyebrow">Session Evidence</p><h2 class="card-title" style="font-size:0.95rem;margin:0.15rem 0 0">Analysis Report</h2></div></div>
    <div id="reportBody"><div class="report-empty">No speech analyzed yet -- start detection or run a file simulation.</div></div>
  </div>

</div>

<script>
const colors = {"AI likely":"#ef4444","Human likely":"#10b981","Analysing":"#a78bfa","Uncertain":"#f59e0b",
                "Silence":"#475569","Error":"#ef4444","Speaker change detected":"#a78bfa","Stabilizing":"#7c3aed"};
const canvas = document.getElementById('history');
const ctx = canvas.getContext('2d');

function drawHistory(history){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle = '#1f2937';
  for(const frac of [0.10,0.90]){  // must track app/decision.py: HUMAN_THRESHOLD, AI_THRESHOLD
    const y = canvas.height * (1 - frac);
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(canvas.width,y); ctx.stroke();
  }
  if(!history || history.length < 2) return;
  ctx.strokeStyle = '#60a5fa'; ctx.lineWidth = 2; ctx.beginPath();
  history.forEach((v,i)=>{
    const x = (i/(history.length-1)) * canvas.width;
    const y = canvas.height * (1 - v);
    i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
  });
  ctx.stroke();
}

// Drives the semi-circular gauge (arc fill + needle angle + hub color) --
// mirrors src/components/ConfidenceGauge.tsx's geometry (same viewBox,
// same -90deg..+90deg sweep), extended from a binary AI/Human indicator
// to this app's 3-way AI/Human/Uncertain verdict via `color`.
function updateGauge(percent, color){
  const p = Math.max(0, Math.min(100, percent));
  const arc = document.getElementById('gaugeArc');
  const needle = document.getElementById('gaugeNeedle');
  const hub = document.getElementById('gaugeHub');
  arc.setAttribute('stroke-dasharray', `${p} 100`);
  arc.style.stroke = color;
  needle.setAttribute('transform', `rotate(${-90 + p * 1.8} 120 120)`);
  needle.style.stroke = color;
  hub.style.fill = color;
}

function renderResult(d, historyArr){
  const labelEl = document.getElementById('label');
  const noteEl = document.getElementById('stateNote');
  const label = d.speaker_change ? 'Speaker change detected' : d.stabilizing ? 'Stabilizing' :
                d.silence ? 'Silence' : (d.label || 'Analysing');
  labelEl.textContent = label;
  const c = colors[label] || '#f59e0b';
  labelEl.style.color = c; labelEl.style.borderColor = c; labelEl.style.background = c + '1a';

  if(d.speaker_change){
    noteEl.textContent = 'A different speaker appears to have started talking. Prediction paused while the new speaker is identified.';
  } else if(d.stabilizing){
    noteEl.textContent = 'Listening for stable speech from the new speaker before resuming prediction...';
  } else {
    noteEl.textContent = '';
  }

  if(d.smoothed !== null && d.smoothed !== undefined){
    document.getElementById('score').textContent = (d.smoothed*100).toFixed(1) + '%';
    updateGauge(d.smoothed*100, c);
    document.getElementById('rawline').textContent =
      `raw: ${(d.raw*100).toFixed(1)}%  |  smoothed: ${(d.smoothed*100).toFixed(1)}%`;
  } else {
    document.getElementById('score').textContent = d.silence ? '(silence)' :
      (d.speaker_change || d.stabilizing) ? '(paused)' : '--';
    updateGauge(0, c);
    document.getElementById('rawline').textContent = 'raw: --  |  smoothed: --';
  }
  if(d.total_ms){
    const featStr = d.feature_ms ? ` + features ${d.feature_ms.toFixed(0)}ms` : '';
    document.getElementById('latency').textContent =
      `~${d.total_ms.toFixed(0)}ms/window (preprocess ${d.preprocess_ms?.toFixed(0)||0}ms, inference ${d.inference_ms?.toFixed(0)||0}ms${featStr})`;
  }
  drawHistory(historyArr);
  document.getElementById('err').textContent = d.error || '';
}

function fmtTime(s){
  const m = Math.floor(s/60), sec = Math.floor(s%60);
  return `${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
}

function renderReport(report){
  const body = document.getElementById('reportBody');
  if(!report || !report.n_windows){
    body.innerHTML = '<div class="report-empty">No speech analyzed yet -- start detection or run a file simulation.</div>';
    return;
  }
  const strengthColor = {STRONG:'#ef4444', MODERATE:'#f59e0b', WEAK:'#64748b'}[report.evidence_strength] || '#64748b';
  const overallColor = colors[report.overall_result] || '#f59e0b';
  const aiShare = report.n_windows ? Math.round(100*report.ai_count/report.n_windows) : 0;

  let html = '';
  html += `<div class="report-overall">
    <div><div style="color:#64748b;font-size:0.72rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase">Overall Result</div>
         <div class="big" style="color:${overallColor}">${report.overall_result}</div></div>
    <div><div style="color:#64748b;font-size:0.72rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase">Evidence Strength</div>
         <div class="strength" style="background:${strengthColor}22;color:${strengthColor}">${report.evidence_strength}</div></div>
  </div>`;
  html += `<div class="report-bar"><div class="report-bar-fill" style="width:${aiShare}%"></div></div>`;
  html += `<div class="count-row">
    <div>Speech windows<b>${report.n_windows}</b></div>
    <div>Speech analyzed<b>${report.speech_seconds}s</b></div>
    <div>AI-like<b style="color:#ef4444">${report.ai_count}</b></div>
    <div>Uncertain<b style="color:#f59e0b">${report.uncertain_count}</b></div>
    <div>Human-like<b style="color:#10b981">${report.human_count}</b></div>
  </div>`;

  html += `<h2>Speaker Changes</h2><div style="font-size:0.85rem;color:#94a3b8">${report.speaker_changes} detected</div>`;
  html += '<h2>Speaker Segments</h2><div class="window-list">';
  report.speaker_segments.forEach(s=>{
    const range = (s.t_start !== null && s.t_end !== null) ? `${fmtTime(s.t_start)}&ndash;${fmtTime(s.t_end)}` : '';
    if(s.status){
      html += `<div style="margin-bottom:0.5rem"><b>Speaker ${s.speaker_index}</b> ${range}<br>
               <span style="color:#94a3b8">Status: ${s.status}</span></div>`;
    } else {
      const rc = colors[s.overall_result] || '#f59e0b';
      html += `<div style="margin-bottom:0.5rem"><b>Speaker ${s.speaker_index}</b> ${range}<br>
               ${s.n_windows} speech windows &mdash;
               <span style="color:${rc}">${s.overall_result}</span>
               (${s.evidence_strength} evidence)</div>`;
    }
  });
  html += '</div>';

  html += '<h2>What The Model Is Flagging</h2><ul class="flags-list">';
  (report.flags.length ? report.flags : ['No consistent pattern flagged yet']).forEach(f=>{
    html += `<li class="${report.flags.length ? 'yes' : 'no'}">${report.flags.length ? '&#10003;' : '&#8226;'} ${f}</li>`;
  });
  html += '</ul>';
  html += '<h2>Not Enough Evidence</h2><ul class="flags-list">';
  report.not_enough_evidence.forEach(f=>{ html += `<li class="no">&#8226; ${f}</li>`; });
  html += '</ul>';

  html += '<h2>Acoustic Observations</h2><table class="acoustic"><tr><th>Feature</th><th>Mean</th><th>Within-session variation</th><th>Diagnostic status</th></tr>';
  report.acoustic_summary.forEach(row=>{
    const meanStr = row.mean === null ? '--' : (Math.abs(row.mean) >= 10 ? row.mean.toFixed(1) : row.mean.toFixed(4));
    html += `<tr><td>${row.feature}</td><td>${meanStr}${row.unit ? ' '+row.unit : ''}</td>
             <td>${row.variation_label}</td><td class="tag" title="${row.research_note}">${row.diagnostic_label}</td></tr>`;
  });
  html += '</table>';

  if(report.strongest_windows.length){
    html += '<h2>Strongest AI-Like Windows</h2><div class="window-list">';
    report.strongest_windows.forEach(w=>{
      html += `<div class="w-ai">${fmtTime(w.t_start)}&ndash;${fmtTime(w.t_end)}&nbsp;&nbsp;AI score ${(w.score*100).toFixed(0)}%</div>`;
    });
    html += '</div>';
  }
  if(report.weakest_windows.length){
    html += '<h2>Lowest AI-Like Windows</h2><div class="window-list">';
    report.weakest_windows.forEach(w=>{
      html += `<div>${fmtTime(w.t_start)}&ndash;${fmtTime(w.t_end)}&nbsp;&nbsp;AI score ${(w.score*100).toFixed(0)}%</div>`;
    });
    html += '</div>';
  }

  if(report.timeline && report.timeline.length){
    const tEnd = Math.max(...report.timeline.map(s=>s.t_end));
    html += '<h2>Timeline</h2><div class="timeline-track">';
    report.timeline.forEach(seg=>{
      if(seg.type === 'speaker_change'){
        // Zero-duration marker -- render as a thin divider, not a colored bar.
        html += `<div class="timeline-seg speaker-change-marker" title="Speaker change at ${fmtTime(seg.t_start)}"></div>`;
        return;
      }
      const widthPct = 100 * (seg.t_end - seg.t_start) / tEnd;
      const cls = seg.type === 'silence' ? 'silence' : seg.type === 'stabilizing' ? 'stabilizing' :
                  seg.label === 'AI-like' ? 'ai' : seg.label === 'Human-like' ? 'human' : 'uncertain';
      const titleLabel = seg.type === 'stabilizing' ? 'Stabilizing new speaker' : seg.label;
      html += `<div class="timeline-seg ${cls}" style="width:${widthPct}%" title="${fmtTime(seg.t_start)}-${fmtTime(seg.t_end)}: ${titleLabel}${seg.speaker_index ? ' (speaker '+seg.speaker_index+')' : ''}"></div>`;
    });
    html += '</div><div class="timeline-labels"><span>0s</span><span>' + fmtTime(tEnd) + '</span></div>';
    html += '<div style="font-size:0.7rem;color:#475569;margin-top:0.25rem">'
          + '<span style="color:#ef4444">&#9608;</span> AI-like &nbsp;'
          + '<span style="color:#10b981">&#9608;</span> Human-like &nbsp;'
          + '<span style="color:#f59e0b">&#9608;</span> Uncertain &nbsp;'
          + '<span style="color:#1f2937">&#9608;</span> Silence &nbsp;'
          + '<span style="color:#7c3aed">&#9608;</span> Stabilizing &nbsp;'
          + '<span style="color:#a78bfa">|</span> Speaker change'
          + '</div>';
  }

  body.innerHTML = html;
}

function resetDisplay(){
  document.getElementById('label').textContent = 'Analysing';
  document.getElementById('label').style.color = '#a78bfa';
  document.getElementById('label').style.borderColor = '#a78bfa';
  document.getElementById('label').style.background = '#a78bfa1a';
  document.getElementById('score').textContent = '--';
  updateGauge(0, '#a78bfa');
  document.getElementById('rawline').textContent = 'raw: --  |  smoothed: --';
  document.getElementById('latency').textContent = '-- ms/window';
  document.getElementById('err').textContent = '';
  document.getElementById('stateNote').textContent = '';
  drawHistory([]);
  renderReport(null);
}

// ---- Microphone (browser-captured, WebSocket) ----
let ws = null, audioCtx = null, micStream = null, scriptNode = null;
let micState = 'READY';       // READY | STARTING | LIVE | STOPPED | ERROR
let activeSource = 'none';    // 'none' | 'mic' | 'file'
let micHistory = [];

// TEMPORARY LATENCY DIAGNOSTIC (outputs/reports/live_stream_latency.md) --
// opt-in via ?diag=1 in the page URL. When off (default), nothing below
// changes: same ScriptProcessorNode capture, same raw-Float32 WebSocket
// send, same message handling. Safe to delete this whole block (and the
// matching backend "diag" branch in app/main.py:ws_mic) once done.
const DIAG = new URLSearchParams(location.search).get('diag') === '1';
window._diagSamples = [];
window._diagSummary = function(){
  const s = window._diagSamples;
  if(!s.length){ console.log('No diag samples yet.'); return; }
  const pick = k => s.map(x=>x[k]).sort((a,b)=>a-b);
  const stat = arr => ({median: arr[Math.floor(arr.length*0.5)], p95: arr[Math.floor(arr.length*0.95)]});
  const out = {
    n: s.length,
    capture_to_backend_ms: stat(pick('captureToBackend')),
    backend_processing_ms: stat(pick('backendProcessing')),
    backend_to_browser_ms: stat(pick('backendToBrowser')),
    end_to_end_ms: stat(pick('endToEnd')),
  };
  console.table(out);
  return out;
};

// TEMPORARY MIC-PROCESSING A/B DIAGNOSTIC
// (outputs/reports/mic_processing_ab_test.md) -- opt-in via ?micraw=1.
// Default (micraw absent/0): unchanged getUserMedia({audio:true}), i.e.
// Chrome's normal default echoCancellation/noiseSuppression/autoGainControl
// apply. micraw=1: requests the mic with all three explicitly disabled,
// to test whether that browser-side voice processing is what's altering
// live CNN scores relative to file-mode. Everything else (WebSocket,
// chunk size, backend, model, decision layer) is untouched either way.
// Safe to delete this whole block once the experiment is done.
const MIC_RAW = new URLSearchParams(location.search).get('micraw') === '1';
if(MIC_RAW){
  const banner = document.getElementById('micTestBanner');
  banner.style.display = 'block';
  banner.textContent = 'MIC A/B TEST: RAW mode (echoCancellation/noiseSuppression/autoGainControl disabled)';
}
window._micTest = {
  mode: MIC_RAW ? 'RAW (echoCancellation/noiseSuppression/autoGainControl=false)' : 'CURRENT ({audio:true}, browser defaults)',
  requestedConstraints: null, trackSettings: null,
  tStart: null, tFirstScore: null, tFirstConfirmedLabel: null,
  results: [], aiConfirmations: 0, humanConfirmations: 0,
  _lastLabel: null,
};
window._micTestReport = function(){
  const t = window._micTest;
  const scores = t.results.filter(r => r.raw !== null).map(r => r.raw);
  const summary = {
    mode: t.mode,
    requestedConstraints: t.requestedConstraints,
    actualTrackSettings: t.trackSettings,
    n_results: t.results.length,
    n_scored: scores.length,
    score_mean: scores.length ? scores.reduce((a,b)=>a+b,0)/scores.length : null,
    score_max: scores.length ? Math.max(...scores) : null,
    score_min: scores.length ? Math.min(...scores) : null,
    ai_confirmations: t.aiConfirmations,
    human_confirmations: t.humanConfirmations,
    time_to_first_score_ms: (t.tFirstScore && t.tStart) ? (t.tFirstScore - t.tStart) : null,
    time_to_first_confirmed_label_ms: (t.tFirstConfirmedLabel && t.tStart) ? (t.tFirstConfirmedLabel - t.tStart) : null,
  };
  console.table(summary);
  console.log('Full result timeline (copy this for the report):', JSON.stringify(t.results));
  return summary;
};

function setMicState(s, errMsg){
  micState = s;
  const dot = document.getElementById('dot');
  document.getElementById('micstatus').textContent = s;
  dot.className = 'dot' + (s === 'LIVE' ? ' live' : (s === 'ERROR' ? ' err' : ''));
  document.getElementById('startBtn').style.display = (s === 'LIVE' || s === 'STARTING') ? 'none' : 'inline-block';
  document.getElementById('stopBtn').style.display = (s === 'LIVE' || s === 'STARTING') ? 'inline-block' : 'none';
  if(errMsg) document.getElementById('err').textContent = errMsg;
}

async function startDetection(){
  if(micState === 'STARTING' || micState === 'LIVE') return;
  activeSource = 'mic';
  micHistory = [];
  setMicState('STARTING');

  window._micTest.tStart = Date.now();
  window._micTest.tFirstScore = null;
  window._micTest.tFirstConfirmedLabel = null;
  window._micTest.results = [];
  window._micTest.aiConfirmations = 0;
  window._micTest.humanConfirmations = 0;
  window._micTest._lastLabel = null;
  const micConstraints = MIC_RAW
    ? {audio: {echoCancellation: false, noiseSuppression: false, autoGainControl: false}}
    : {audio: true};
  window._micTest.requestedConstraints = micConstraints.audio;

  try{
    micStream = await navigator.mediaDevices.getUserMedia(micConstraints);
  }catch(e){
    setMicState('ERROR', 'Microphone permission denied or unavailable: ' + e.message);
    activeSource = 'none';
    return;
  }
  const _micTrack = micStream.getAudioTracks()[0];
  window._micTest.trackSettings = _micTrack && _micTrack.getSettings ? _micTrack.getSettings() : null;
  console.log('[micTest] mode:', window._micTest.mode);
  console.log('[micTest] requested constraints:', window._micTest.requestedConstraints);
  console.log('[micTest] actual track settings:', window._micTest.trackSettings);

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(micStream);
  // ScriptProcessorNode is deprecated but universally supported and by
  // far the simplest way to get raw PCM samples for this hackathon
  // prototype (AudioWorklet would need a separate module file for no
  // real benefit here). A muted gain node is required downstream so
  // onaudioprocess actually fires in all browsers WITHOUT audibly
  // routing the live mic signal to the speakers (which would cause
  // feedback/echo during a presentation).
  scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
  const silentGain = audioCtx.createGain();
  silentGain.gain.value = 0;

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const diagParam = DIAG ? '&diag=1' : '';
  ws = new WebSocket(`${proto}://${location.host}/ws/mic?sr=${Math.round(audioCtx.sampleRate)}${diagParam}`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    setMicState('LIVE');
    source.connect(scriptNode);
    scriptNode.connect(silentGain);
    silentGain.connect(audioCtx.destination);
    scriptNode.onaudioprocess = (e) => {
      if(!ws || ws.readyState !== WebSocket.OPEN) return;
      const input = e.inputBuffer.getChannelData(0);
      if(DIAG){
        // prepend an 8-byte float64 capture timestamp (epoch ms) so the
        // backend can echo it back for round-trip latency measurement.
        const out = new ArrayBuffer(8 + input.length * 4);
        new Float64Array(out, 0, 1)[0] = Date.now();
        new Float32Array(out, 8).set(input);
        ws.send(out);
      } else {
        ws.send(input.slice().buffer);  // copy: the source buffer is reused next callback
      }
    };
  };
  ws.onmessage = (evt) => {
    const d = JSON.parse(evt.data);
    if(activeSource !== 'mic') return;
    if(d.type === 'error'){ setMicState('ERROR', d.error); return; }
    if(DIAG && d._diag){
      const tBrowserReceive = Date.now();
      const sample = {
        captureToBackend: d._diag.t_backend_receive - d._diag.capture_ts,
        backendProcessing: d._diag.t_response_send - d._diag.t_backend_receive,
        backendToBrowser: tBrowserReceive - d._diag.t_response_send,
        endToEnd: tBrowserReceive - d._diag.capture_ts,
      };
      window._diagSamples.push(sample);
      console.log('[diag]', sample, 'total_ms(server)=', d.total_ms);
    }
    if(d.smoothed !== null && d.smoothed !== undefined){
      micHistory.push(d.smoothed);
      if(micHistory.length > 60) micHistory.shift();
    }
    // TEMPORARY MIC-PROCESSING A/B DIAGNOSTIC bookkeeping (see mode toggle
    // above) -- records every result so window._micTestReport() can
    // summarize score distribution / confirmation counts / first-latency.
    const now = Date.now();
    if(!d.silence && d.raw !== null && d.raw !== undefined && window._micTest.tFirstScore === null){
      window._micTest.tFirstScore = now;
    }
    const t = window._micTest;
    if(d.label !== t._lastLabel){
      if(d.label === 'AI likely') t.aiConfirmations++;
      if(d.label === 'Human likely') t.humanConfirmations++;
      if((d.label === 'AI likely' || d.label === 'Human likely') && t.tFirstConfirmedLabel === null){
        t.tFirstConfirmedLabel = now;
      }
      t._lastLabel = d.label;
    }
    t.results.push({t: now, raw: d.raw, smoothed: d.smoothed, label: d.label, silence: d.silence});
    renderResult(d, micHistory);
    renderReport(d.report);
  };
  ws.onerror = () => { setMicState('ERROR', 'WebSocket connection error.'); };
  ws.onclose = () => { if(micState !== 'STOPPED' && micState !== 'ERROR') setMicState('STOPPED'); };
}

function stopDetection(){
  activeSource = 'none';
  if(ws){ try{ ws.close(); }catch(e){} ws = null; }
  if(scriptNode){ try{ scriptNode.disconnect(); }catch(e){} scriptNode = null; }
  if(audioCtx){ try{ audioCtx.close(); }catch(e){} audioCtx = null; }
  if(micStream){ micStream.getTracks().forEach(t => t.stop()); micStream = null; }  // actually releases the OS mic
  setMicState('STOPPED');
  resetDisplay();
}

resetDisplay();
</script>
</body></html>
"""
