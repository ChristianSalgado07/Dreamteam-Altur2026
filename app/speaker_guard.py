"""
Speaker-change GUARD (not speaker recognition, not voice-identity
verification -- see module docstring rationale below). A lightweight
heuristic that pauses AI/Human prediction when a SUSTAINED shift in a
few already-computed acoustic characteristics (pitch primarily,
corroborated by spectral shape) suggests the target channel has
switched to a different speaker, so one speaker's acoustic
characteristics can't contaminate the prediction attributed to another.

Explicitly NOT built:
  - no speaker-recognition model, no embeddings, no neural network here
  - no claim of proving two voices belong to different people -- only
    that a significant, PERSISTENT acoustic change was observed,
    consistent with (not proof of) a speaker change

Thresholds are grounded in this project's OWN existing audio, not
invented: a quick empirical check (this module's development, not
re-run at import time) of consecutive-window relative changes WITHIN
single-speaker recordings found:
  - F0 (median of voiced pYIN frames per window): same-speaker window-
    to-window relative change median ~0.05-0.06, p90 ~0.18-0.32,
    observed max ~0.44. F0_REL_CHANGE_THRESHOLD=0.40 sits at/above that
    natural ceiling -- deliberately conservative (natural intonation,
    emotion, a single high-pitched word must NOT trigger this alone).
  - Spectral centroid/bandwidth/ZCR are much noisier window-to-window
    even within one speaker (p90 ~0.6-0.7) -- so these are used ONLY as
    a secondary corroborating signal (checked only once pitch has
    already crossed its own threshold), never as the primary trigger.
    RMS is deliberately excluded from the secondary set: prior research
    (outputs/reports/modeling_evaluation_report.md) flagged RMS/peak
    amplitude as a likely recording-loudness confound, not a genuine
    speaker characteristic.
  - Persistence (CONFIRM_WINDOWS consecutive candidate windows) is what
    actually keeps false positives low -- a single window crossing both
    the pitch and a secondary threshold is still fairly plausible under
    natural variation, but two independent CONSECUTIVE windows doing so
    is much less likely by chance.

If F0 is unavailable for a window (too few voiced pYIN frames --
common in this project's 8kHz telephone audio, empirically ~30% of
speech windows), that window never counts as a candidate shift, in
either direction -- per the task requirement, a speaker change is never
declared from missing pitch data.
"""
import numpy as np

F0_REL_CHANGE_THRESHOLD = 0.40
SECONDARY_REL_CHANGE_THRESHOLD = 0.35
SECONDARY_FEATURES = ("spectral_centroid", "spectral_bandwidth", "zcr")
MIN_VOICED_FRAMES = 8       # below this, a window's F0 estimate is not trusted at all
CONFIRM_WINDOWS = 2         # consecutive candidate windows required to CONFIRM a speaker change
STABILIZATION_WINDOWS = 2   # additional windows after confirmation before prediction resumes
EMA_ALPHA = 0.3             # profile update rate for a confirmed, ongoing speaker


class SpeakerProfile:
    def __init__(self, f0_median, features: dict):
        self.f0_median = f0_median
        self.features = dict(features)

    def update(self, f0_median, features: dict):
        if f0_median is not None:
            self.f0_median = (f0_median if self.f0_median is None
                               else EMA_ALPHA * f0_median + (1 - EMA_ALPHA) * self.f0_median)
        for k in SECONDARY_FEATURES:
            if k in features:
                self.features[k] = (features[k] if k not in self.features
                                     else EMA_ALPHA * features[k] + (1 - EMA_ALPHA) * self.features[k])


class SpeakerGuard:
    """
    Call `process(f0_median, features)` once per non-silent speech
    window (never for silence -- silence must not affect this guard at
    all, it is handled entirely upstream). Returns one of three phases:
      "normal"         -- same speaker (or not yet enough evidence to
                          say otherwise); prediction proceeds as usual.
      "speaker_change" -- returned exactly once, on the window where a
                          sustained shift was just CONFIRMED.
      "stabilizing"    -- the next STABILIZATION_WINDOWS windows after
                          confirmation; prediction stays paused while a
                          fresh profile is established from real data.
    """

    def __init__(self):
        self.profile = None
        self.pending_count = 0
        self.stabilizing_remaining = 0
        self._stabilizing_f0 = []
        self._stabilizing_features = {k: [] for k in SECONDARY_FEATURES}

    def _is_candidate_shift(self, f0_median, features: dict) -> bool:
        if self.profile is None or f0_median is None or self.profile.f0_median is None:
            return False
        rel = abs(f0_median - self.profile.f0_median) / self.profile.f0_median
        if rel <= F0_REL_CHANGE_THRESHOLD:
            return False
        for k in SECONDARY_FEATURES:
            baseline = self.profile.features.get(k)
            if k in features and baseline:
                if abs(features[k] - baseline) / baseline > SECONDARY_REL_CHANGE_THRESHOLD:
                    return True
        return False

    def process(self, f0_median, features: dict) -> dict:
        if self.stabilizing_remaining > 0:
            self._stabilizing_f0.append(f0_median)
            for k in SECONDARY_FEATURES:
                if k in features:
                    self._stabilizing_features[k].append(features[k])
            self.stabilizing_remaining -= 1
            if self.stabilizing_remaining == 0:
                valid_f0 = [f for f in self._stabilizing_f0 if f is not None]
                new_f0 = float(np.median(valid_f0)) if valid_f0 else None
                new_feats = {k: float(np.mean(v)) for k, v in self._stabilizing_features.items() if v}
                self.profile = SpeakerProfile(new_f0, new_feats)
                self._stabilizing_f0 = []
                self._stabilizing_features = {k: [] for k in SECONDARY_FEATURES}
                return {"phase": "normal", "new_speaker": False}
            return {"phase": "stabilizing", "new_speaker": False}

        if self.profile is None:
            self.profile = SpeakerProfile(f0_median, features)
            return {"phase": "normal", "new_speaker": False}

        if self._is_candidate_shift(f0_median, features):
            self.pending_count += 1
            if self.pending_count >= CONFIRM_WINDOWS:
                self.pending_count = 0
                self.stabilizing_remaining = STABILIZATION_WINDOWS
                return {"phase": "speaker_change", "new_speaker": True}
            # Below persistence threshold: per the task's explicit
            # requirement not to trigger from a single unusual window,
            # keep predicting normally with the OLD profile rather than
            # pausing on unconfirmed suspicion.
            return {"phase": "normal", "new_speaker": False}

        self.pending_count = 0
        self.profile.update(f0_median, features)
        return {"phase": "normal", "new_speaker": False}
