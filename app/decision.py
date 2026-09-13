"""
Live decision layer: turns a raw per-window CNN probability into a
stable, human-readable classification for the demo UI. It does NOT
change what the model outputs -- it only smooths it over time and
requires several consecutive windows to agree before flipping the
displayed label, so the UI doesn't flicker on a single noisy frame. The
raw, unsmoothed probability is always kept available too (for debugging
/ the benchmark and known-examples scripts).

IMPORTANT (see outputs/reports/cnn_training_report.md): internal
validation ROC-AUC is ~1.00 but the existing external-validation set
drops to ~0.53 (chance level). The label this class produces is a
"model score", NOT a scientifically validated probability that speech is
AI-generated. Thresholds below are simple, fixed, interpretable demo
defaults -- not fit/tuned on any validation or test result.

DECISION THRESHOLDS -- the single, clearly-defined configuration point
for this whole app (app/evidence.py imports these same two constants
rather than redefining them, so changing them here is sufficient):
AI_THRESHOLD / HUMAN_THRESHOLD. Conservative-by-design: a wide band
(0.10-0.90) is deliberately treated as inconclusive rather than forced
into AI/Human, so the detector only commits to a verdict on strong
evidence. `smoothed` is the model's own sigmoid output for the
synthetic/AI class (LABEL_TO_INT = {"human": 0, "synthetic": 1} in
modeling/train_cnn.py) averaged over SMOOTHING_WINDOW raw windows --
never inverted, never re-derived.

DISPLAYED LABEL IS STICKY: an inconclusive stretch never overwrites a
already-confirmed "AI likely"/"Human likely" verdict for the current
speaker -- it only shows "Analysing" before the first verdict exists.
See update()'s commit guard.
"""
from collections import deque

AI_THRESHOLD = 0.90        # smoothed score >= this -> "AI likely"
HUMAN_THRESHOLD = 0.10     # smoothed score <= this -> "Human likely"
SMOOTHING_WINDOW = 5       # raw window scores averaged into the displayed number
CONSECUTIVE_REQUIRED = 2   # windows needed, in agreement, before the label is allowed to flip
SILENCE_RMS_THRESHOLD = 0.01   # pre-normalization RMS below this -> treated as silence, not scored

# CONSECUTIVE_REQUIRED was 3 (~4.5s); demo testing on real phone-call
# recordings (natural conversational pauses every few seconds) found a
# synthetic recording averaging 0.98 raw score STILL ending the call as
# "Uncertain", because no single speech stretch survived uninterrupted
# for 3 full windows before the next pause. Lowered to 2 (~3s) -- a
# short confirmation period, not a threshold change (AI_THRESHOLD/
# HUMAN_THRESHOLD are untouched) -- per the explicit guidance to prefer
# smoothing/a short confirmation period over lowering thresholds.


class DecisionEngine:
    def __init__(self):
        self.history = deque(maxlen=SMOOTHING_WINDOW)
        self.label = "Analysing"   # shown only until the first AI/Human verdict is confirmed
        self._candidate_label = None
        self._candidate_count = 0

    @staticmethod
    def _classify(smoothed: float) -> str:
        """Internal per-window candidate only -- "Uncertain" here never
        reaches the UI directly (see update()'s commit guard below); it
        just means "this window's score doesn't clear either threshold"."""
        if smoothed >= AI_THRESHOLD:
            return "AI likely"
        if smoothed <= HUMAN_THRESHOLD:
            return "Human likely"
        return "Uncertain"

    def reset_for_new_speaker(self):
        """
        Unlike silence (which must NOT reset anything, see update()
        below), a CONFIRMED speaker change (app/speaker_guard.py) is a
        deliberate, explicit reset: the previous speaker's smoothing
        history and confirmed label must not leak into the new
        speaker's evidence. Called exactly once, the moment a speaker
        change is confirmed.
        """
        self.history.clear()
        self.label = "Analysing"
        self._candidate_label = None
        self._candidate_count = 0

    def update(self, raw_prob, is_silence: bool = False) -> dict:
        """raw_prob is None when is_silence=True. Returns the current
        raw score, smoothed score, and the (possibly still-debouncing)
        displayed label.

        Silence is a COOLDOWN, not a reset: per this project's dataset
        structure, the target speaker is consistently either human or
        synthetic for the whole recording, so a pause between turns
        carries no information either way. On silence this method does
        not touch `history`, `_candidate_label`, or `_candidate_count` at
        all -- no new evidence is added and none of the accumulated
        evidence is discarded. The UI still displays "Silence" for that
        instant (via the `silence` flag below), but the underlying
        evidence state is completely untouched and resumes exactly where
        it left off once speech returns.
        """
        if is_silence:
            return {"raw": None, "smoothed": None, "label": "Silence", "silence": True}

        self.history.append(raw_prob)
        smoothed = sum(self.history) / len(self.history)
        candidate = self._classify(smoothed)

        if candidate == self._candidate_label:
            self._candidate_count += 1
        else:
            self._candidate_label = candidate
            self._candidate_count = 1

        # Only a confident ("AI likely" / "Human likely") candidate is ever
        # allowed to become the displayed verdict. An ambiguous stretch
        # (candidate == "Uncertain") never overwrites self.label -- the UI
        # keeps showing whatever was last confirmed (or "Analysing" if
        # nothing has been confirmed yet for this speaker) instead of
        # flapping back to a neutral state on every noisy window.
        if candidate != "Uncertain" and self._candidate_count >= CONSECUTIVE_REQUIRED:
            self.label = candidate

        return {"raw": raw_prob, "smoothed": smoothed, "label": self.label, "silence": False}
