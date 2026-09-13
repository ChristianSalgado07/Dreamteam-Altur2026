export function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(' ');
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function parsePercent(value: string) {
  return Number.parseFloat(value.replace('%', '')) || 0;
}

export const breakdownMeta = [
  ['timing_and_environment', 'Timing & Environment', 'Turn latencies, overlap dynamics, ambient noise'],
  ['caller_acoustics', 'Caller Acoustics', 'MFCC, zero-crossing rate and spectral texture'],
  ['biological_and_phase', 'Biological & Phase', 'Micro-tremors, breathing proxies and vocoder artifacts'],
  ['confidence', 'Verdict Confidence', 'Aggregate confidence of the trained classifier'],
] as const;
