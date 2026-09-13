import { ChevronDown, ChevronUp, Clock3, Fingerprint, Radar } from 'lucide-react';
import { motion } from 'framer-motion';
import { useState } from 'react';
import { Card } from './ui';
import { ConfidenceGauge } from './ConfidenceGauge';
import { BreakdownBars } from './BreakdownBars';
import { RadarAnalysis } from './RadarAnalysis';
import { ActionBanner } from './ActionBanner';
import { LoadingAnalysis } from './LoadingAnalysis';
import type { AnalysisRecord } from '../lib/types';

export function ResultPanel({ result, loading }: { result: AnalysisRecord | null; loading: boolean }) {
  const [details, setDetails] = useState(false);
  if (loading) return <Card className="min-h-[520px] p-6"><LoadingAnalysis /></Card>;
  if (!result) return <Card className="flex min-h-[520px] flex-col items-center justify-center p-8 text-center"><div className="mb-5 grid size-16 place-items-center rounded-2xl bg-slate-800/70 text-slate-500"><Radar size={30} /></div><h3 className="text-lg font-semibold text-slate-300">Upload a call to begin analysis</h3><p className="mt-2 max-w-xs text-sm leading-relaxed text-slate-500">Your analysis cockpit will appear here once a stereo call recording is ready.</p></Card>;
  return <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }}><Card className={`overflow-hidden p-5 sm:p-6 ${result.is_synthetic ? 'result-danger' : 'result-safe'}`}><div className="mb-7 flex items-start justify-between"><div><p className="eyebrow">ANALYSIS COMPLETE</p><h2 className="mt-1 truncate text-xl font-semibold text-white">{result.filename}</h2></div><div className="flex items-center gap-1.5 text-xs text-slate-500"><Clock3 size={13} /> {new Date(result.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div></div><ConfidenceGauge confidence={result.confidence} synthetic={result.is_synthetic} /><div className="my-7 h-px bg-line" /><BreakdownBars breakdown={result.breakdown} confidence={result.confidence} synthetic={result.is_synthetic} /><div className="my-7 h-px bg-line" /><RadarAnalysis breakdown={result.breakdown} confidence={result.confidence} synthetic={result.is_synthetic} /><div className="my-7 h-px bg-line" /><ActionBanner synthetic={result.is_synthetic} /><button onClick={() => setDetails((value) => !value)} className="focus-ring mt-4 flex w-full items-center justify-center gap-2 rounded-lg py-2 text-xs font-semibold text-slate-500 hover:text-slate-200">{details ? 'Hide technical details' : 'Show technical details'}{details ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</button>{details && <div className="grid grid-cols-1 gap-3 rounded-xl border border-line bg-slate-950/35 p-4 text-xs text-slate-400 sm:grid-cols-3"><div><p className="eyebrow">RAW CONFIDENCE</p><p className="mt-1 font-mono text-slate-200">{result.rawConfidence.toFixed(6)}</p></div><div><p className="eyebrow">TIMESTAMP</p><p className="mt-1 text-slate-200">{new Date(result.timestamp).toISOString()}</p></div><div><p className="eyebrow">RECORD ID</p><p className="mt-1 truncate font-mono text-slate-200">{result.id}</p></div></div>}</Card></motion.div>;
}
