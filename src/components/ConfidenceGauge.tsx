import { AlertTriangle, CheckCircle2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { Badge } from './ui';

export function ConfidenceGauge({ confidence, synthetic }: { confidence: number; synthetic: boolean }) {
  const percent = Math.max(0, Math.min(100, confidence * 100));
  const angle = -90 + percent * 1.8;
  const color = synthetic ? '#ef4444' : '#10b981';
  return <div className="text-center">
    <div className="relative mx-auto h-56 max-w-[360px]" role="img" aria-label={`${percent.toFixed(0)} percent ${synthetic ? 'synthetic' : 'human'} voice confidence`}>
      <svg viewBox="0 0 240 145" className="h-full w-full overflow-visible">
        <defs><linearGradient id="gaugeDanger" x1="0" x2="1"><stop stopColor="#f59e0b" /><stop offset="1" stopColor="#ef4444" /></linearGradient><linearGradient id="gaugeSafe" x1="0" x2="1"><stop stopColor="#10b981" /><stop offset="1" stopColor="#3b82f6" /></linearGradient></defs>
        <path d="M 20 120 A 100 100 0 0 1 220 120" fill="none" stroke="#1f2937" strokeWidth="18" strokeLinecap="round" />
        <path d="M 20 120 A 100 100 0 0 1 220 120" fill="none" stroke={`url(#${synthetic ? 'gaugeDanger' : 'gaugeSafe'})`} strokeWidth="18" strokeLinecap="round" pathLength="100" strokeDasharray={`${percent} 100`} />
        {[0, 20, 40, 60, 80, 100].map((tick) => { const radians = ((180 - tick * 1.8) * Math.PI) / 180; const x = 120 + 106 * Math.cos(radians); const y = 120 - 106 * Math.sin(radians); return <text key={tick} x={x} y={y} fill="#64748b" fontSize="8" textAnchor="middle">{tick}%</text>; })}
        <motion.line x1="120" y1="120" x2="120" y2="43" stroke={color} strokeWidth="2.5" strokeLinecap="round" initial={{ rotate: -90 }} animate={{ rotate: angle }} transition={{ type: 'spring', stiffness: 55, damping: 14 }} style={{ transformOrigin: '120px 120px' }} />
        <circle cx="120" cy="120" r="6" fill={color} /><circle cx="120" cy="120" r="2.5" fill="#0a0e1a" />
      </svg>
      <div className="absolute inset-x-0 bottom-3"><p className="text-5xl font-bold tracking-tight text-white">{percent.toFixed(0)}<span className="text-2xl text-slate-500">%</span></p><p className="mt-1 text-xs font-medium uppercase tracking-widest text-slate-400">{synthetic ? 'Synthetic Voice Confidence' : 'Human Voice Confidence'}</p></div>
    </div>
    <Badge className={synthetic ? 'border-red-500/30 bg-red-500/10 px-4 py-2 text-red-300' : 'border-emerald-500/30 bg-emerald-500/10 px-4 py-2 text-emerald-300'}>{synthetic ? <AlertTriangle size={15} className="mr-2" /> : <CheckCircle2 size={15} className="mr-2" />}{synthetic ? 'SYNTHETIC VOICE DETECTED' : 'HUMAN VOICE VERIFIED'}</Badge>
  </div>;
}
