import { Activity, ShieldCheck } from 'lucide-react';
import { motion } from 'framer-motion';

export function Header({ isOnline, alert }: { isOnline: boolean; alert: boolean }) {
  return (
    <header className={alert ? 'border-b border-red-500/30 bg-[#0d111e] shadow-[0_0_28px_rgba(239,68,68,0.12)]' : 'border-b border-line bg-[#0d111e]/95'}>
      <div className="mx-auto flex max-w-[1440px] items-center justify-between px-5 py-4 sm:px-8">
        <div className="flex items-center gap-3">
          <div className="grid size-10 place-items-center rounded-xl bg-blue-500/10 text-blue-400 ring-1 ring-blue-400/25">
            <ShieldCheck size={24} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-lg font-bold tracking-tight text-white">Altur</span>
              <span className="rounded bg-blue-500/10 px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-[0.2em] text-blue-300">SECURITY OPS</span>
            </div>
            <p className="hidden text-xs text-slate-400 sm:block">Voice Deepfake Detection Console</p>
          </div>
        </div>
        <motion.div animate={alert ? { opacity: [1, .55, 1] } : { opacity: 1 }} transition={{ repeat: alert ? Infinity : 0, duration: 1.8 }} className="flex items-center gap-2 rounded-full border border-line bg-slate-950/40 px-3 py-2 text-xs font-medium">
          <Activity size={14} className={isOnline ? 'text-emerald-400' : 'text-red-400'} />
          <span className={isOnline ? 'text-emerald-300' : 'text-red-300'}>{isOnline ? 'Backend connected' : 'Backend offline'}</span>
          <span className={isOnline ? 'size-1.5 rounded-full bg-emerald-400 shadow-[0_0_10px_#10b981]' : 'size-1.5 rounded-full bg-red-400'} />
        </motion.div>
      </div>
    </header>
  );
}
