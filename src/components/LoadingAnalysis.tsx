import { motion } from 'framer-motion';
import { BrainCircuit, LoaderCircle } from 'lucide-react';

export function LoadingAnalysis() {
  return <div className="flex min-h-[520px] flex-col items-center justify-center text-center"><motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1.5, ease: 'linear' }} className="mb-5 text-blue-400"><LoaderCircle size={42} /></motion.div><BrainCircuit className="mb-3 text-violet-400" size={24} /><h3 className="text-lg font-semibold text-white">Analyzing acoustic patterns...</h3><p className="mt-2 max-w-xs text-sm text-slate-500">Running temporal, spectral, semantic and biological feature extraction.</p><div className="mt-6 h-1 w-48 overflow-hidden rounded-full bg-slate-800"><motion.div animate={{ x: ['-100%', '100%'] }} transition={{ repeat: Infinity, duration: 1.4 }} className="h-full w-1/2 bg-gradient-to-r from-blue-500 to-violet-500" /></div></div>;
}
