import { useState } from 'react';
import { AlertCircle, LockKeyhole } from 'lucide-react';
import { Header } from '../components/Header';
import { UploadZone } from '../components/UploadZone';
import { ResultPanel } from '../components/ResultPanel';
import { HistoryTable } from '../components/HistoryTable';
import { useBackendStatus } from '../hooks/useBackendStatus';
import { useAnalysisHistory } from '../hooks/useAnalysisHistory';
import { detectVoice } from '../lib/api';
import type { AnalysisRecord } from '../lib/types';

interface AudioFile { file: File; duration: number; channels: number; }

export function Dashboard() {
  const { isOnline } = useBackendStatus();
  const { history, addRecord } = useAnalysisHistory();
  const [selected, setSelected] = useState<AudioFile | null>(null);
  const [result, setResult] = useState<AnalysisRecord | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function analyze() {
    if (!selected) return;
    setError('');
    setLoading(true);
    try {
      const encoded = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const value = String(reader.result);
          resolve(value.includes(',') ? value.split(',')[1] : value);
        };
        reader.onerror = () => reject(new Error('No se pudo leer el archivo.'));
        reader.readAsDataURL(selected.file);
      });
      const response = await detectVoice(encoded);
      const record: AnalysisRecord = { ...response, id: crypto.randomUUID(), filename: selected.file.name, timestamp: Date.now(), rawConfidence: response.confidence };
      setResult(record);
      addRecord(record);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'No se pudo completar el análisis.';
      setError(message.includes('400') ? 'El backend rechazó el audio. Confirma que sea WAV estéreo.' : message);
    } finally {
      setLoading(false);
    }
  }

  return <div className="min-h-screen bg-ink text-white"><Header isOnline={isOnline} alert={Boolean(result?.is_synthetic)} /><main className="mx-auto max-w-[1440px] px-5 py-8 sm:px-8 lg:py-10"><div className="mb-8 flex flex-col justify-between gap-4 sm:flex-row sm:items-end"><div><p className="eyebrow text-blue-400">ALTUR / HACKMTY 2026</p><h1 className="mt-2 max-w-2xl text-3xl font-bold tracking-tight text-white sm:text-4xl">Voice intelligence for every <span className="text-gradient">banking call.</span></h1><p className="mt-3 max-w-xl text-sm leading-relaxed text-slate-400">A live security cockpit for detecting synthetic voices before they become a fraud event.</p></div><div className="hidden items-center gap-2 rounded-lg border border-line bg-surface/50 px-3 py-2 text-xs text-slate-500 md:flex"><LockKeyhole size={14} className="text-blue-400" /> Encrypted analysis pipeline</div></div>{error && <div role="alert" className="mb-5 flex items-center gap-2 rounded-xl border border-red-500/25 bg-red-500/10 px-4 py-3 text-sm text-red-200"><AlertCircle size={17} />{error}<button onClick={() => setError('')} className="ml-auto text-red-300">Dismiss</button></div>}<div className="grid items-start gap-6 xl:grid-cols-[.78fr_1.22fr]"><UploadZone selected={selected} onChange={setSelected} onAnalyze={analyze} isLoading={loading} /><ResultPanel result={result} loading={loading} /></div><HistoryTable records={history} onSelect={setResult} /></main><footer className="mx-auto max-w-[1440px] px-5 pb-8 pt-2 text-center text-xs text-slate-600 sm:px-8">Dreamteam · HackMTY 2026 · Altur Voice Security</footer></div>;
}
