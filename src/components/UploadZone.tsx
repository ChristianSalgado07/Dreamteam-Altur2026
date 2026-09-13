import { FileAudio, FileCheck2, UploadCloud, X } from 'lucide-react';
import { motion } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { useState } from 'react';
import { Button, Card } from './ui';
import { formatBytes } from '../lib/utils';

interface AudioFile { file: File; duration: number; channels: number; }

export function UploadZone({ selected, onChange, onAnalyze, isLoading }: { selected: AudioFile | null; onChange: (value: AudioFile | null) => void; onAnalyze: () => void; isLoading: boolean }) {
  const [error, setError] = useState('');
  const onDrop = (files: File[]) => {
    const file = files[0];
    if (!file) return;
    const objectUrl = URL.createObjectURL(file);
    const audio = document.createElement('audio');
    audio.src = objectUrl;
    audio.onloadedmetadata = async () => {
      URL.revokeObjectURL(objectUrl);
      try {
        const context = new AudioContext();
        const buffer = await context.decodeAudioData(await file.arrayBuffer());
        await context.close();
        if (buffer.numberOfChannels < 2) {
          setError('El archivo debe ser WAV estéreo con dos canales.');
          return;
        }
        setError('');
        onChange({ file, duration: buffer.duration, channels: buffer.numberOfChannels });
      } catch {
        setError('No se pudo validar el formato WAV del archivo.');
      }
    };
    audio.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      setError('No se pudo leer el archivo de audio.');
    };
  };
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { 'audio/wav': ['.wav'] }, multiple: false });
  return <Card className="p-5 sm:p-6">
    <div className="mb-5 flex items-start justify-between">
      <div><p className="eyebrow">INPUT CHANNEL</p><h2 className="mt-1 text-xl font-semibold text-white">Upload Call Recording</h2></div>
      <FileAudio className="text-slate-500" size={21} />
    </div>
    <div {...getRootProps()} className={`focus-ring group cursor-pointer rounded-2xl border border-dashed p-7 text-center transition-all sm:p-10 ${isDragActive ? 'border-blue-400 bg-blue-500/10' : 'border-slate-700 bg-slate-950/25 hover:border-blue-400/60 hover:bg-blue-500/[0.04]'}`}>
      <input {...getInputProps()} aria-label="Upload WAV recording" />
      <motion.div animate={{ y: isDragActive ? -7 : [0, -3, 0] }} transition={{ repeat: isDragActive ? 0 : Infinity, duration: 2.8 }} className="mx-auto mb-4 grid size-14 place-items-center rounded-2xl bg-blue-500/10 text-blue-400 ring-1 ring-blue-400/20"><UploadCloud size={28} /></motion.div>
      <p className="font-medium text-slate-200">{isDragActive ? 'Drop to inspect recording' : 'Arrastra un archivo WAV aquí'}</p>
      <p className="mt-1 text-sm text-slate-500">o haz click para seleccionar · WAV estéreo</p>
      <div className="mt-5 flex justify-center gap-2 text-[10px] font-semibold uppercase tracking-widest text-slate-600"><span>8 kHz</span><span>•</span><span>2 channels</span><span>•</span><span>Secure client-side upload</span></div>
    </div>
    {error && <p className="mt-3 text-sm text-red-300">{error}</p>}
    {selected && <div className="mt-4 flex items-center justify-between rounded-xl border border-emerald-500/20 bg-emerald-500/[0.06] p-3">
      <div className="flex min-w-0 items-center gap-3"><FileCheck2 size={18} className="shrink-0 text-emerald-400" /><div className="min-w-0"><p className="truncate text-sm font-medium text-slate-200">{selected.file.name}</p><p className="text-xs text-slate-500">{formatBytes(selected.file.size)} · {selected.duration ? `${selected.duration.toFixed(1)} sec` : 'Stereo WAV'}</p></div></div>
      <Button aria-label="Remove selected file" onClick={(event) => { event.stopPropagation(); onChange(null); }} className="p-2 text-slate-500 hover:text-white"><X size={16} /></Button>
    </div>}
    <Button onClick={onAnalyze} disabled={!selected || isLoading} className="mt-5 flex w-full items-center justify-center gap-2 bg-gradient-to-r from-blue-600 to-violet-600 py-3.5 font-semibold text-white shadow-[0_0_24px_rgba(59,130,246,0.2)] hover:from-blue-500 hover:to-violet-500 hover:shadow-[0_0_30px_rgba(59,130,246,0.4)]">
      {isLoading ? <><span className="spinner" /> Analyzing acoustic patterns...</> : <>Analyze Call <span className="text-blue-200">→</span></>}
    </Button>
  </Card>;
}
