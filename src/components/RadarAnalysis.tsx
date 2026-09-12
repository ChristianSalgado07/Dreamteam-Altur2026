import { PolarAngleAxis, PolarGrid, PolarRadiusAxis, Radar, RadarChart, ResponsiveContainer, Legend } from 'recharts';
import type { Breakdown } from '../lib/types';
import { breakdownMeta, parsePercent } from '../lib/utils';

export function RadarAnalysis({ breakdown, confidence, synthetic }: { breakdown: Breakdown; confidence: number; synthetic: boolean }) {
  const data = breakdownMeta.map(([key, label]) => ({ subject: label.replace(' & ', ' / '), score: key === 'confidence' ? confidence * 100 : parsePercent(breakdown[key as keyof Breakdown]), baseline: 50 }));
  const color = synthetic ? '#ef4444' : '#10b981';
  return <div><p className="eyebrow">SIGNAL MAP</p><h3 className="mt-1 text-lg font-semibold text-white">Multi-dimensional Analysis</h3><div className="mt-2 h-[250px]" role="img" aria-label="Radar chart comparing analysis features against a human baseline"><ResponsiveContainer width="100%" height="100%"><RadarChart data={data} cx="50%" cy="50%" outerRadius="68%"><PolarGrid stroke="#27334a" /><PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 9 }} /><PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} /><Radar name="Human baseline" dataKey="baseline" stroke="#64748b" fill="none" strokeDasharray="4 4" /><Radar name="This call" dataKey="score" stroke={color} fill={color} fillOpacity={.22} strokeWidth={2} /><Legend wrapperStyle={{ fontSize: 10, color: '#94a3b8' }} /></RadarChart></ResponsiveContainer></div></div>;
}
