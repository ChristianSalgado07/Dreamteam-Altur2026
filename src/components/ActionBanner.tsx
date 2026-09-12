import { ArrowRight, MessageCircleWarning, ShieldCheck } from 'lucide-react';
import { motion } from 'framer-motion';

export function ActionBanner({ synthetic }: { synthetic: boolean }) {
  return <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className={`flex gap-3 rounded-xl border p-4 ${synthetic ? 'border-red-500/25 bg-red-500/[0.08]' : 'border-emerald-500/25 bg-emerald-500/[0.08]'}`}>
    {synthetic ? <MessageCircleWarning className="mt-0.5 shrink-0 text-red-400" size={20} /> : <ShieldCheck className="mt-0.5 shrink-0 text-emerald-400" size={20} />}
    <div className="min-w-0"><p className={`text-sm font-semibold ${synthetic ? 'text-red-200' : 'text-emerald-200'}`}>{synthetic ? 'Recommended Action: Trigger WhatsApp 2FA Verification' : 'Recommended Action: Proceed with Call'}</p><p className="mt-1 text-xs leading-relaxed text-slate-400">{synthetic ? "The system identified synthetic patterns. Sending security challenge to the customer's registered WhatsApp." : 'Voice verified as human. No additional verification required.'}</p></div><ArrowRight className="ml-auto shrink-0 text-slate-600" size={16} />
  </motion.div>;
}
