import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from 'react';
import { cn } from '../lib/utils';

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('rounded-2xl border border-line bg-surface/80 shadow-card', className)} {...props} />;
}

export function Button({ className, ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button className={cn('focus-ring rounded-xl transition-all disabled:cursor-not-allowed disabled:opacity-50', className)} {...props} />;
}

export function Badge({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cn('inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold', className)}>{children}</span>;
}
