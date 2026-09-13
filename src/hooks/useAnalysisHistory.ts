import { useCallback, useEffect, useState } from 'react';
import type { AnalysisRecord } from '../lib/types';

const STORAGE_KEY = 'altur-analysis-history';

export function useAnalysisHistory() {
  const [history, setHistory] = useState<AnalysisRecord[]>([]);

  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        setHistory(JSON.parse(stored) as AnalysisRecord[]);
      } catch {
        localStorage.removeItem(STORAGE_KEY);
      }
    }
  }, []);

  const addRecord = useCallback((record: AnalysisRecord) => {
    setHistory((current) => {
      const next = [record, ...current].slice(0, 20);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  return { history, addRecord };
}
