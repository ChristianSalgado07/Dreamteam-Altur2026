import { useCallback, useEffect, useState } from 'react';
import { checkBackend } from '../lib/api';

export function useBackendStatus() {
  const [isOnline, setIsOnline] = useState(false);
  const ping = useCallback(async () => {
    try {
      await checkBackend();
      setIsOnline(true);
    } catch {
      setIsOnline(false);
    }
  }, []);

  useEffect(() => {
    void ping();
    const timer = window.setInterval(() => void ping(), 10_000);
    return () => window.clearInterval(timer);
  }, [ping]);

  return { isOnline };
}
