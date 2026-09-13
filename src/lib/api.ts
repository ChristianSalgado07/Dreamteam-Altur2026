import axios from 'axios';
import type { DetectionResponse } from './types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 120_000,
});

export async function checkBackend() {
  const response = await api.get('/');
  return response.data as { status: string };
}

export async function detectVoice(audioBase64: string) {
  const response = await api.post<DetectionResponse>('/detect', { audio_base64: audioBase64 });
  return response.data;
}
