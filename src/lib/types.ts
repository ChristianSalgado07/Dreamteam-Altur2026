export type RecommendedAction = 'trigger_whatsapp_2fa' | 'proceed_call';

export interface Breakdown {
  timing_and_environment: string;
  caller_acoustics: string;
  agent_acoustics: string;
  semantics: string;
  biological_and_phase: string;
}

export interface DetectionResponse {
  is_synthetic: boolean;
  confidence: number;
  recommended_action: RecommendedAction;
  breakdown: Breakdown;
}

export interface AnalysisRecord extends DetectionResponse {
  id: string;
  filename: string;
  timestamp: number;
  rawConfidence: number;
}
