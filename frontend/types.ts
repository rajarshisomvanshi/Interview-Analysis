
export enum View {
  LANDING = 'LANDING',
  FEATURES = 'FEATURES',
  DASHBOARD = 'DASHBOARD'
}

export interface Candidate {
  id: string;
  name: string;
  role: string;
  status: 'active' | 'completed' | 'scheduled' | 'processing' | 'failed';
  timeAgo?: string;
  avatar: string;
  integrityScore?: number;
  confidenceScore?: number;
  riskScore?: number;
  executiveSummary?: string;
  chatHistory?: Message[];
  slices?: TimeSlice[];
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  isThinking?: boolean;
}

export interface TimeSlice {
  start_ms: number;
  end_ms: number;
  insight: string;
  score: number;
  summary: string;
  scene_description?: string;
  dialogue?: string;
  behavioral_analysis?: string;
}

export interface AnalysisStatus {
  status: 'created' | 'processing' | 'completed' | 'failed';
  progress: number;
  currentStep: string;
  errorMessage?: string;
  slices?: TimeSlice[];
}
