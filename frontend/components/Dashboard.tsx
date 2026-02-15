
import React, { useState, useRef, useEffect } from 'react';
import { View, Candidate, Message } from '../types';
import { api, API_BASE_URL } from '../services/api';
import { generateSpeech, playBase64Audio } from '../services/geminiService';

interface DashboardProps {
  onNavigate: (view: View) => void;
}

const Dashboard: React.FC<DashboardProps> = ({ onNavigate }) => {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [activeCandidate, setActiveCandidate] = useState<Candidate | null>(null);
  const [videoSource, setVideoSource] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [analysisStatus, setAnalysisStatus] = useState<any>(null);
  const [currentSliceIndex, setCurrentSliceIndex] = useState(0);
  const [viewMode, setViewMode] = useState<'summary' | 'qa' | 'slices'>('summary');
  const [highlightedSlices, setHighlightedSlices] = useState<{ indices: number[], color: string } | null>(null);
  const [translatedSummary, setTranslatedSummary] = useState<string | null>(null);
  const [translatedSlices, setTranslatedSlices] = useState<Record<number, { insight: string, summary: string }>>({});
  const [isTranslating, setIsTranslating] = useState(false);
  const [expandedSlices, setExpandedSlices] = useState<Record<number, boolean>>({});
  const [expandedImprovements, setExpandedImprovements] = useState<Record<number, boolean>>({});
  const videoRef = useRef<HTMLVideoElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Load sessions on mount
  useEffect(() => {
    loadSessions();
  }, []);

  // Poll for status if processing
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (activeCandidate?.status === 'processing') {
      const pollStatus = async () => {
        try {
          const status = await api.getAnalysisStatus(activeCandidate.id);
          if (status) {
            setAnalysisStatus(status);

            // If completed, reload full data to get final scores
            if (status.status === 'completed' || status.status === 'failed') {
              clearInterval(interval);
              // Update the active candidate's status locally so the details fetcher triggers
              setActiveCandidate(prev => prev && prev.id === activeCandidate.id ? { ...prev, status: status.status } : prev);
              loadSessions(); // Reload sidebar

              // Fetch full details immediately for the active one
              fetchCandidateDetails(activeCandidate.id);
            }
          }
        } catch (e) {
          console.error("Polling error", e);
        }
      };

      pollStatus(); // Initial call
      interval = setInterval(pollStatus, 2000);
    } else {
      setAnalysisStatus(null);
    }

    return () => clearInterval(interval);
  }, [activeCandidate?.id, activeCandidate?.status]);

  // Update messages and video when active candidate changes
  useEffect(() => {
    if (activeCandidate) {
      // URL format: {API_BASE_URL}/videos/{id}/source_video.mp4
      // Use custom route that handles Range headers (seeking) correctly
      const videoUrl = `${API_BASE_URL}/videos/${activeCandidate.id}/source_video.mp4`;
      setVideoSource(videoUrl);

      // If we had chat history stored, we could load it here
      // For now, reset or keep simple
      setMessages([
        {
          id: 'init',
          role: 'assistant',
          content: `Analysis for ${activeCandidate.name} loaded. Ask me about behavioral integrity or risk factors.`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        }
      ]);
      setAnalysisStatus(null);
      setCurrentSliceIndex(0);

      if ((activeCandidate.status === 'completed' || activeCandidate.status === 'failed') && !activeCandidate.slices) {
        fetchCandidateDetails(activeCandidate.id);
      }
    }
  }, [activeCandidate?.id, activeCandidate?.status]);

  const loadSessions = async () => {
    try {
      setIsLoading(true);
      const data = await api.getSessions();
      setCandidates(data);
    } catch (e) {
      console.error("Failed to load sessions", e);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchCandidateDetails = async (id: string) => {
    try {
      const results = await api.getResults(id);
      if (results) {
        setCandidates(prev => prev.map(c => c.id === id ? {
          ...c,
          slices: results.slices,
          executiveSummary: results.executive_summary,
          qaPairs: results.qa_pairs,
          integrityScore: results.integrity_score,
          confidenceScore: results.confidence_score,
          riskScore: results.risk_score,
          mentalAlertnessScore: results.mental_alertness_score,
          criticalAssimilationScore: results.critical_assimilation_score,
          clearExpositionScore: results.clear_exposition_score,
          balanceJudgmentScore: results.balance_judgment_score,
          interestDepthScore: results.interest_depth_score,
          socialCohesionScore: results.social_cohesion_score,
          intellectualIntegrityScore: results.intellectual_integrity_score,
          stateAwarenessScore: results.state_awareness_score
        } : c));

        setActiveCandidate(prev => prev?.id === id ? {
          ...prev,
          slices: results.slices,
          executiveSummary: results.executive_summary,
          qaPairs: results.qa_pairs,
          integrityScore: results.integrity_score,
          confidenceScore: results.confidence_score,
          riskScore: results.risk_score,
          mentalAlertnessScore: results.mental_alertness_score,
          criticalAssimilationScore: results.critical_assimilation_score,
          clearExpositionScore: results.clear_exposition_score,
          balanceJudgmentScore: results.balance_judgment_score,
          interestDepthScore: results.interest_depth_score,
          socialCohesionScore: results.social_cohesion_score,
          intellectualIntegrityScore: results.intellectual_integrity_score,
          stateAwarenessScore: results.state_awareness_score
        } : prev);
      }
    } catch (e) {
      console.error("Failed to fetch details", e);
    }
  };

  const handleDeleteSession = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation(); // Don't trigger session selection
    if (!window.confirm('Are you sure you want to delete this session? This action cannot be undone.')) return;

    try {
      await api.deleteSession(id);
      setCandidates(prev => prev.filter(c => c.id !== id));
      if (activeCandidate?.id === id) {
        setActiveCandidate(null);
        setVideoSource(null);
      }
    } catch (e) {
      console.error("Delete failed", e);
      alert("Failed to delete session.");
    }
  };

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);



  const handleSendMessage = async (text: string, highlightColor: string = "") => {
    if ((!text && !input) || !activeCandidate) return;

    const messageContent = text || input;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: messageContent,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsSpeaking(true);

    const thinkingMsg: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: 'Accessing neural scores...',
      timestamp: '',
      isThinking: true
    };
    setMessages(prev => [...prev, thinkingMsg]);

    try {
      const responseText = await api.chat(activeCandidate.id, messageContent, messages);

      // Parse special tag from response for highlighting
      // Expected format: [[HIGHLIGHT: 1, 3, 5]]
      const highlightMatch = responseText.match(/\[\[HIGHLIGHT:\s*([\d,\s]+)\]\]/);
      if (highlightMatch && highlightColor) {
        const indices = highlightMatch[1].split(',').map(s => parseInt(s.trim()) - 1); // 0-indexed
        setHighlightedSlices({ indices, color: highlightColor });
      } else {
        setHighlightedSlices(null);
      }

      // Clean request from response text
      const cleanResponse = responseText.replace(/\[\[HIGHLIGHT:.*?\]\]/, '').trim();

      setMessages(prev => prev.filter(m => !m.isThinking).concat({
        id: (Date.now() + 2).toString(),
        role: 'assistant',
        content: cleanResponse,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }));
    } catch (e) {
      console.error("Chat error", e);
      setMessages(prev => prev.filter(m => !m.isThinking).concat({
        id: (Date.now() + 2).toString(),
        role: 'assistant',
        content: "Connection to analytic engine failed.",
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }));
    } finally {
      setIsSpeaking(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Show local preview immediately
    const url = URL.createObjectURL(file);
    setVideoSource(url);
    setIsUploading(true);

    try {
      // Upload to backend (starts analysis automatically)
      const { sessionId } = await api.uploadVideo(file, file.name.split('.')[0]);

      // Refresh list to get the new session
      const newCandidates = await api.getSessions();
      setCandidates(newCandidates);

      // Auto-select the new session
      const newSession = newCandidates.find(c => c.id === sessionId);
      if (newSession) {
        setActiveCandidate(newSession);
      }

    } catch (e) {
      console.error("Upload failed", e);
      alert(`Upload failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handlePlayAudioReport = async () => {
    if (!activeCandidate?.executiveSummary) return;
    setIsSpeaking(true);
    const audioData = await generateSpeech(activeCandidate.executiveSummary);
    if (audioData) {
      await playBase64Audio(audioData);
    }
    setIsSpeaking(false);
  };

  const handleTranslateSummary = async () => {
    if (!activeCandidate?.executiveSummary) return;
    try {
      setIsTranslating(true);
      const translated = await api.translateText(activeCandidate.executiveSummary, 'Hindi');
      setTranslatedSummary(translated);
    } catch (e) {
      console.error("Translation failed", e);
    } finally {
      setIsTranslating(false);
    }
  };

  const handleTranslateSlice = async (index: number, insight: string, summary: string) => {
    try {
      // Optimistic update or loading state could be added here
      const tInsight = await api.translateText(insight, 'Hindi');
      const tSummary = await api.translateText(summary, 'Hindi');

      setTranslatedSlices(prev => ({
        ...prev,
        [index]: { insight: tInsight, summary: tSummary }
      }));
    } catch (e) {
      console.error("Slice translation failed", e);
    }
  };

  const formatTime = (ms: number) => {
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const seekToSlice = (slice: any) => {
    if (videoRef.current) {
      videoRef.current.currentTime = slice.start_ms / 1000;
      videoRef.current.play();
    }
  };

  const renderQAList = () => {
    const qaPairs = activeCandidate?.qaPairs || []; // Use activeCandidate from props/state

    if (qaPairs.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center py-12 opacity-50">
          <span className="material-symbols-outlined text-4xl mb-2">forum</span>
          <p className="text-sm">No structured Q&A data available for this session.</p>
        </div>
      );
    }

    return (
      <div className="space-y-4 max-h-[500px] overflow-y-auto custom-scrollbar pr-2">
        {qaPairs.map((qa, idx) => (
          <div key={idx} className="bg-black/20 border border-white/5 rounded-xl p-5 hover:bg-black/40 transition-colors">
            <div className="flex flex-col gap-3">
              {/* Question */}
              <div className="flex gap-3">
                <div className="mt-1 w-5 h-5 rounded-full bg-indigo-500/20 text-indigo-400 flex items-center justify-center shrink-0 border border-indigo-500/30">
                  <span className="text-[10px] font-black">Q</span>
                </div>
                <p className="text-sm text-indigo-100/90 font-medium leading-relaxed">{qa.question_text}</p>
              </div>

              {/* Connecting Line */}
              <div className="ml-2.5 w-px h-2 bg-white/5"></div>

              {/* Answer */}
              <div className="flex gap-3">
                <div className="mt-1 w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0 border border-emerald-500/30">
                  <span className="text-[10px] font-black">A</span>
                </div>
                <p className="text-sm text-slate-300 leading-relaxed">{qa.response_text}</p>
              </div>
            </div>

            {/* Timestamp Footer */}
            <div className="mt-3 pt-3 border-t border-white/5 flex justify-end">
              <span className="text-[10px] text-slate-600 font-mono">
                {formatTime(qa.question_start_ms)} - {formatTime(qa.response_end_ms)}
              </span>
            </div>
          </div>
        ))}
      </div>
    );
  };

  const renderSliceIntelligence = () => {
    const slices = analysisStatus?.slices || activeCandidate?.slices || [];
    const currentSlice = slices[currentSliceIndex];

    if (!currentSlice) return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 text-center opacity-30">
        <span className="material-symbols-outlined text-4xl mb-4">analytics</span>
        <p className="text-[10px] uppercase font-bold tracking-widest text-slate-500">Awaiting Slice Data</p>
      </div>
    );

    return (
      <div className="flex flex-col h-full bg-black/40">
        <div className="p-4 border-b border-white/10 flex justify-between items-center shrink-0">
          <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] orbitron">Slice Intelligence</h4>
          <span className="text-[10px] text-white/50">{currentSliceIndex + 1} / {slices.length}</span>
        </div>

        <div className="flex-1 overflow-y-auto p-6 custom-scrollbar space-y-8">
          {/* Main Display Score */}
          {/* Main Display Score - REMOVED TEXT/NUMBER, KEPT BAR */}
          <div className="space-y-4">
            <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
              <div className={`h-full transition-all duration-500 ${currentSlice.score > 70 ? 'bg-emerald-500 shadow-[0_0_8px_#10b981]' : 'bg-red-500 shadow-[0_0_8px_#ef4444]'}`} style={{ width: `${currentSlice.score}%` }}></div>
            </div>
          </div>

          <div className="space-y-4 pt-4 border-t border-white/5">
            <div className="flex justify-between items-center text-[8px] font-black text-slate-500 uppercase tracking-widest orbitron">
              <span>{formatTime(currentSlice.start_ms)} Behavioral Insight</span>
              <button
                onClick={() => handleTranslateSlice(currentSliceIndex, currentSlice.insight, currentSlice.summary)}
                className="hover:text-white transition-colors flex items-center gap-1"
                title="Translate to Hindi"
              >
                <span className="material-symbols-outlined text-sm text-emerald-500">translate</span>
              </button>
            </div>
            <h5 className="text-sm font-bold text-white leading-snug">"{currentSlice.insight}"</h5>
            {translatedSlices[currentSliceIndex] && (
              <h5 className="text-sm font-bold text-emerald-400 leading-snug font-hindi mt-2">"{translatedSlices[currentSliceIndex].insight}"</h5>
            )}

            <div className="h-px w-8 bg-white/20"></div>

            {/* Behavioral Analysis Section */}
            <div>
              <p className="text-[8px] font-black text-slate-500 uppercase tracking-widest orbitron mb-2">Behavioral Analysis</p>
              <p className="text-xs text-slate-300 leading-relaxed">
                {currentSlice.behavioral_analysis || currentSlice.summary}
              </p>
            </div>

            {/* Collapsible Q&A (Transcription) Section */}
            <div className="pt-2">
              <button
                onClick={() => setExpandedSlices(prev => ({ ...prev, [currentSliceIndex]: !prev[currentSliceIndex] }))}
                className="flex items-center gap-2 text-[8px] font-black text-emerald-500 uppercase tracking-widest orbitron hover:text-emerald-400 transition-colors"
              >
                <span className="material-symbols-outlined text-[10px]">
                  {expandedSlices[currentSliceIndex] ? 'keyboard_arrow_up' : 'keyboard_arrow_down'}
                </span>
                OnA (Transcript)
              </button>

              {expandedSlices[currentSliceIndex] && (
                <div className="mt-4 p-4 bg-white/5 rounded-lg border border-white/5 animate-fade-in">
                  <p className="text-[10px] text-slate-400 leading-relaxed font-mono whitespace-pre-wrap">
                    {currentSlice.dialogue || "No transcript available for this segment."}
                  </p>
                </div>
              )}
            </div>
          </div>

          <button
            onClick={() => seekToSlice(currentSlice)}
            className="w-full py-4 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl text-[10px] font-black uppercase tracking-widest text-white transition-all flex items-center justify-center gap-2"
          >
            <span className="material-symbols-outlined text-sm">replay</span>
            Replay Segment
          </button>
        </div>
      </div >
    );
  };

  const renderHorizontalForensicGraph = () => {
    const slices = analysisStatus?.slices || activeCandidate?.slices || [];
    if (slices.length === 0) return null;

    return (
      <div className="bg-black/60 border border-white/10 rounded-2xl overflow-hidden shadow-2xl">
        <div className="px-4 py-2 border-b border-white/5 flex justify-between items-center">
          <h4 className="text-[10px] font-black text-slate-500 uppercase tracking-[0.2em] orbitron">Forensic Feature Synthesis</h4>
          <div className="flex items-center gap-3">
            {[
              { label: 'High Confidence (>70)', color: 'bg-emerald-500' },
              { label: 'Needs Attention (<70)', color: 'bg-red-500' }
            ].map(l => (
              <div key={l.label} className="flex items-center gap-1.5">
                <div className={`w-1.5 h-1.5 rounded-full ${l.color}`}></div>
                <span className="text-[7px] orbitron font-bold text-slate-400 uppercase tracking-widest">{l.label}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="flex gap-[1px] p-2 bg-black/40 h-8">
          {slices.map((slice: any, idx: number) => {
            const isActive = idx === currentSliceIndex;
            const scoreColor = slice.score > 70 ? 'bg-emerald-500' : 'bg-red-500';

            return (
              <div
                key={idx}
                onClick={() => {
                  setCurrentSliceIndex(idx);
                  seekToSlice(slice);
                }}
                className={`flex-1 h-full cursor-pointer transition-all relative group overflow-hidden first:rounded-l-lg last:rounded-r-lg ${isActive ? 'scale-y-125 z-10' : 'hover:scale-y-110'}`}
              >
                <div className={`w-full h-full ${scoreColor} ${isActive ? 'opacity-100' : 'opacity-40 group-hover:opacity-70'} transition-opacity`}></div>

                {isActive && (
                  <div className="absolute top-0 left-0 w-full h-full border-x border-white bg-white/10"></div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  };



  if (isLoading) {
    return (
      <div className="w-full h-screen bg-black flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="w-16 h-16 border-4 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin mx-auto"></div>
          <p className="text-emerald-500 font-bold tracking-widest animate-pulse">NOVA CORE INITIALIZING...</p>
        </div>
      </div>
    );
  }

  // Empty State
  if (!activeCandidate && candidates.length === 0) {
    return (
      <div className="flex h-[calc(100vh-5rem)] bg-black text-white overflow-hidden font-display">
        {/* Sidebar: Candidates (Empty) */}
        <aside className="w-72 flex flex-col border-r border-white/10 bg-black z-20">
          <div className="p-6 border-b border-white/10 flex items-center gap-3 cursor-pointer" onClick={() => onNavigate(View.LANDING)}>
            <div className="w-8 h-8 bg-white flex items-center justify-center rounded-sm text-black">
              <span className="material-symbols-outlined">psychology</span>
            </div>
            <div>
              <h1 className="orbitron text-sm font-bold text-white leading-none">DEXTORA AI</h1>
              <p className="text-[10px] text-slate-500 mt-1 uppercase tracking-widest">Analysis Engine v2.4</p>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto px-2 space-y-1 mt-4">
            <div className="px-3 py-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest">History</div>
            <div className="px-3 py-4 text-slate-600 text-[10px] uppercase font-bold tracking-widest">No history available</div>
          </div>
        </aside>

        <main className="flex-1 flex flex-col items-center justify-center bg-zinc-950 relative">
          <div className="glass-card p-12 text-center max-w-lg border border-white/10 bg-zinc-900/50 rounded-xl">
            <span className="material-symbols-outlined text-6xl text-slate-600 mb-6">analytics</span>
            <h2 className="text-2xl font-bold text-white mb-2 orbitron">Ready for Analysis</h2>
            <p className="text-slate-400 mb-8">Upload an interview recording to begin behavioral processing.</p>

            <label className="px-8 py-4 bg-white hover:bg-slate-200 text-black rounded-lg cursor-pointer text-sm font-bold transition-all shadow-lg hover:scale-105 inline-flex items-center gap-2">
              <span className="material-symbols-outlined">upload_file</span>
              {isUploading ? 'Starting Analysis...' : 'Upload & Analyze Video'}
              <input
                type="file"
                accept="video/*"
                className="hidden"
                onChange={handleFileUpload}
                disabled={isUploading}
              />
            </label>
          </div>
        </main>
      </div>
    );
  }

  const current = activeCandidate;

  return (
    <div className="flex h-[calc(100vh-5rem)] bg-black text-white overflow-hidden font-display">
      {/* Sidebar: Candidates */}
      <aside className="w-64 flex flex-col border-r border-white/10 bg-black z-20 shrink-0">
        <div className="p-6 border-b border-white/10 flex items-center gap-3 cursor-pointer" onClick={() => onNavigate(View.LANDING)}>
          <div className="w-8 h-8 bg-white flex items-center justify-center rounded-sm text-black">
            <span className="material-symbols-outlined">psychology</span>
          </div>
          <div>
            <h1 className="orbitron text-sm font-bold text-white leading-none">DEXTORA AI</h1>
            <p className="text-[10px] text-slate-500 mt-1 uppercase tracking-widest">Analysis Engine v2.4</p>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-2 space-y-1 mt-4">
          <div className="px-3 py-2 text-[10px] font-bold text-slate-500 uppercase tracking-widest">History</div>
          {candidates.map(c => (
            <div
              key={c.id}
              onClick={() => setActiveCandidate(c)}
              className={`group flex items-center gap-3 p-3 cursor-pointer transition-all border-l-2 ${activeCandidate?.id === c.id ? 'bg-white/10 border-white' : 'hover:bg-white/5 border-transparent'}`}
            >
              <div
                className={`w-8 h-8 rounded-sm bg-center bg-cover border ${activeCandidate?.id === c.id ? 'border-white/30' : 'border-white/10'}`}
                style={{ backgroundImage: `url('${c.avatar}')` }}
              />
              <div className="flex-1 min-w-0">
                <p className={`text-[11px] font-semibold truncate ${activeCandidate?.id === c.id ? 'text-white' : 'text-slate-400'}`}>{c.name}</p>
                <p className={`text-[9px] uppercase font-medium ${c.status === 'processing' ? 'text-orange-500 animate-pulse' : 'text-slate-500'}`}>
                  {c.status === 'processing' ? 'Analyzing...' : c.timeAgo}
                </p>
              </div>

              <button
                onClick={(e) => handleDeleteSession(e, c.id)}
                className="opacity-0 group-hover:opacity-100 p-1 hover:text-red-500 transition-all"
                title="Delete Session"
              >
                <span className="material-symbols-outlined text-sm">delete</span>
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* Main Analysis Area */}
      <main className="flex-1 flex flex-col min-w-0 bg-zinc-950 relative">
        <header className="h-16 border-b border-white/10 flex items-center justify-between px-8 bg-black/50 backdrop-blur-md z-10 shrink-0">
          <div className="flex items-center gap-4">
            <span className="orbitron text-[10px] text-black font-bold px-2 py-1 bg-white border border-white rounded-sm">{current ? 'ARCHIVE' : 'NEW SESSION'}</span>
            <h2 className="text-sm font-bold text-white flex items-center gap-2">
              {current ? `Analysis: ${current.name}` : 'Neural Analysis Dashboard'}
              {current && <span className="text-slate-500 font-normal text-xs">— {current.role}</span>}
            </h2>
            {current && (
              <button
                onClick={() => { setActiveCandidate(null); setVideoSource(null); }}
                className="ml-4 flex items-center gap-2 px-3 py-1.5 bg-white/5 hover:bg-white/10 text-white text-[10px] orbitron font-bold rounded-sm transition-all border border-white/10"
              >
                <span className="material-symbols-outlined text-sm">add_circle</span>
                START NEW SESSION
              </button>
            )}
          </div>
          {current && (
            <button className="flex items-center gap-2 px-3 py-1.5 bg-zinc-900 hover:bg-zinc-800 text-white text-[10px] font-semibold rounded-sm transition-colors border border-white/10">
              <span className="material-symbols-outlined text-sm">download</span>
              Export Results
            </button>
          )}
        </header>

        <div className="flex-1 overflow-y-auto p-8 space-y-8 custom-scrollbar">
          {!current ? (
            <div className="h-full flex flex-col items-center justify-center py-20">
              <div className="glass-card p-12 text-center max-w-lg border border-white/10 bg-zinc-900/40 rounded-3xl shadow-2xl relative overflow-hidden group">
                <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                <div className="relative z-10">
                  <div className="w-20 h-20 bg-white/5 rounded-2xl flex items-center justify-center mx-auto mb-8 border border-white/10 group-hover:scale-110 transition-transform duration-500">
                    <span className="material-symbols-outlined text-4xl text-white opacity-40 group-hover:opacity-100 transition-opacity">cloud_upload</span>
                  </div>
                  <h2 className="text-3xl font-black text-white mb-4 orbitron tracking-tight">Neural Ingestion</h2>
                  <p className="text-slate-400 mb-10 text-sm leading-relaxed">
                    Upload interview footage to begin temporal signal extraction and behavioral pattern analysis.
                  </p>

                  <label className="inline-flex items-center gap-3 px-10 py-5 bg-white hover:bg-zinc-200 text-black rounded-xl cursor-pointer text-xs orbitron font-black transition-all shadow-[0_0_30px_rgba(255,255,255,0.1)] hover:shadow-[0_0_40px_rgba(255,255,255,0.2)] hover:-translate-y-1 active:scale-95">
                    <span className="material-symbols-outlined text-lg">play_circle</span>
                    {isUploading ? 'SYNCHRONIZING...' : 'SELECT SOURCE VIDEO'}
                    <input
                      type="file"
                      accept="video/*"
                      className="hidden"
                      onChange={handleFileUpload}
                      disabled={isUploading}
                    />
                  </label>

                  {isUploading && (
                    <div className="mt-8 space-y-3">
                      <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
                        <div className="h-full bg-white animate-progress-indefinite"></div>
                      </div>
                      <p className="text-[10px] orbitron font-bold text-slate-500 animate-pulse tracking-widest uppercase">Initializing Forensic Stack...</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* Top Integrated Row: Slice Summary + Video */}
              <div className="flex flex-col lg:flex-row gap-8 min-h-[480px]">
                {/* Left Panel: Slice Intelligence */}
                <div className="lg:w-72 bg-zinc-900/40 border border-white/10 rounded-2xl overflow-hidden shadow-2xl shrink-0">
                  {renderSliceIntelligence()}
                </div>

                {/* Center Panel: Video Display */}
                <div className="flex-1 flex flex-col gap-6">
                  <div className="glass-card overflow-hidden h-[480px] relative bg-zinc-900 border border-white/10 rounded-2xl shadow-2xl bg-black">
                    {videoSource ? (
                      <video
                        ref={videoRef}
                        className="w-full h-full object-contain"
                        src={videoSource}
                        controls
                        autoPlay
                      />
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center text-slate-500 gap-4">
                        <span className="material-symbols-outlined text-4xl mb-2 opacity-20">video_library</span>
                        <p className="text-xs font-medium uppercase tracking-widest opacity-40 text-white">Select session footage</p>
                      </div>
                    )}
                  </div>

                  {/* Horizontal Forensic Graph (Below Video) */}
                  {renderHorizontalForensicGraph()}
                </div>
              </div>

              {/* Executive Summary (Persistent at Bottom) */}
              {current && (
                <section className="space-y-6">
                  {/* High-Level Score Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {/* Integrity Card */}
                    <div className="glass-card p-6 relative overflow-hidden group border border-white/10 bg-zinc-900/40 rounded-2xl shadow-xl">
                      {/* Tooltip Overlay */}
                      <div className="invisible group-hover:visible absolute inset-0 z-20 bg-black/95 backdrop-blur-xl p-6 flex flex-col justify-center transition-all duration-300 opacity-0 group-hover:opacity-100 rounded-2xl">
                        <p className="text-[10px] font-black orbitron text-emerald-500 mb-2 uppercase tracking-widest">Behavioral Integrity</p>
                        <p className="text-[10px] leading-relaxed text-slate-300">
                          Measures consistency between vocal stress and facial micro-expressions. Lower variance between verbal and non-verbal channels indicates higher truthfulness and data reliability.
                        </p>
                      </div>

                      <div className="flex justify-between items-center mb-2">
                        <p className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] orbitron">Integrity Score</p>
                        <span className="material-symbols-outlined text-emerald-500 text-sm">verified_user</span>
                      </div>
                      <h3 className="text-4xl font-black text-white leading-none mb-4">{current.integrityScore || '--'}%</h3>
                      <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                        <div className="h-full bg-emerald-500 shadow-[0_0_8px_#10b981]" style={{ width: `${current.integrityScore || 0}%` }}></div>
                      </div>
                    </div>

                    {/* Risk Card */}
                    <div className="glass-card p-6 relative overflow-hidden group border border-white/10 bg-zinc-900/40 rounded-2xl shadow-xl">
                      {/* Tooltip Overlay */}
                      <div className="invisible group-hover:visible absolute inset-0 z-20 bg-black/95 backdrop-blur-xl p-6 flex flex-col justify-center transition-all duration-300 opacity-0 group-hover:opacity-100 rounded-2xl">
                        <p className="text-[10px] font-black orbitron text-orange-500 mb-2 uppercase tracking-widest">Risk Assessment</p>
                        <p className="text-[10px] leading-relaxed text-slate-300">
                          Identifies deceptive markers like gaze aversion and speech disfluencies. High frequency of clusters associated with evasion elevates the risk level.
                        </p>
                      </div>

                      <div className="flex justify-between items-center mb-2">
                        <p className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] orbitron">Risk Level</p>
                        <span className="material-symbols-outlined text-orange-500 text-sm">warning</span>
                      </div>
                      <h3 className="text-4xl font-black text-white leading-none mb-4">{current.riskScore || '--'}%</h3>
                      <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                        <div className="h-full bg-orange-500 shadow-[0_0_8px_#f97316]" style={{ width: `${current.riskScore || 0}%` }}></div>
                      </div>
                    </div>

                    {/* Confidence Card */}
                    <div className="glass-card p-6 relative overflow-hidden group border border-white/10 bg-zinc-900/40 rounded-2xl shadow-xl">
                      {/* Tooltip Overlay */}
                      <div className="invisible group-hover:visible absolute inset-0 z-20 bg-black/95 backdrop-blur-xl p-6 flex flex-col justify-center transition-all duration-300 opacity-0 group-hover:opacity-100 rounded-2xl">
                        <p className="text-[10px] font-black orbitron text-blue-500 mb-2 uppercase tracking-widest">Neural Certainty</p>
                        <p className="text-[10px] leading-relaxed text-slate-300">
                          Represents AI certainty based on facial landmark precision and vocal telemetry clarity. Higher signal-to-noise ratio in data processing results in higher score.
                        </p>
                      </div>

                      <div className="flex justify-between items-center mb-2">
                        <p className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.2em] orbitron">Confidence</p>
                        <span className="material-symbols-outlined text-blue-500 text-sm">analytics</span>
                      </div>
                      <h3 className="text-4xl font-black text-white leading-none mb-4">{current.confidenceScore || '--'}%</h3>
                      <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                        <div className="h-full bg-blue-500 shadow-[0_0_8px_#3b82f6]" style={{ width: `${current.confidenceScore || 0}%` }}></div>
                      </div>
                    </div>
                  </div>

                  {/* UPSC Specific 8-Parameter Grid */}
                  <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mt-8">
                    {[
                      { label: 'Mental Alertness', key: 'mentalAlertnessScore', icon: 'psychology' },
                      { label: 'Critical Assimilation', key: 'criticalAssimilationScore', icon: 'account_tree' },
                      { label: 'Clear Exposition', key: 'clearExpositionScore', icon: 'record_voice_over' },
                      { label: 'Judgment Balance', key: 'balanceJudgmentScore', icon: 'balance' },
                      { label: 'Interest Depth', key: 'interestDepthScore', icon: 'search_insights' },
                      { label: 'Social Cohesion', key: 'socialCohesionScore', icon: 'groups' },
                      { label: 'Intellectual Integrity', key: 'intellectualIntegrityScore', icon: 'auto_awesome' },
                      { label: 'State Awareness', key: 'stateAwarenessScore', icon: 'public' }
                    ].map(param => (
                      <div key={param.key} className="glass-card p-4 border border-white/5 bg-zinc-900/20 rounded-xl">
                        <div className="flex justify-between items-center mb-1">
                          <p className="text-[7px] font-bold text-slate-500 uppercase tracking-widest orbitron">{param.label}</p>
                          <span className="material-symbols-outlined text-slate-500 text-[10px]">{param.icon}</span>
                        </div>
                        <div className="flex items-baseline gap-2">
                          <h4 className="text-xl font-black text-white leading-none">{(current as any)[param.key] || '--'}%</h4>
                          <div className="flex-1 h-0.5 bg-white/5 rounded-full overflow-hidden">
                            <div className="h-full bg-slate-400" style={{ width: `${(current as any)[param.key] || 0}%` }}></div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="glass-card p-10 border border-white/10 bg-zinc-900/40 rounded-2xl shadow-xl relative overflow-hidden group">
                    <div className="absolute top-0 left-0 w-1 h-full bg-white opacity-20 group-hover:opacity-100 transition-opacity"></div>
                    <div className="flex justify-between items-center mb-8">
                      <div className="flex items-center gap-6">
                        <h4 className="orbitron text-xs font-black text-white tracking-[0.3em] uppercase">Executive Intelligence Report</h4>

                        {/* View Toggles */}
                        <div className="flex bg-black/40 rounded-lg p-1 border border-white/5">
                          <button
                            onClick={() => setViewMode('summary')}
                            className={`px-3 py-1 rounded-md text-[10px] uppercase font-bold transition-all ${viewMode === 'summary' ? 'bg-white text-black shadow-lg' : 'text-slate-500 hover:text-white'}`}
                          >
                            Summary
                          </button>
                          <button
                            onClick={() => setViewMode('qa')}
                            className={`px-3 py-1 rounded-md text-[10px] uppercase font-bold transition-all ${viewMode === 'qa' ? 'bg-white text-black shadow-lg' : 'text-slate-500 hover:text-white'}`}
                          >
                            Q & A
                          </button>
                          <button
                            onClick={() => setViewMode('slices')}
                            className={`px-3 py-1 rounded-md text-[10px] uppercase font-bold transition-all ${viewMode === 'slices' ? 'bg-white text-black shadow-lg' : 'text-slate-500 hover:text-white'}`}
                          >
                            Slice Analysis
                          </button>
                        </div>
                      </div>

                      <button
                        onClick={handleTranslateSummary}
                        disabled={isTranslating || !current.executiveSummary}
                        className="flex items-center gap-3 px-4 py-2 bg-white/5 hover:bg-white/10 text-[10px] orbitron text-white rounded-full transition-all border border-white/10 disabled:opacity-50"
                      >
                        <span className="material-symbols-outlined text-sm">translate</span>
                        {isTranslating ? 'TRANSLATING...' : 'TRANSLATE TO HINDI'}
                      </button>
                    </div>

                    {viewMode === 'summary' ? (
                      <div className="space-y-4">
                        <p className="text-base leading-[1.8] text-slate-300 font-light fade-in">
                          {current.executiveSummary || 'Awaiting final synthesis of temporal data segments...'}
                        </p>
                        {translatedSummary && (
                          <div className="mt-4 pt-4 border-t border-white/10">
                            <p className="text-[10px] text-emerald-500 font-bold mb-2 uppercase tracking-widest">HINDI TRANSLATION</p>
                            <p className="text-base leading-[1.8] text-slate-300 font-light fade-in font-hindi">
                              {translatedSummary}
                            </p>
                          </div>
                        )}
                      </div>
                    ) : viewMode === 'qa' ? (
                      <div className="fade-in">
                        {renderQAList()}
                      </div>
                    ) : (
                      <div className="fade-in space-y-4 max-h-[500px] overflow-y-auto custom-scrollbar pr-2">
                        {(current.slices || []).map((slice: any, idx: number) => (
                          <div key={idx} className="bg-black/20 border border-white/5 rounded-xl p-5 hover:bg-black/40 transition-colors cursor-pointer" onClick={() => { setCurrentSliceIndex(idx); seekToSlice(slice); }}>
                            <div className="flex justify-between items-start mb-2">
                              <div className="flex items-center gap-2">
                                <span className="text-[10px] font-black orbitron text-emerald-500">SEGMENT {idx + 1}</span>
                                <span className="text-[10px] text-slate-600 font-mono">{formatTime(slice.start_ms)} - {formatTime(slice.end_ms)}</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <div className={`w-1.5 h-1.5 rounded-full ${slice.score > 70 ? 'bg-emerald-500' : 'bg-red-500'}`}></div>
                                <span className="text-[10px] font-bold text-slate-400">Score: {slice.score}</span>
                              </div>
                            </div>
                            <h4 className="text-sm font-bold text-white mb-2">"{slice.insight}"</h4>
                            <p className="text-xs text-slate-400 leading-relaxed italic">{slice.summary}</p>
                            {translatedSlices[idx] && (
                              <div className="mt-3 pt-3 border-t border-white/5">
                                <p className="text-[10px] text-emerald-500 font-bold mb-1 uppercase tracking-widest">HINDI</p>
                                <h4 className="text-sm font-bold text-emerald-400 font-hindi mb-1">"{translatedSlices[idx].insight}"</h4>
                                <p className="text-xs text-slate-400 italic font-hindi">{translatedSlices[idx].summary}</p>
                              </div>
                            )}
                          </div>
                        ))}
                        {(!current.slices || current.slices.length === 0) && (
                          <div className="text-center py-12 opacity-50">
                            <span className="material-symbols-outlined text-4xl mb-2">analytics</span>
                            <p className="text-sm">No slice data available yet.</p>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </section>
              )}
            </>
          )}
        </div>
      </main>

      {/* AI Chat Sidebar */}
      <aside className="w-80 flex flex-col border-l border-white/10 bg-black/95 backdrop-blur-xl shrink-0">
        <div className="p-6 border-b border-white/10 flex justify-between items-center">
          <h4 className="orbitron text-[10px] font-black text-white tracking-widest flex items-center gap-2">
            <span className="material-symbols-outlined text-white text-base">auto_awesome</span>
            FORENSIC CO-PILOT
          </h4>
          <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse shadow-[0_0_10px_#10b981]"></span>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">


          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[85%] p-4 rounded-2xl text-xs leading-relaxed ${msg.role === 'user'
                ? 'bg-white text-black font-medium'
                : 'bg-white/5 text-slate-300 border border-white/10'
                }`}>
                {msg.content}
                {msg.isThinking && <span className="inline-block w-1.5 h-3 ml-1 bg-emerald-500 animate-pulse col-span-2"></span>}
              </div>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        <div className="p-6 border-t border-white/10 bg-black">
          <div className="relative group">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(''); } }}
              className="w-full bg-zinc-900 border border-white/5 text-[11px] p-4 pr-12 focus:ring-1 focus:ring-white/20 focus:border-white/20 rounded-xl h-24 resize-none transition-all placeholder:text-slate-700"
              placeholder="Query neural patterns..."
            />
            <button
              onClick={() => handleSendMessage('')}
              className="absolute bottom-4 right-4 text-white/40 hover:text-white transition-colors"
            >
              <span className="material-symbols-outlined">send</span>
            </button>
          </div>
        </div>
      </aside>
    </div>
  );
};

export default Dashboard;

