
import React from 'react';
import { View } from '../types';

interface FeaturesPageProps {
  onNavigate: (view: View) => void;
}

const FeaturesPage: React.FC<FeaturesPageProps> = ({ onNavigate }) => {
  const capabilities = [
    { icon: 'center_focus_strong', title: 'Object Detection', desc: 'Identifying candidates, environmental cues, and potential security threats with sub-pixel precision.' },
    { icon: 'psychology', title: 'Behavioral Analysis', desc: 'Tracking micro-expressions, body language patterns, and biometric stress indicators via video stream.' },
    { icon: 'description', title: 'Smart Reports', desc: 'Automated, AI-generated summary PDFs with actionable insights and confidence scores.' },
    { icon: 'bolt', title: 'Real-Time Processing', desc: 'Low-latency stream analysis powered by distributed edge computing nodes for instant feedback.' },
    { icon: 'hub', title: 'Multi-Modal Fusion', desc: 'Combining audio, video, and text data into a unified behavioral profile for deep context.' },
    { icon: 'bubble_chart', title: 'Visual Clustering', desc: 'Grouping similar behavioral patterns and anomalies across multiple sessions and candidates.' }
  ];

  return (
    <div className="relative min-h-screen pt-20 pb-20 bg-black">
      <div className="max-w-7xl mx-auto px-6 pt-12">
        <div className="mb-20 text-center">
          <div className="inline-flex items-center gap-2 mb-4 px-3 py-1 bg-white/5 rounded-full border border-white/20">
            <span className="text-white text-xs font-bold uppercase tracking-widest">Core Intelligence Layer</span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold leading-tight mb-6 text-white">
            System <span className="text-slate-500">Capabilities</span>
          </h2>
          <p className="max-w-2xl mx-auto text-slate-400 text-lg leading-relaxed font-light">
            Enterprise-grade AI analysis designed for clarity and precision. Processing visual telemetry in real-time.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {capabilities.map((item, idx) => (
            <div key={idx} className="bg-white/5 backdrop-blur-sm p-8 flex flex-col gap-6 hover:bg-white/10 transition-all rounded-2xl border border-white/10 hover:border-white/20 group cursor-default">
              <div className="w-12 h-12 bg-white flex items-center justify-center rounded-xl group-hover:scale-110 transition-all">
                <span className="material-symbols-outlined text-black text-2xl transition-colors">{item.icon}</span>
              </div>
              <div>
                <h3 className="text-lg font-bold mb-3 text-white">{item.title}</h3>
                <p className="text-slate-400 text-sm leading-relaxed">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-32 border border-white/10 relative overflow-hidden group rounded-2xl bg-zinc-900">
          <div className="absolute inset-0 z-0">
            <video
              className="w-full h-full object-cover opacity-50 grayscale transition-all duration-1000 group-hover:opacity-70 group-hover:grayscale-0"
              autoPlay
              loop
              muted
              playsInline
            >
              <source src="/frontend_video.mp4" type="video/mp4" />
            </video>
          </div>
          <div className="absolute inset-0 bg-gradient-to-t from-black via-black/40 to-transparent z-10"></div>

          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center z-20">
            <div className="p-8 bg-black/60 backdrop-blur-xl border border-white/10 max-w-lg rounded-2xl shadow-2xl">
              <h4 className="text-2xl font-bold mb-2 text-white orbitron">Ready to Analyze?</h4>
              <p className="text-sm text-slate-300 mb-6 font-medium">Connect your CCTV network or upload files to begin real-time behavioral monitoring.</p>
              <button
                onClick={() => onNavigate(View.DASHBOARD)}
                className="px-8 py-3 bg-white text-black font-extrabold rounded-full hover:bg-slate-200 transition-all hover:scale-105 shadow-[0_0_20px_rgba(255,255,255,0.3)] uppercase tracking-widest text-xs"
              >
                Start Analysis
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FeaturesPage;
