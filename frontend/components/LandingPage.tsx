
import React from 'react';
import { View } from '../types';
import FeatureSlider, { FeatureCardProps } from './FeatureSlider';

interface LandingPageProps {
  onNavigate: (view: View) => void;
}

const signalDomains: FeatureCardProps[] = [
  {
    icon: 'face',
    title: 'Facial Biometry',
    desc: '468-pt landmarks, Micro-expressions & Jaw tension mapping.',
    category: 'Signal Domain'
  },
  {
    icon: 'visibility',
    title: 'Ocular Tracking',
    desc: 'Blink rate volatility, Gaze fixation & Saccade velocity.',
    category: 'Signal Domain'
  },
  {
    icon: 'mic',
    title: 'Vocal Mechanics',
    desc: 'Fundamental frequency, Pitch tremor & Latency analysis.',
    category: 'Signal Domain'
  },
  {
    icon: 'description',
    title: 'Linguistic Logic',
    desc: 'Policy vocabulary, Over-hedging & Structure clarity.',
    category: 'Signal Domain'
  },
  {
    icon: 'accessibility_new',
    title: 'Postural Analysis',
    desc: 'Shoulder elevation, Spine alignment & Fidget frequency.',
    category: 'Signal Domain'
  }
];

const intelligenceMetrics: FeatureCardProps[] = [
  {
    icon: 'verified_user',
    title: 'Integrity Index',
    desc: 'Cognitive load mismatch & Intra-answer consistency.',
    category: 'Intelligence'
  },
  {
    icon: 'speed',
    title: 'Stress Recovery',
    desc: 'Metric deviation recovery after high-difficulty topics.',
    category: 'Intelligence'
  },
  {
    icon: 'account_balance',
    title: 'Admin Maturity',
    desc: 'Ethical reasoning depth & Bureaucratic tone analysis.',
    category: 'Intelligence'
  },
  {
    icon: 'groups',
    title: 'Board Impression',
    desc: 'Aggregated prediction of candidate suitability scores.',
    category: 'Intelligence'
  },
  {
    icon: 'analytics',
    title: 'Forensic Memory',
    desc: 'Semantic event bundling for explainable decision logic.',
    category: 'Intelligence'
  }
];

const LandingPage: React.FC<LandingPageProps> = ({ onNavigate }) => {
  return (
    <div className="relative min-h-screen bg-black text-white overflow-hidden font-display selection:bg-white selection:text-black">
      {/* Background Layer */}
      <div className="fixed inset-0 z-0">
        <video
          className="absolute inset-0 w-full h-full object-cover opacity-70"
          autoPlay
          loop
          muted
          playsInline
        >
          <source src="/ostracized_remix.webm" type="video/webm" />
        </video>
        <div className="absolute inset-0 bg-black/40"></div>
      </div>

      <div className="relative z-10 container mx-auto px-6 lg:px-12 flex flex-col min-h-screen">
        {/* Navigation Spacer */}
        <div className="h-24" />

        <main className="flex-1 flex flex-col lg:flex-row items-center gap-12 lg:gap-20 py-12">
          {/* Left Column: Vision */}
          <div className="flex-1 space-y-10 text-left">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 backdrop-blur-md">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-[10px] uppercase font-bold tracking-[0.2em] text-slate-400">
                UPSC Performance Forensics v2.4
              </span>
            </div>

            <div className="space-y-6">
              <h1 className="text-6xl lg:text-8xl font-black leading-[0.9] orbitron tracking-tighter uppercase drop-shadow-[0_0_30px_rgba(255,255,255,0.1)]">
                Behavioral <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-white via-white to-slate-500">
                  Intelligence
                </span>
              </h1>
              <p className="max-w-xl text-slate-400 text-lg lg:text-xl font-medium leading-relaxed">
                The world's first Human Performance Forensics system specialized for UPSC interviews. Decode high-stakes cognitive and behavioral signals with precision.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row items-center gap-6 pt-4">
              <button
                onClick={() => onNavigate(View.DASHBOARD)}
                className="w-full sm:w-auto px-10 h-14 bg-white text-black font-black uppercase text-sm tracking-widest hover:bg-slate-200 transition-all hover:scale-105 shadow-[0_0_40px_rgba(255,255,255,0.2)] rounded-sm"
              >
                Launch Intelligence Engine
              </button>
              <button
                onClick={() => onNavigate(View.FEATURES)}
                className="w-full sm:w-auto px-10 h-14 bg-transparent text-white border border-white/20 font-black uppercase text-sm tracking-widest hover:bg-white/5 transition-all rounded-sm"
              >
                View Rubric
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-8 pt-8 border-t border-white/10">
              <div>
                <p className="text-2xl font-black text-white orbitron">24ms</p>
                <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Signal Latency</p>
              </div>
              <div>
                <p className="text-2xl font-black text-white orbitron">468pts</p>
                <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Mesh Density</p>
              </div>
              <div className="hidden sm:block">
                <p className="text-2xl font-black text-white orbitron">3-DOM</p>
                <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Cross-Correlation</p>
              </div>
            </div>
          </div>

          {/* Right Column: Visualizer */}
          <div className="w-full lg:w-[450px] h-[600px] flex gap-4 pointer-events-none select-none">
            <div className="flex-1 h-full">
              <FeatureSlider features={signalDomains} direction="down" speed={30} />
            </div>
            <div className="flex-1 h-full pt-16">
              <FeatureSlider features={intelligenceMetrics} direction="up" speed={35} />
            </div>
          </div>
        </main>
      </div>

      {/* Decorative Gradient Fog */}
      <div className="fixed bottom-0 left-0 w-full h-1/4 bg-gradient-to-t from-black to-transparent z-10 pointer-events-none"></div>
    </div>
  );
};

export default LandingPage;
