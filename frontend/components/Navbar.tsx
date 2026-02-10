import React from 'react';
import { View } from '../types';

interface NavbarProps {
    currentView: View;
    onNavigate: (view: View) => void;
}

const Navbar: React.FC<NavbarProps> = ({ currentView, onNavigate }) => {
    return (
        <nav className="glass-nav fixed top-0 w-full z-50">
            <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
                <div
                    className="flex items-center gap-3 cursor-pointer group"
                    onClick={() => onNavigate(View.LANDING)}
                >
                    <div className="w-10 h-10 bg-white flex items-center justify-center rounded-sm transition-transform group-hover:scale-105">
                        <span className="material-symbols-outlined text-black text-xl">psychology</span>
                    </div>
                    <h1 className="text-xl font-bold tracking-tighter text-white group-hover:text-slate-200 transition-colors orbitron">DEXTORA AI</h1>
                </div>

                <div className="hidden md:flex items-center gap-8">
                    <button
                        onClick={() => {
                            onNavigate(View.LANDING);
                            // Scroll to top if already on landing
                            if (currentView === View.LANDING) {
                                window.scrollTo({ top: 0, behavior: 'smooth' });
                            }
                        }}
                        className={`text-sm font-medium transition-colors ${currentView === View.LANDING ? 'text-white' : 'text-slate-400 hover:text-white'}`}
                    >
                        Home
                    </button>
                    <button
                        onClick={() => {
                            onNavigate(View.FEATURES); // In new layout, this might just scroll
                            const featuresSection = document.getElementById('features');
                            if (featuresSection) {
                                featuresSection.scrollIntoView({ behavior: 'smooth' });
                            }
                        }}
                        className="text-sm font-medium text-slate-400 hover:text-white transition-colors"
                    >
                        Features
                    </button>
                </div>

                <div>
                    <button
                        onClick={() => onNavigate(View.DASHBOARD)}
                        className="px-5 py-2.5 bg-white text-black hover:bg-slate-200 text-sm font-bold rounded-lg transition-all shadow-[0_0_15px_rgba(255,255,255,0.2)]"
                    >
                        {currentView === View.DASHBOARD ? 'Dashboard Active' : 'Login'}
                    </button>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
