
import React, { useState } from 'react';
import { View } from './types';
import LandingPage from './components/LandingPage';
import FeaturesPage from './components/FeaturesPage';
import Dashboard from './components/Dashboard';
import Navbar from './components/Navbar';

import MouseSpotlight from './components/MouseSpotlight';

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<View>(View.LANDING);

  const navigateTo = (view: View) => {
    setCurrentView(view);
    if (view === View.LANDING) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-black text-white selection:bg-white selection:text-black font-display">
      <MouseSpotlight />
      <Navbar currentView={currentView} onNavigate={navigateTo} />

      {currentView === View.DASHBOARD ? (
        <div className="pt-20">
          <Dashboard onNavigate={navigateTo} />
        </div>
      ) : (
        <>
          <LandingPage onNavigate={navigateTo} />
          <div id="features">
            <FeaturesPage onNavigate={navigateTo} />
          </div>
        </>
      )}
    </div>
  );
};

export default App;
