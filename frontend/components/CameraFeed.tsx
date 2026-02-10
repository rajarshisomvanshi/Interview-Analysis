
import React, { useRef, useEffect } from 'react';

const CameraFeed: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    async function setupCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Error accessing camera:", err);
      }
    }
    setupCamera();
    return () => {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="relative w-full h-full bg-black overflow-hidden group">
      <video 
        ref={videoRef} 
        autoPlay 
        muted 
        playsInline 
        className="w-full h-full object-cover grayscale opacity-60 group-hover:grayscale-0 group-hover:opacity-100 transition-all duration-700"
      />
      
      {/* HUD Overlay */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="scanline"></div>
        
        {/* Corner Brackets */}
        <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-primary/60"></div>
        <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-primary/60"></div>
        <div className="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 border-primary/60"></div>
        <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-primary/60"></div>

        {/* Dynamic Data Labels */}
        <div className="absolute top-6 left-14 flex flex-col gap-1">
          <span className="orbitron text-[8px] text-primary/80 uppercase font-bold tracking-widest">Tracking_ID: 9X-214</span>
          <span className="orbitron text-[8px] text-primary/80 uppercase font-bold tracking-widest">Face_Locked: TRUE</span>
        </div>

        <div className="absolute bottom-6 right-14 text-right">
          <span className="orbitron text-[8px] text-primary/80 uppercase font-bold tracking-widest">Telemetry_Sync: ACTIVE</span>
          <div className="flex gap-1 mt-1 justify-end">
             {[1,2,3,4,5].map(i => <div key={i} className="w-1 h-3 bg-primary/40 animate-pulse" style={{animationDelay: `${i*0.2}s`}}></div>)}
          </div>
        </div>

        {/* Simulated Facial Nodes */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 border border-white/10 rounded-full animate-ping opacity-20"></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 border border-primary/20 rounded-full animate-pulse"></div>
      </div>
    </div>
  );
};

export default CameraFeed;
