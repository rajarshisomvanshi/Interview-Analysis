
import React from 'react';

export interface FeatureCardProps {
    icon: string;
    title: string;
    desc: string;
    category?: string;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ icon, title, desc, category }) => (
    <div className="w-full mb-6 p-4 glass-card bg-white/5 border border-white/10 rounded-xl backdrop-blur-xl hover:bg-white/10 transition-all group">
        <div className="flex items-start gap-4">
            <div className="p-2 bg-white/10 rounded-lg group-hover:scale-110 transition-transform">
                <span className="material-symbols-outlined text-white text-xl">{icon}</span>
            </div>
            <div className="flex-1 min-w-0">
                {category && (
                    <span className="text-[10px] uppercase font-bold text-slate-500 tracking-widest block mb-1">
                        {category}
                    </span>
                )}
                <h4 className="text-sm font-bold text-white mb-1 orbitron truncate">{title}</h4>
                <p className="text-[11px] text-slate-400 leading-relaxed font-medium">{desc}</p>
            </div>
        </div>
    </div>
);

interface FeatureSliderProps {
    features: FeatureCardProps[];
    direction?: 'up' | 'down';
    speed?: number; // Duration in seconds
}

const FeatureSlider: React.FC<FeatureSliderProps> = ({ features, direction = 'down', speed = 40 }) => {
    // Triple the features to ensure seamless loop
    const totalFeatures = [...features, ...features, ...features];

    const animationName = direction === 'down' ? 'scroll-down' : 'scroll-up';

    return (
        <div className="h-full overflow-hidden relative mask-fade-vertical">
            <div
                className="flex flex-col"
                style={{
                    animation: `${animationName} ${speed}s linear infinite`,
                    willChange: 'transform'
                }}
            >
                {totalFeatures.map((f, i) => (
                    <FeatureCard key={`${f.title}-${i}`} {...f} />
                ))}
            </div>

            <style jsx>{`
        .mask-fade-vertical {
          mask-image: linear-gradient(to bottom, transparent, black 10%, black 90%, transparent);
        }
        @keyframes scroll-down {
          0% { transform: translateY(-33.33%); }
          100% { transform: translateY(0); }
        }
        @keyframes scroll-up {
          0% { transform: translateY(0); }
          100% { transform: translateY(-33.33%); }
        }
      `}</style>
        </div>
    );
};

export default FeatureSlider;
