import React from 'react';
import { Sparkles, Users, TrendingUp, Globe } from 'lucide-react';

const RecommendationTabs = ({ activeTab, onTabChange }) => {
  const tabs = [
    { id: 'content', label: 'For You', icon: Sparkles, desc: 'AI-matched based on your skills' },
    { id: 'cf', label: 'Similar Users', icon: Users, desc: 'What similar users are looking at' },
    { id: 'hybrid', label: 'Hybrid', icon: TrendingUp, desc: 'The best of both worlds' },
    { id: 'realtime', label: 'Real-Time Jobs', icon: Globe, desc: 'Live jobs from top job boards' },
    { id: 'saved', label: 'Saved', icon: Users, desc: 'Jobs you have bookmarked' },
  ];

  return (
    <div className="mb-8">
      <div className="flex flex-col md:flex-row gap-4">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`flex-1 flex items-start gap-4 p-4 rounded-2xl border transition-all text-left ${
                isActive
                  ? 'bg-white border-indigo-200 shadow-md ring-1 ring-indigo-100'
                  : 'bg-slate-50 border-transparent hover:bg-white hover:border-slate-200'
              }`}
            >
              <div className={`p-2 rounded-xl ${isActive ? 'bg-indigo-600 text-white' : 'bg-slate-200 text-slate-500'}`}>
                <Icon size={20} />
              </div>
              <div>
                <h3 className={`font-bold ${isActive ? 'text-slate-900' : 'text-slate-500'}`}>{tab.label}</h3>
                <p className="text-xs text-slate-400 mt-1">{tab.desc}</p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default RecommendationTabs;
