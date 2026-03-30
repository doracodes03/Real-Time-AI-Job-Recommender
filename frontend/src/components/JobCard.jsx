import React from 'react';
import { Bookmark, Briefcase, MapPin, Star } from 'lucide-react';

const JobCard = ({ job, onInteraction, onSelect }) => {
  const score = job.final_score || job.cf_score || job.hybrid_score || 0;
  const displayScore = Math.round(score * 100);

  const handleApply = (e) => {
    e.stopPropagation();
    // Log interaction
    onInteraction(job.id, 'apply');
    // Open job posting URL in new tab
    if (job.apply_url) {
      window.open(job.apply_url, '_blank');
    }
  };

  return (
    <div 
      onClick={() => onSelect(job)}
      className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 hover:shadow-md transition-all group cursor-pointer hover:border-indigo-200 ring-0 hover:ring-2 hover:ring-indigo-50"
    >
      <div className="flex justify-between items-start mb-4">
        <div className="flex gap-4">
          <div className="w-12 h-12 bg-indigo-50 rounded-xl flex items-center justify-center text-indigo-600">
            <Briefcase size={24} />
          </div>
          <div>
            <h3 className="text-lg font-bold text-slate-900 group-hover:text-indigo-600 transition-colors">
              {job['Job Title']}
            </h3>
            <p className="text-slate-500 font-medium">{job['Company'] || job['Company Name']}</p>
          </div>
        </div>
        <div className="flex flex-col items-end">
          <div className="flex items-center gap-1 text-amber-500 font-bold bg-amber-50 px-2 py-1 rounded-lg text-sm">
            <Star size={14} fill="currentColor" />
            {displayScore}%
          </div>
        </div>
      </div>

      <div className="flex gap-3 mb-4 text-sm text-slate-500">
        <div className="flex items-center gap-1">
          <MapPin size={14} />
          {job.Location || 'Remote'}
        </div>
      </div>

      <div className="flex flex-wrap gap-2 mb-6">
        {(job.skills || job.Skills)?.split(',').slice(0, 3).map((skill, i) => (
          <span key={i} className="px-3 py-1 bg-slate-100 text-slate-600 rounded-full text-xs font-medium">
            {skill.trim()}
          </span>
        ))}
        {(job.skills || job.Skills)?.split(',').length > 3 && (
          <span className="px-3 py-1 bg-slate-50 text-slate-400 rounded-full text-xs font-medium">
            +{(job.skills || job.Skills).split(',').length - 3} more
          </span>
        )}
      </div>

      <div className="flex gap-3">
        <button 
          onClick={handleApply}
          className="flex-1 bg-indigo-600 text-white font-bold py-2.5 rounded-xl hover:bg-indigo-700 transition-colors shadow-lg shadow-indigo-100"
        >
          Apply Now
        </button>
        <button 
          onClick={(e) => { e.stopPropagation(); onInteraction(job.id, 'save'); }}
          className="p-2.5 text-slate-400 border border-slate-200 rounded-xl hover:bg-slate-50 hover:text-indigo-600 transition-all"
        >
          <Bookmark size={20} />
        </button>
      </div>
    </div>
  );
};

export default JobCard;
