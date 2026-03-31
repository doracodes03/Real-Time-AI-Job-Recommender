import React, { useState } from 'react';
import axios from 'axios';
import { X, MapPin, Briefcase, Calendar, GraduationCap, Sparkles, Building2, Send, BrainCircuit, CheckCircle2, AlertCircle, ListChecks, ThumbsUp } from 'lucide-react';

const JobDetailModal = ({ job, resumeText, onClose, onApply }) => {
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  if (!job) return null;

  const handleExplain = async () => {
    if (!resumeText) {
      setError("Please upload a resume first to get an AI explanation.");
      return;
    }
    
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('job_id', job.id);
      formData.append('resume_text', resumeText);
      // Always send inline job fields so real-time (JSearch) jobs work too
      formData.append('job_title_inline', job['Job Title'] || '');
      formData.append('job_desc_inline', job['Job Description'] || '');
      formData.append('job_skills_inline', (job.skills || job.Skills || ''));
      
      const response = await axios.post(`http://localhost:8000/recommend/explain`, formData);
      setExplanation(response.data);
    } catch (err) {
      console.error(err);
      const detail = err?.response?.data?.detail;
      setError(detail || "Failed to generate AI explanation. Check your API key and connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-6">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-slate-900/60 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      ></div>

      {/* Modal Content */}
      <div className="relative bg-white w-full max-w-2xl max-h-[90vh] overflow-hidden rounded-3xl shadow-2xl flex flex-col animate-in fade-in zoom-in duration-200">
        
        {/* Header */}
        <div className="p-6 border-b border-slate-100 flex justify-between items-start gap-4">
          <div className="flex gap-4">
            <div className="w-14 h-14 bg-indigo-600 rounded-2xl flex items-center justify-center text-white shadow-lg shadow-indigo-100 shrink-0">
              <Briefcase size={28} />
            </div>
            <div>
              <h2 className="text-2xl font-black text-slate-900 leading-tight">{job['Job Title']}</h2>
              <div className="flex items-center gap-2 mt-1 text-slate-500 font-medium">
                <Building2 size={16} />
                <span>{job['Company'] || job['Company Name']}</span>
              </div>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-2 hover:bg-slate-100 rounded-xl transition-colors text-slate-400 hover:text-slate-600"
          >
            <X size={24} />
          </button>
        </div>

        {/* Body */}
        <div className="p-6 overflow-y-auto flex-1 custom-scrollbar">
          
          {/* Quick Info */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-8">
            <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
              <div className="text-slate-400 mb-1"><MapPin size={16} /></div>
              <div className="text-sm font-bold text-slate-900">{job.Location || 'Remote'}</div>
              <div className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">Location</div>
            </div>
            <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
              <div className="text-slate-400 mb-1"><Calendar size={16} /></div>
              <div className="text-sm font-bold text-slate-900">{job.Experience || '2-5'} Years</div>
              <div className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">Experience</div>
            </div>
            <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100">
              <div className="text-slate-400 mb-1"><GraduationCap size={16} /></div>
              <div className="text-sm font-bold text-slate-900">{job.Education || 'Degree'}</div>
              <div className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">Education</div>
            </div>
          </div>

          {/* Description */}
          <div className="mb-8">
            <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
              <Sparkles className="text-indigo-600" size={18} />
              Job Description
            </h3>
            <div className="text-slate-600 leading-relaxed text-sm whitespace-pre-wrap">
              {job['Job Description']}
            </div>
          </div>

          {/* Skills */}
          <div>
            <h3 className="text-lg font-bold text-slate-900 mb-4">Required Skills</h3>
            <div className="flex flex-wrap gap-2">
              {(job.skills || job.Skills)?.split(',').map((skill, i) => (
                <span key={i} className="px-4 py-2 bg-indigo-50 text-indigo-700 rounded-xl text-xs font-bold border border-indigo-100">
                  {skill.trim()}
                </span>
              ))}
            </div>
          </div>

          {/* Explainable AI Section */}
          <div className="mt-8 border-t border-slate-100 pt-8">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-black text-slate-900 flex items-center gap-2">
                <BrainCircuit className="text-indigo-600" size={24} />
                Explainable AI Insights
              </h3>
              {!explanation && !loading && (
                <button 
                  onClick={handleExplain}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-xl text-xs font-bold shadow-lg shadow-indigo-100 flex items-center gap-2 transition-all active:scale-95"
                >
                  <Sparkles size={14} />
                  Analyze Match
                </button>
              )}
            </div>

            {loading && (
              <div className="bg-slate-50 rounded-2xl p-8 border border-dashed border-slate-200 flex flex-col items-center justify-center gap-4 animate-in fade-in duration-300">
                <div className="w-12 h-12 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
                <p className="text-slate-500 font-bold text-sm">Gemini is analyzing your resume against this job...</p>
              </div>
            )}

            {error && (
              <div className="bg-rose-50 border border-rose-100 p-4 rounded-2xl flex items-center gap-3 text-rose-600 mb-6 text-sm font-medium">
                <AlertCircle size={18} />
                {error}
              </div>
            )}

            {explanation && (
              <div className="space-y-6 animate-in slide-in-from-bottom-4 duration-500">
                {/* Score & Match Level */}
                <div className="flex items-center gap-4 bg-indigo-50 p-4 rounded-2xl border border-indigo-100">
                  <div className="w-16 h-16 bg-white rounded-xl flex items-center justify-center shadow-sm">
                    <span className="text-2xl font-black text-indigo-600">{explanation.score}%</span>
                  </div>
                  <div>
                    <div className="text-sm font-black text-indigo-900 uppercase tracking-wider">AI Match Score</div>
                    <p className="text-xs text-indigo-600 font-medium leading-tight mt-0.5">
                      {explanation.score > 80 ? "This is a perfect match for your profile!" : 
                       explanation.score > 60 ? "Great match! Some minor skill gaps identified." : 
                       "Moderate match. Review suggestions to improve your odds."}
                    </p>
                  </div>
                </div>

                {/* Reason */}
                <div className="bg-white rounded-2xl border border-slate-100 p-5 shadow-sm">
                  <h4 className="flex items-center gap-2 text-slate-900 font-bold mb-3">
                    <ThumbsUp size={18} className="text-indigo-600" />
                    AI Reasoning
                  </h4>
                  <p className="text-slate-600 text-sm leading-relaxed italic">
                    "{explanation.reason}"
                  </p>
                </div>

                {/* Matched vs Missing */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="bg-emerald-50/50 rounded-2xl border border-emerald-100 p-5">
                    <h4 className="text-emerald-800 font-bold text-sm mb-4 flex items-center gap-2">
                      <CheckCircle2 size={16} /> Matched Skills
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {explanation.matched_skills.map((s, i) => (
                        <span key={i} className="px-3 py-1 bg-white text-emerald-700 rounded-lg text-[10px] font-bold border border-emerald-100 shadow-sm uppercase">
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="bg-amber-50/50 rounded-2xl border border-amber-100 p-5">
                    <h4 className="text-amber-800 font-bold text-sm mb-4 flex items-center gap-2">
                      <AlertCircle size={16} /> Missing Skills
                    </h4>
                    <div className="flex flex-wrap gap-2">
                      {explanation.missing_skills.map((s, i) => (
                        <span key={i} className="px-3 py-1 bg-white text-amber-700 rounded-lg text-[10px] font-bold border border-amber-100 shadow-sm uppercase">
                          {s}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Suggestions */}
                <div className="bg-indigo-600 rounded-2xl p-6 text-white shadow-xl shadow-indigo-100 relative overflow-hidden">
                  <Sparkles className="absolute -right-4 -bottom-4 text-indigo-500 opacity-30" size={100} />
                  <h4 className="text-lg font-bold mb-4 flex items-center gap-2 relative z-10">
                    <ListChecks size={20} /> How to Stand Out
                  </h4>
                  <ul className="space-y-3 relative z-10">
                    {explanation.suggestions.map((s, i) => (
                      <li key={i} className="flex gap-3 text-sm font-medium text-indigo-50">
                        <div className="shrink-0 w-5 h-5 bg-indigo-500/50 rounded-full flex items-center justify-center text-[10px] text-white font-bold">{i+1}</div>
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-slate-100 bg-slate-50 mt-auto">
          <button 
            onClick={() => {
              onApply(job.id);
              // Open job posting URL if available
              if (job.apply_url) {
                window.open(job.apply_url, '_blank');
              }
            }}
            className="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-black py-4 rounded-2xl transition-all shadow-xl shadow-indigo-100 flex items-center justify-center gap-2 group"
          >
            <Send size={18} className="group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />
            Apply For This Position
          </button>
        </div>
      </div>
    </div>
  );
};

export default JobDetailModal;
