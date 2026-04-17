import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import ResumeUpload from './components/ResumeUpload';
import RecommendationTabs from './components/RecommendationTabs';
import JobCard from './components/JobCard';
import JobDetailModal from './components/JobDetailModal';
import { Rocket, Sparkles, AlertCircle, Briefcase } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

function App() {
  const [userId] = useState("user_" + Math.random().toString(36).substr(2, 9));
  const [activeTab, setActiveTab] = useState('content');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastResume, setLastResume] = useState("");
  const [selectedJob, setSelectedJob] = useState(null);
  const [searchQuery, setSearchQuery] = useState("data scientist");
  const [searchLocation, setSearchLocation] = useState("United States");

  const fetchRecommendations = async (type, resumeText = lastResume) => {
    if (!resumeText && type !== 'cf' && type !== 'saved' && type !== 'realtime') return;
    
    setLoading(true);
    setError(null);
    try {
      let response;
      const formData = new FormData();
      formData.append('resume_text', resumeText);

      if (type === 'content') {
        // ⚡ Use fast endpoint for instant results
        response = await axios.post(`${API_BASE}/recommend/fast`, formData);
      } else if (type === 'cf') {
        response = await axios.get(`${API_BASE}/recommend/collaborative/${userId}`);
      } else if (type === 'hybrid') {
        response = await axios.post(`${API_BASE}/recommend/hybrid/${userId}`, formData);
      } else if (type === 'saved') {
        response = await axios.get(`${API_BASE}/recommend/saved/${userId}`);
      } else if (type === 'realtime') {
        formData.append('query', searchQuery);
        formData.append('location', searchLocation);
        response = await axios.post(`${API_BASE}/recommend/realtime/${userId}`, formData);
      }
      
      setRecommendations(response.data.recommendations);
    } catch (err) {
      console.error(err);
      if (err.response?.status === 400) {
        setError("Error: " + err.response.data.detail);
      } else {
        setError("Failed to fetch recommendations. Is the backend running?");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleResumeUpload = (text) => {
    setLastResume(text);
    fetchRecommendations(activeTab, text);
  };

  const handleInteraction = async (jobId, type) => {
    try {
      await axios.post(`${API_BASE}/interaction`, {
        user_id: userId,
        job_id: jobId,
        interaction_type: type
      });
      // Optionally refresh CF/Hybrid if on that tab
      if (activeTab !== 'content') {
        fetchRecommendations(activeTab);
      }
    } catch (err) {
      console.error("Interaction failed", err);
    }
  };

  useEffect(() => {
    if ((lastResume || activeTab === 'cf' || activeTab === 'saved') && activeTab !== 'realtime') {
      fetchRecommendations(activeTab);
    }
  }, [activeTab]);

  return (
    <div className="min-h-screen bg-slate-50 pb-20">
      <Navbar userId={userId} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Input */}
          <div className="lg:col-span-4">
            <div className="sticky top-24">
              <ResumeUpload onUpload={handleResumeUpload} loading={loading} />
              
              <div className="bg-indigo-600 rounded-2xl p-6 text-white overflow-hidden relative shadow-xl shadow-indigo-200">
                <Rocket className="absolute -right-4 -bottom-4 text-indigo-500 opacity-50 rotate-12" size={120} />
                <h3 className="text-xl font-bold mb-2 flex items-center gap-2">
                  <Sparkles size={18} />
                  AI Powered
                </h3>
                <p className="text-indigo-100 text-sm leading-relaxed relative z-10">
                  Our system combines Content-Based filtering with Collaborative behavior tracking to find your perfect job.
                </p>
                <button className="mt-4 bg-white text-indigo-600 font-bold py-2 px-4 rounded-xl text-sm hover:bg-slate-50 transition-colors relative z-10">
                  Learn how it works
                </button>
              </div>
            </div>
          </div>

          {/* Right Column: Results */}
          <div className="lg:col-span-8">
            <div className="flex items-end justify-between mb-8">
              <div>
                <h1 className="text-3xl font-black text-slate-900 tracking-tight">
                  Recommended Jobs
                </h1>
                <p className="text-slate-500 mt-1 font-medium">
                  Showing top matches based on your profile and behavior.
                </p>
              </div>
              <div className="text-right hidden sm:block">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-widest">
                  Sort By: Best Match
                </span>
              </div>
            </div>

            <RecommendationTabs activeTab={activeTab} onTabChange={setActiveTab} />

            {activeTab === 'realtime' && (
              <div className="bg-white rounded-2xl p-6 mb-8 border border-slate-200 shadow-sm">
                <h3 className="text-lg font-bold text-slate-900 mb-4">🌐 Real-Time Job Search</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-2">Job Title or Keyword</label>
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="e.g., Data Scientist, Product Manager"
                      className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-2">Location</label>
                    <input
                      type="text"
                      value={searchLocation}
                      onChange={(e) => setSearchLocation(e.target.value)}
                      placeholder="e.g., United States, Remote"
                      className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                    />
                  </div>
                  <div className="flex items-end">
                    <button
                      onClick={() => fetchRecommendations('realtime')}
                      disabled={!lastResume || loading}
                      className="w-full bg-indigo-600 text-white font-bold py-2 px-4 rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {loading ? 'Searching...' : 'Search Jobs'}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {error && (
              <div className="bg-rose-50 border border-rose-200 p-4 rounded-2xl flex items-center gap-3 text-rose-600 mb-8">
                <AlertCircle size={20} />
                <span className="font-medium text-sm">{error}</span>
              </div>
            )}

            {loading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} className="bg-white rounded-2xl p-6 h-64 border border-slate-200 animate-pulse">
                    <div className="flex gap-4 mb-4">
                      <div className="w-12 h-12 bg-slate-100 rounded-xl"></div>
                      <div className="flex-1 space-y-3">
                        <div className="h-4 bg-slate-100 rounded w-3/4"></div>
                        <div className="h-3 bg-slate-100 rounded w-1/2"></div>
                      </div>
                    </div>
                    <div className="h-24 bg-slate-50 rounded-xl mb-4"></div>
                    <div className="flex gap-3">
                      <div className="flex-1 h-10 bg-slate-100 rounded-xl"></div>
                      <div className="w-10 h-10 bg-slate-100 rounded-xl"></div>
                    </div>
                  </div>
                ))}
              </div>
            ) : recommendations.length > 0 ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {recommendations.map((job) => (
                  <JobCard 
                    key={job.id} 
                    job={job} 
                    onInteraction={handleInteraction}
                    onSelect={setSelectedJob}
                  />
                ))}
              </div>
            ) : (
              <div className="bg-white rounded-3xl p-16 text-center border-2 border-dashed border-slate-200">
                <div className="w-20 h-20 bg-slate-50 rounded-full flex items-center justify-center mx-auto mb-6 text-slate-300">
                  <Briefcase size={40} />
                </div>
                <h3 className="text-xl font-bold text-slate-900 mb-2">No recommendations yet</h3>
                <p className="text-slate-500 max-w-sm mx-auto">
                  {activeTab === 'cf' 
                    ? "Interact with some jobs to see what similar users are interested in."
                    : "Upload your resume in the left panel to see AI-powered job matches."}
                </p>
              </div>
            )}
          </div>
        </div>
      </main>

      {selectedJob && (
        <JobDetailModal 
          job={selectedJob} 
          resumeText={lastResume}
          onClose={() => setSelectedJob(null)}
          onApply={(jobId) => {
            handleInteraction(jobId, 'apply');
            setSelectedJob(null);
          }}
        />
      )}
    </div>
  );
}

export default App;
