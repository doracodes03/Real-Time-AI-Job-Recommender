import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Navbar from './components/Navbar';
import ResumeUpload from './components/ResumeUpload';
import RecommendationTabs from './components/RecommendationTabs';
import JobCard from './components/JobCard';
import JobDetailModal from './components/JobDetailModal';
import AuthModal from './components/AuthModal';
import { Rocket, Sparkles, AlertCircle, Briefcase, MapPin, SlidersHorizontal, ChevronLeft, ChevronRight } from 'lucide-react';

const API_BASE = 'http://localhost:8000';

function App() {
  const [token, setToken] = useState(localStorage.getItem('token') || '');
  const [user, setUser] = useState(localStorage.getItem('username') || '');
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('content');
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastResume, setLastResume] = useState("");
  const [selectedJob, setSelectedJob] = useState(null);
  const [searchQuery, setSearchQuery] = useState("data scientist");
  const [searchLocation, setSearchLocation] = useState("United States");
  
  // New Global Filters
  const [filterLocation, setFilterLocation] = useState("");
  const [filterExperience, setFilterExperience] = useState("");
  
  // Pagination State
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalResults, setTotalResults] = useState(0);

  const handleAuthSuccess = (newToken, username) => {
    setToken(newToken);
    setUser(username);
    localStorage.setItem('token', newToken);
    localStorage.setItem('username', username);
  };

  const handleLogout = () => {
    setToken('');
    setUser('');
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    setRecommendations([]);
  };

  const fetchRecommendations = async (type, resumeText = lastResume, pageToFetch = currentPage) => {
    if (!resumeText && type !== 'cf' && type !== 'saved' && type !== 'realtime') return;
    
    // Auth check for protected routes
    if (!token && type !== 'content') {
      setIsAuthModalOpen(true);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      let response;
      const headers = token ? { Authorization: `Bearer ${token}` } : {};
      const formData = new FormData();
      formData.append('resume_text', resumeText);
      formData.append('page', pageToFetch);
      formData.append('page_size', 10);

      if (type === 'content') {
        if (filterLocation) formData.append('location', filterLocation);
        if (filterExperience) formData.append('max_experience', filterExperience);
        response = await axios.post(`${API_BASE}/recommend/fast`, formData, { headers });
      } else if (type === 'cf') {
        response = await axios.get(`${API_BASE}/recommend/collaborative/${user || 'anon'}?page=${pageToFetch}&page_size=10`, { headers });
      } else if (type === 'hybrid') {
        if (filterLocation) formData.append('location', filterLocation);
        if (filterExperience) formData.append('max_experience', filterExperience);
        response = await axios.post(`${API_BASE}/recommend/hybrid/${user || 'anon'}`, formData, { headers });
      } else if (type === 'saved') {
        response = await axios.get(`${API_BASE}/recommend/saved/${user || 'anon'}`, { headers });
      } else if (type === 'realtime') {
        formData.append('query', searchQuery);
        formData.append('location', searchLocation);
        response = await axios.post(`${API_BASE}/recommend/realtime/${user || 'anon'}`, formData, { headers });
      }
      
      const payload = response.data;
      setRecommendations(payload.recommendations || []);
      
      // Update pagination metadata if returned by endpoint
      if (payload.total !== undefined) {
        setTotalResults(payload.total);
        setTotalPages(Math.ceil(payload.total / (payload.page_size || 10)));
      } else {
        setTotalResults(payload.recommendations?.length || 0);
        setTotalPages(1);
      }
      setCurrentPage(pageToFetch);

    } catch (err) {
      console.error(err);
      if (err.response?.status === 401) {
        setError("Please login to access this feature.");
        setIsAuthModalOpen(true);
      } else if (err.response?.status === 400) {
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
    setCurrentPage(1);
    fetchRecommendations(activeTab, text, 1);
  };

  const handleInteraction = async (jobId, type) => {
    if (!token) {
      setIsAuthModalOpen(true);
      return;
    }

    try {
      await axios.post(`${API_BASE}/interaction`, {
        job_id: jobId,
        interaction_type: type
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (activeTab !== 'content') {
        fetchRecommendations(activeTab);
      }
    } catch (err) {
      console.error("Interaction failed", err);
      if (err.response?.status === 401) {
        setIsAuthModalOpen(true);
      }
    }
  };

  useEffect(() => {
    setCurrentPage(1);
    if ((lastResume || activeTab === 'cf' || activeTab === 'saved') && activeTab !== 'realtime') {
      fetchRecommendations(activeTab, lastResume, 1);
    }
  }, [activeTab]);

  return (
    <div className="min-h-screen bg-slate-50 pb-20">
      <Navbar user={user} onLoginClick={() => setIsAuthModalOpen(true)} onLogout={handleLogout} />
      
      <AuthModal 
        isOpen={isAuthModalOpen} 
        onClose={() => setIsAuthModalOpen(false)} 
        onAuthSuccess={handleAuthSuccess}
        API_BASE={API_BASE}
      />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Input */}
          <div className="lg:col-span-4">
            <div className="sticky top-24">
              <ResumeUpload onUpload={handleResumeUpload} loading={loading} />
              
              {/* New Search Filters Panel */}
              <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 mt-6 mb-6">
                <h3 className="text-lg font-bold text-slate-900 mb-4 flex items-center gap-2">
                  <SlidersHorizontal size={20} className="text-indigo-600" />
                  Search Preferences
                </h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-1">
                      Target Location
                    </label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <MapPin size={16} className="text-slate-400" />
                      </div>
                      <input
                        type="text"
                        value={filterLocation}
                        onChange={(e) => setFilterLocation(e.target.value)}
                        placeholder="e.g. San Francisco, CA"
                        className="w-full pl-9 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors text-sm"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-semibold text-slate-700 mb-1">
                      Max Experience (Years)
                    </label>
                    <div className="relative">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <Briefcase size={16} className="text-slate-400" />
                      </div>
                      <input
                        type="number"
                        min="0"
                        step="0.5"
                        value={filterExperience}
                        onChange={(e) => setFilterExperience(e.target.value)}
                        placeholder="e.g. 3"
                        className="w-full pl-9 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors text-sm"
                      />
                    </div>
                  </div>
                  
                  <button 
                    onClick={() => { setCurrentPage(1); fetchRecommendations(activeTab, lastResume, 1); }}
                    disabled={!lastResume || loading}
                    className="w-full mt-4 bg-indigo-50 text-indigo-700 font-bold py-2 px-4 rounded-xl hover:bg-indigo-100 transition-colors text-sm flex justify-center items-center disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    Apply & Refresh
                  </button>
                </div>
              </div>

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
                      onClick={() => { setCurrentPage(1); fetchRecommendations('realtime', lastResume, 1); }}
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
              <>
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
                
                {totalPages > 1 && (
                  <div className="flex flex-col sm:flex-row justify-between items-center bg-white p-4 rounded-2xl border border-slate-200 mt-6 shadow-sm gap-4">
                    <button 
                      disabled={currentPage <= 1 || loading}
                      onClick={() => {
                        const newPage = currentPage - 1;
                        fetchRecommendations(activeTab, lastResume, newPage);
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                      }}
                      className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-xl transition-colors disabled:opacity-50"
                    >
                      <ChevronLeft size={16} /> Previous
                    </button>
                    <div className="text-sm font-medium text-slate-500">
                      Page <span className="text-indigo-600 font-bold">{currentPage}</span> of {totalPages} 
                      <span className="text-slate-300 mx-3">|</span> 
                      {totalResults} total matches
                    </div>
                    <button 
                      disabled={currentPage >= totalPages || loading}
                      onClick={() => {
                        const newPage = currentPage + 1;
                        fetchRecommendations(activeTab, lastResume, newPage);
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                      }}
                      className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-xl transition-colors disabled:opacity-50"
                    >
                      Next <ChevronRight size={16} />
                    </button>
                  </div>
                )}
              </>
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
