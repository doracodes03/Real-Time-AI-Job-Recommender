import React, { useState } from 'react';
import { Upload, FileText, CheckCircle2, Loader2 } from 'lucide-react';

const ResumeUpload = ({ onUpload, loading }) => {
  const [text, setText] = useState('');
  const [success, setSuccess] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim()) {
      onUpload(text);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    }
  };

  return (
    <div className="bg-white rounded-2xl p-8 shadow-sm border border-slate-200 mb-8">
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-indigo-50 text-indigo-600 rounded-lg">
          <Upload size={20} />
        </div>
        <h2 className="text-xl font-bold text-slate-900">Upload Resume</h2>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="relative mb-4">
          <textarea
            className="w-full h-40 p-4 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all outline-none resize-none text-slate-700 placeholder:text-slate-400"
            placeholder="Paste your resume text here to get personalized recommendations..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="absolute bottom-4 right-4 text-slate-400 pointer-events-none">
            <FileText size={20} />
          </div>
        </div>

        <button
          type="submit"
          disabled={loading || !text.trim()}
          className={`w-full py-3 rounded-xl font-bold flex items-center justify-center gap-2 transition-all ${
            loading 
              ? 'bg-slate-100 text-slate-400 cursor-not-allowed' 
              : success 
                ? 'bg-emerald-500 text-white'
                : 'bg-indigo-600 text-white hover:bg-indigo-700 shadow-lg shadow-indigo-100'
          }`}
        >
          {loading ? (
            <>
              <Loader2 className="animate-spin" size={20} />
              Analyzing Resume...
            </>
          ) : success ? (
            <>
              <CheckCircle2 size={20} />
              Recommendations Updated!
            </>
          ) : (
            'Get Recommendations'
          )}
        </button>
      </form>
    </div>
  );
};

export default ResumeUpload;
