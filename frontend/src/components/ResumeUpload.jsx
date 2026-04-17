import React, { useState } from 'react';
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist/legacy/build/pdf';
import pdfjsWorker from 'pdfjs-dist/build/pdf.worker?worker';
GlobalWorkerOptions.workerSrc = pdfjsWorker;
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

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (file && file.type === 'application/pdf') {
      const reader = new FileReader();
      reader.onload = async function() {
        const typedarray = new Uint8Array(this.result);
        try {
          const pdf = await getDocument({ data: typedarray }).promise;
          let textContent = '';
          for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const txt = await page.getTextContent();
            textContent += txt.items.map((s) => s.str).join(' ') + '\n';
          }
          setText(textContent);
        } catch (err) {
          alert('Failed to parse PDF.');
        }
      };
      reader.readAsArrayBuffer(file);
    } else {
      alert('Please upload a valid PDF file.');
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
        <div className="mb-4">
          <label className="block mb-2 font-medium text-slate-700">Or upload PDF resume:</label>
          <input type="file" accept="application/pdf" onChange={handleFileChange} className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100" />
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
