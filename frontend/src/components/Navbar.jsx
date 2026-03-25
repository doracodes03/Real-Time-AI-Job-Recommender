import React from 'react';
import { Briefcase, Bell, Search, User } from 'lucide-react';

const Navbar = ({ userId }) => {
  return (
    <nav className="bg-white border-b border-slate-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center gap-2">
            <div className="bg-indigo-600 p-2 rounded-xl text-white">
              <Briefcase size={24} />
            </div>
            <span className="text-xl font-black text-slate-900 tracking-tight">JobAI</span>
          </div>
          
          <div className="hidden md:flex flex-1 justify-center max-w-lg mx-8">
            <div className="relative w-full">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
              <input 
                type="text" 
                placeholder="Search jobs, companies, skills..." 
                className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none transition-all"
              />
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button className="p-2 text-slate-400 hover:bg-slate-50 hover:text-indigo-600 rounded-xl transition-all">
              <Bell size={20} />
            </button>
            <div className="h-8 w-[1px] bg-slate-200 mx-2"></div>
            <div className="flex items-center gap-3 bg-slate-50 p-1.5 pr-4 rounded-xl border border-slate-200">
              <div className="h-8 w-8 bg-indigo-100 rounded-lg flex items-center justify-center text-indigo-600">
                <User size={18} />
              </div>
              <div className="hidden sm:block">
                <p className="text-xs font-bold text-slate-900">User ID</p>
                <p className="text-[10px] text-slate-500 font-medium">{userId}</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
