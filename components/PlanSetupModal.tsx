import React, { useState, useEffect } from 'react';
import { 
  X, 
  Layers, 
  Calendar, 
  ArrowRight, 
  Target, 
  CheckCircle, 
  Network
} from 'lucide-react';
import { PlanningContext } from '../types';
import { api } from '../services/api';

interface PlanSetupModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (context: PlanningContext) => void;
}

export const PlanSetupModal: React.FC<PlanSetupModalProps> = ({ isOpen, onClose, onSubmit }) => {
  // --- STATE ---
  const [levels, setLevels] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  // Form Configuration
  const [aggregateLevel, setAggregateLevel] = useState<string>('');
  const [anchorLevel, setAnchorLevel] = useState<string>('');
  
  const [startYear, setStartYear] = useState<number>(new Date().getFullYear());
  const [horizonYears, setHorizonYears] = useState<number>(5);
  const [timeBucket, setTimeBucket] = useState<'Week' | 'Month'>('Month');

  // --- INITIALIZATION ---
  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      // Fetch the Enterprise Ontology defined in the Wizard
      api.ontology.getStructure('PRODUCT')
        .then(data => {
            const hierarchy = data && data.length > 0 ? data : ['Division', 'Department', 'Category', 'SKU'];
            setLevels(hierarchy);
            
            // Smart Defaults
            if (hierarchy.length > 0) {
                setAggregateLevel(hierarchy[0]); // Highest level (e.g. Division)
                // Default anchor to 2 levels down or bottom
                setAnchorLevel(hierarchy.length > 2 ? hierarchy[2] : hierarchy[hierarchy.length - 1]);
            }
        })
        .catch(err => {
            console.error("Failed to load ontology", err);
            // Fallback
            setLevels(['Division', 'Category', 'SKU']);
            setAggregateLevel('Division');
            setAnchorLevel('Category');
        })
        .finally(() => setLoading(false));
    }
  }, [isOpen]);

  // --- HANDLER FIX ---
  const handleSubmit = () => {
    // 🛑 CRITICAL FIX: Removed onClose(). 
    // We ONLY fire onSubmit. The App.tsx handler will close the modal 
    // AFTER updating the context state, preventing the redirect loop.
    onSubmit({
        aggregateLevel,
        anchorLevel,
        startYear,
        horizonYears,
        timeBucket
    });
  };

  // Validation: Anchor must be deeper than Aggregate
  const aggIndex = levels.indexOf(aggregateLevel);
  const anchorIndex = levels.indexOf(anchorLevel);
  const isValid = anchorIndex > aggIndex;

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-2xl overflow-hidden border border-slate-200 flex flex-col max-h-[90vh]">
        
        {/* HEADER */}
        <div className="p-6 bg-slate-50 border-b border-slate-100 flex justify-between items-start">
            <div className="flex gap-4">
                <div className="p-3 bg-indigo-600 text-white rounded-xl shadow-lg shadow-indigo-200">
                    <Target size={24} />
                </div>
                <div>
                    <h2 className="text-xl font-bold text-slate-900">Initialize Planning Session</h2>
                    <p className="text-sm text-slate-500 mt-1">Configure the dimensions for your financial simulation.</p>
                </div>
            </div>
            <button onClick={onClose} className="text-slate-400 hover:text-slate-600 transition-colors">
                <X size={24} />
            </button>
        </div>

        {/* BODY */}
        <div className="p-8 overflow-y-auto custom-scrollbar space-y-8">
            
            {/* 1. SCOPE DEFINITION */}
            <section>
                <div className="flex items-center gap-2 mb-4">
                    <Layers className="text-indigo-600" size={18}/>
                    <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">Hierarchy & Scope</h3>
                </div>
                
                <div className="grid grid-cols-2 gap-6">
                    <div className="space-y-2">
                        <label className="text-xs font-bold text-slate-500">Group Data By (Rollup)</label>
                        <div className="relative">
                            <select 
                                value={aggregateLevel}
                                onChange={(e) => setAggregateLevel(e.target.value)}
                                className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl text-sm font-bold text-slate-700 appearance-none outline-none focus:ring-2 focus:ring-indigo-500"
                            >
                                {levels.map((l, i) => (
                                    <option key={l} value={l} disabled={i >= levels.length -1}>{l}</option>
                                ))}
                            </select>
                            <Network className="absolute right-3 top-3 text-slate-400 pointer-events-none" size={16}/>
                        </div>
                        <p className="text-[10px] text-slate-400">Read-only aggregation level.</p>
                    </div>

                    <div className="space-y-2">
                        <label className="text-xs font-bold text-slate-500">Plan At (Edit Level)</label>
                        <div className="relative">
                            <select 
                                value={anchorLevel}
                                onChange={(e) => setAnchorLevel(e.target.value)}
                                className={`w-full p-3 border rounded-xl text-sm font-bold appearance-none outline-none focus:ring-2 focus:ring-indigo-500 ${
                                    isValid ? 'bg-white border-slate-200 text-indigo-700' : 'bg-rose-50 border-rose-200 text-rose-700'
                                }`}
                            >
                                {levels.map(l => (
                                    <option key={l} value={l}>{l}</option>
                                ))}
                            </select>
                            <Target className="absolute right-3 top-3 text-slate-400 pointer-events-none" size={16}/>
                        </div>
                        {!isValid && <p className="text-[10px] text-rose-500 font-bold">Must be lower than Group level</p>}
                        {isValid && <p className="text-[10px] text-slate-400">Target level for data entry.</p>}
                    </div>
                </div>
            </section>

            <hr className="border-slate-100" />

            {/* 2. TIME HORIZON */}
            <section>
                <div className="flex items-center gap-2 mb-4">
                    <Calendar className="text-emerald-600" size={18}/>
                    <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">Time Horizon</h3>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="space-y-2">
                        <label className="text-xs font-bold text-slate-500">Start Year</label>
                        <input 
                            type="number" 
                            value={startYear}
                            onChange={(e) => setStartYear(parseInt(e.target.value))}
                            className="w-full p-3 bg-white border border-slate-200 rounded-xl text-sm font-bold text-slate-700 outline-none focus:ring-2 focus:ring-emerald-500"
                        />
                    </div>

                    <div className="space-y-2">
                        <label className="text-xs font-bold text-slate-500">Duration (Years)</label>
                        <div className="flex items-center gap-2">
                            {[1, 3, 5].map(yr => (
                                <button
                                    key={yr}
                                    onClick={() => setHorizonYears(yr)}
                                    className={`flex-1 py-3 rounded-xl text-sm font-bold border transition-all ${
                                        horizonYears === yr 
                                        ? 'bg-emerald-600 text-white border-emerald-600 shadow-md' 
                                        : 'bg-white text-slate-500 border-slate-200 hover:bg-emerald-50'
                                    }`}
                                >
                                    {yr}Y
                                </button>
                            ))}
                        </div>
                    </div>

                    <div className="space-y-2">
                        <label className="text-xs font-bold text-slate-500">Granularity</label>
                        <div className="flex bg-slate-100 p-1 rounded-xl">
                            <button 
                                onClick={() => setTimeBucket('Week')}
                                className={`flex-1 py-2 rounded-lg text-xs font-bold transition-all ${timeBucket === 'Week' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-400'}`}
                            >
                                Weekly
                            </button>
                            <button 
                                onClick={() => setTimeBucket('Month')}
                                className={`flex-1 py-2 rounded-lg text-xs font-bold transition-all ${timeBucket === 'Month' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-400'}`}
                            >
                                Monthly
                            </button>
                        </div>
                    </div>
                </div>
            </section>
        </div>

        {/* FOOTER */}
        <div className="p-6 bg-slate-50 border-t border-slate-100 flex justify-between items-center">
            <div className="flex items-center gap-2 text-xs text-slate-500">
                {isValid && (
                    <>
                        <CheckCircle size={14} className="text-emerald-500"/>
                        <span>Config Ready: {aggregateLevel} &gt; {anchorLevel} ({horizonYears} Years)</span>
                    </>
                )}
            </div>
            
            <div className="flex gap-3">
                <button onClick={onClose} className="px-6 py-3 text-sm font-bold text-slate-500 hover:bg-slate-100 rounded-xl transition-colors">
                    Cancel
                </button>
                <button 
                    onClick={handleSubmit}
                    disabled={!isValid || loading}
                    className="px-8 py-3 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl font-bold shadow-lg shadow-indigo-200 transition-all flex items-center gap-2"
                >
                    {loading ? 'Loading...' : 'Launch Grid'} <ArrowRight size={18} />
                </button>
            </div>
        </div>

      </div>
    </div>
  );
};
