import React, { useState, useEffect } from 'react';
import { TrendingUp, Activity, Zap, GitCommit, RefreshCw, AlertTriangle, PlayCircle } from 'lucide-react';
import { PlanningScope } from '../types';
import { api } from '../services/api';

interface ForecastWorkbenchProps {
  scope: PlanningScope;
}

export const ForecastWorkbench: React.FC<ForecastWorkbenchProps> = ({ scope }) => {
  const [forecastData, setForecastData] = useState<number[]>([]);
  const [confidence, setConfidence] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [trainLoading, setTrainLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const DEMO_SKU = "SKU-101";

  const fetchLiveForecast = async () => {
    setLoading(true);
    setError(null);
    try {
        const data = await api.intelligence.getForecast(DEMO_SKU, 8);
        
        // Safety Check: Backend might return error in JSON
        if (data.error) {
            setError(data.error);
            setForecastData([]);
        } else {
            setForecastData(data.forecast || []);
            setConfidence(data.model_confidence || 0);
        }
    } catch (err) {
        console.error(err);
        setError("Failed to connect to ML Engine.");
    } finally {
        setLoading(false);
    }
  };

  // NEW: Retrain Handler
  const handleRetrain = async () => {
      setTrainLoading(true);
      try {
          const res = await api.intelligence.triggerTraining();
          if (res.status === 'success') {
              alert(`Training Complete!\nAccuracy: ${res.accuracy}%\nSamples: ${res.samples}`);
              fetchLiveForecast(); // Refresh chart immediately
          } else {
              alert(`Training Error: ${res.message}`);
          }
      } catch (e) {
          alert("Training failed. Check backend console.");
      } finally {
          setTrainLoading(false);
      }
  };

  useEffect(() => {
    fetchLiveForecast();
  }, []);

  const periods = ['D+1', 'D+2', 'D+3', 'D+4', 'D+5', 'D+6', 'D+7', 'D+8'];

  return (
    <div className="flex flex-col h-full space-y-6 animate-in fade-in">
      
      {/* 1. MODEL CONTROL PANEL */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 shrink-0">
         {/* A. Model Leaderboard */}
         <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
             <div className="flex justify-between items-center mb-4">
                 <h3 className="text-xs font-bold text-slate-700 uppercase tracking-widest flex items-center">
                     <Activity size={14} className="mr-2 text-indigo-600"/> Live Model Inference
                 </h3>
                 <span className={`text-[10px] px-2 py-0.5 rounded font-bold ${confidence > 70 ? 'bg-emerald-50 text-emerald-700' : 'bg-amber-50 text-amber-700'}`}>
                    {confidence}% Confidence
                 </span>
             </div>
             <div className="space-y-2">
                 <div className="p-3 rounded-lg border bg-indigo-50 border-indigo-500 ring-1 ring-indigo-500 flex justify-between items-center">
                     <div className="flex items-center space-x-2">
                         <div className="w-2 h-2 rounded-full bg-indigo-600 animate-pulse"></div>
                         <span className="text-xs font-bold text-indigo-900">RandomForest (V5)</span>
                     </div>
                     <button 
                        onClick={handleRetrain}
                        disabled={trainLoading}
                        className="text-[10px] font-bold text-indigo-600 hover:text-indigo-800 underline disabled:opacity-50"
                     >
                        {trainLoading ? 'Training...' : 'Retrain Now'}
                     </button>
                 </div>
             </div>
         </div>

         {/* B. Feature Attribution */}
         <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm lg:col-span-2">
             <h3 className="text-xs font-bold text-slate-700 uppercase tracking-widest flex items-center mb-4">
                 <Zap size={14} className="mr-2 text-amber-500"/> Demand Signals
             </h3>
             <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                 {[
                     { label: 'Target SKU', val: DEMO_SKU, desc: 'Primary Item', pos: true },
                     { label: 'Horizon', val: '8 Days', desc: 'Recursive Lag', pos: true },
                     { label: 'Feedback', val: 'Active', desc: 'Auto-Regressive', pos: true },
                     { label: 'Source', val: 'Ledger', desc: 'Real-time SQLite', pos: true },
                 ].map(d => (
                     <div key={d.label} className="p-3 bg-slate-50 rounded-lg border border-slate-100">
                         <div className="text-[10px] text-slate-400 font-bold uppercase mb-1">{d.label}</div>
                         <div className="text-sm font-bold text-slate-700">{d.val}</div>
                         <div className="text-[10px] text-slate-500 leading-tight mt-1">{d.desc}</div>
                     </div>
                 ))}
             </div>
         </div>
      </div>

      {/* 2. INTERACTIVE FORECAST CHART */}
      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex-1 min-h-[300px] flex flex-col">
          <div className="flex justify-between items-center mb-6">
              <h3 className="font-bold text-slate-800 flex items-center">
                  <TrendingUp size={16} className="mr-2 text-indigo-600"/> 
                  Forecast Horizon
              </h3>
              <div className="flex space-x-2">
                  <button 
                    onClick={fetchLiveForecast}
                    disabled={loading}
                    className="flex items-center space-x-1 px-3 py-1.5 bg-slate-100 rounded-lg text-xs font-bold text-slate-600 hover:bg-slate-200"
                  >
                      <RefreshCw size={12} className={loading ? "animate-spin" : ""}/> <span>Refresh</span>
                  </button>
                  <button className="flex items-center space-x-1 px-3 py-1.5 bg-indigo-600 rounded-lg text-xs font-bold text-white shadow-md hover:bg-indigo-700">
                      <GitCommit size={12}/> <span>Commit Plan</span>
                  </button>
              </div>
          </div>
          
          <div className="flex-1 flex items-end justify-between space-x-2 px-4 pb-4 border-b border-l border-slate-200 relative">
              {/* Error State */}
              {error ? (
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-rose-500 bg-white/80 z-10">
                      <AlertTriangle size={32} className="mb-2"/> 
                      <span className="font-bold">{error}</span>
                      <button onClick={handleRetrain} className="mt-4 px-4 py-2 bg-rose-100 text-rose-700 rounded-lg text-xs font-bold hover:bg-rose-200">
                          Fix: Train Model Now
                      </button>
                  </div>
              ) : null}

              {/* Chart Bars */}
              {periods.map((p, i) => {
                  // Safety: Ensure we have data
                  const foreVal = forecastData.length > i ? forecastData[i] : 0;
                  const maxVal = Math.max(...forecastData, 10) * 1.2;
                  
                  return (
                      <div key={p} className="flex-1 flex flex-col items-center justify-end h-full relative group">
                          {forecastData.length > 0 && (
                              <div 
                                style={{ height: `${Math.min((foreVal / maxVal) * 100, 100)}%` }} 
                                className="w-full max-w-[40px] bg-indigo-500/20 border-t-2 border-indigo-500 rounded-t-sm relative transition-all duration-500 group-hover:bg-indigo-500/40"
                              >
                                  <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-slate-800 text-white text-[10px] py-1 px-2 rounded opacity-0 group-hover:opacity-100 transition-opacity z-10 font-mono">
                                      {foreVal}
                                  </div>
                              </div>
                          )}
                          <span className="mt-3 text-[10px] font-bold text-slate-400 uppercase">{p}</span>
                      </div>
                  );
              })}
          </div>
      </div>
    </div>
  );
};
