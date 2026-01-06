import React, { useState } from 'react';
import { 
  GitPullRequest, DollarSign, Percent, Box, Truck, 
  PieChart, Play, RefreshCw, BarChart3, AlertTriangle,
  CheckCircle, XCircle, Zap, Scale, Gavel, Package, Activity,
  TrendingUp // <--- ADDED THIS MISSING IMPORT
} from 'lucide-react';
import { api } from '../services/api';
import { AdvancedPlanningGrid } from '../components/AdvancedPlanningGrid'; 

interface WorkbenchProps {
    contextTitle?: string;
    endpoint?: string;
    [key: string]: any;
}

const DecisionWorkbench: React.FC<WorkbenchProps> = ({ 
    contextTitle = "Decision Workbench",
    endpoint = "/orchestration/run" 
}) => {
  const [candidates, setCandidates] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState('ALL'); 
  const [activeTab, setActiveTab] = useState('REPLENISHMENT'); 
  
  const [processingId, setProcessingId] = useState<string | null>(null);
  const [debatingId, setDebatingId] = useState<string | null>(null);
  const [approvedIds, setApprovedIds] = useState<Set<string>>(new Set());

  // Dynamic Stats
  const actionableCount = candidates.filter(c => ['ORDER', 'HIKE', 'MARKDOWN'].includes(c.decision)).length;
  const autoCount = candidates.filter(c => c.decision === 'AUTO_ORDER').length;
  const debateCount = candidates.filter(c => c.decision === 'DEBATE_POSSIBLE').length;

  const getMode = () => {
      if (endpoint?.includes('price')) return 'PRICING';
      if (endpoint?.includes('markdown')) return 'MARKDOWN';
      if (endpoint?.includes('allocation')) return 'ALLOCATION';
      if (endpoint?.includes('assortment')) return 'ASSORTMENT';
      return 'REPLENISHMENT';
  };

  const runSimulation = async () => {
      setLoading(true);
      setApprovedIds(new Set());
      try {
          const url = `http://localhost:8000${endpoint}`;
          const res = await fetch(url, { method: 'POST' });
          if (!res.ok) throw new Error("Simulation failed");
          
          const data = await res.json();
          setCandidates(data.candidates || []);
      } catch (e) {
          console.error(e);
          alert("Simulation failed. Ensure Backend is running.");
      } finally {
          setLoading(false);
      }
  };

  const handleConveneCouncil = async (row: any) => {
      if (debatingId) return;
      setDebatingId(row.node_id);
      try {
          const res = await fetch('http://localhost:8000/orchestration/convene_council', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ node_id: row.node_id, mode: getMode() })
          });
          if (!res.ok) throw new Error("Council failed");
          const verdict = await res.json();
          
          setCandidates(prev => prev.map(c => {
              if (c.node_id === row.node_id) {
                  return {
                      ...c,
                      decision: verdict.decision === 'HOLD' ? 'HOLD' : (getMode() === 'PRICING' ? 'HIKE' : 'ORDER'),
                      recommendation: verdict.recommendation,
                      rationale: verdict.rationale,
                      transcript: verdict.transcript
                  };
              }
              return c;
          }));
      } catch (e) {
          alert("Failed to convene council.");
      } finally {
          setDebatingId(null);
      }
  };

  const handleApprove = async (row: any) => {
      if (processingId) return;
      setProcessingId(row.node_id);
      try {
          await api.decision.commit(row);
          setApprovedIds(prev => new Set(prev).add(row.node_id));
      } catch (e) {
          alert("Commit failed.");
      } finally {
          setProcessingId(null);
      }
  };

  const filteredCandidates = candidates.filter(c => {
      if (filter === 'ALL') return true;
      if (filter === 'AUTO') return c.decision === 'AUTO_ORDER';
      if (filter === 'MANUAL') return ['ORDER', 'HIKE', 'MARKDOWN', 'DEBATE_POSSIBLE'].includes(c.decision);
      return c.decision === filter;
  });

  // --- RENDER ---
  
  if (contextTitle === "Financial Plan" || contextTitle?.includes("Budget")) {
      return (
        <div className="p-6 h-full flex flex-col">
            <div className="flex justify-between items-end mb-6">
                <h2 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
                    <BarChart3 className="text-indigo-600"/> {contextTitle}
                </h2>
            </div>
            <div className="flex-1 bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
                <AdvancedPlanningGrid />
            </div>
        </div>
      );
  }

  return (
    <div className="p-8 h-full bg-slate-50 font-sans text-slate-900 flex flex-col">
      <div className="flex flex-col md:flex-row justify-between items-end mb-8 gap-4">
        <div>
          <h2 className="text-3xl font-bold text-slate-900 flex items-center gap-3">
            <TrendingUp className="text-indigo-600" /> {contextTitle}
          </h2>
          <p className="text-slate-500 mt-2">
            System 1 (Auto-Pilot) • System 2 (Review) • System 3 (On-Demand)
          </p>
        </div>
        
        <div className="flex gap-4 items-center">
            {candidates.length > 0 && (
                <div className="flex gap-3 mr-4 animate-in fade-in">
                    <div className="px-4 py-2 bg-emerald-50 border border-emerald-200 rounded-xl shadow-sm flex items-center gap-2">
                        <Zap size={14} className="text-emerald-600"/>
                        <span className="font-bold text-lg text-emerald-800">{autoCount}</span>
                    </div>
                    {debateCount > 0 && (
                        <div className="px-4 py-2 bg-purple-50 border border-purple-200 rounded-xl shadow-sm flex items-center gap-2">
                            <Scale size={14} className="text-purple-600"/>
                            <span className="font-bold text-lg text-purple-800">{debateCount}</span>
                        </div>
                    )}
                    <div className="px-4 py-2 bg-white border border-slate-200 rounded-xl shadow-sm">
                        <span className="font-bold text-lg text-indigo-600">{actionableCount}</span>
                    </div>
                </div>
            )}

            <button 
                onClick={runSimulation}
                disabled={loading}
                className={`
                    px-6 py-3 rounded-xl font-bold shadow-lg flex items-center gap-2 transition-all
                    ${loading ? 'bg-slate-100 text-slate-400' : 'bg-indigo-600 text-white hover:scale-105 shadow-indigo-200'}
                `}
            >
                {loading ? <RefreshCw className="animate-spin"/> : <Play fill="currentColor" />}
                {loading ? 'Processing...' : 'Run Simulation'}
            </button>
        </div>
      </div>

      <div className="flex-1 bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden flex flex-col">
          {candidates.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center text-slate-400">
                  <Activity size={48} className="text-indigo-100 mb-4" />
                  <p>Ready to Orchestrate</p>
              </div>
          ) : (
              <>
                <div className="p-4 border-b border-slate-100 flex gap-4 bg-slate-50/50">
                    {['ALL', 'AUTO', 'MANUAL'].map(f => (
                        <button 
                            key={f}
                            onClick={() => setFilter(f)}
                            className={`px-4 py-2 rounded-lg text-xs font-bold transition-colors ${filter === f ? 'bg-slate-800 text-white' : 'bg-white text-slate-600 hover:bg-slate-200'}`}
                        >
                            {f === 'ALL' ? 'All Decisions' : f === 'AUTO' ? 'Auto-Pilot' : 'Review Needed'}
                        </button>
                    ))}
                </div>

                <div className="flex-1 overflow-auto">
                    <table className="w-full text-left text-sm">
                        <thead className="bg-slate-50 text-slate-500 font-bold text-xs uppercase sticky top-0 z-10">
                            <tr>
                                <th className="p-4">Context</th>
                                <th className="p-4 text-right">Current</th>
                                <th className="p-4 text-right">Forecast</th>
                                <th className="p-4 text-right">Proposal</th>
                                <th className="p-4 w-1/3">Rationale</th>
                                <th className="p-4 text-center">Action</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            {filteredCandidates.map((row, i) => {
                                const isApproved = approvedIds.has(row.node_id);
                                const isDebate = row.decision === 'DEBATE_POSSIBLE';
                                const isAuto = row.decision === 'AUTO_ORDER';
                                
                                return (
                                <tr key={i} className={`hover:bg-slate-50 ${isAuto ? 'bg-emerald-50/10' : ''}`}>
                                    <td className="p-4">
                                        <div className="font-bold text-slate-800">{row.name}</div>
                                        <div className="text-[10px] text-slate-400 font-mono">{row.node_id}</div>
                                    </td>
                                    <td className="p-4 text-right font-mono">{row.current_inv}</td>
                                    <td className="p-4 text-right font-mono">{row.daily_demand}</td>
                                    <td className="p-4 text-right font-bold text-indigo-600 font-mono">
                                        {isDebate ? '?' : row.recommendation}
                                    </td>
                                    <td className="p-4 text-xs text-slate-500">
                                        {row.transcript ? (
                                            <div className="p-2 bg-indigo-50 rounded border border-indigo-100 space-y-1">
                                                <div className="font-bold text-indigo-700 flex items-center gap-1"><Gavel size={12}/> {row.rationale}</div>
                                                {row.transcript.slice(-1).map((t: any, idx: number) => (
                                                    <div key={idx} className="italic text-slate-600">"{t.verdict || t.arg}"</div>
                                                ))}
                                            </div>
                                        ) : row.rationale}
                                    </td>
                                    <td className="p-4 text-center">
                                        {isAuto ? (
                                            <span className="text-emerald-600 text-xs font-bold flex items-center justify-center gap-1"><Zap size={12} fill="currentColor"/> Executed</span>
                                        ) : isApproved ? (
                                            <span className="text-green-600 text-xs font-bold flex items-center justify-center gap-1"><CheckCircle size={12}/> Done</span>
                                        ) : isDebate ? (
                                            <button 
                                                onClick={() => handleConveneCouncil(row)}
                                                disabled={debatingId === row.node_id}
                                                className="px-3 py-1 bg-purple-100 text-purple-700 rounded-lg text-xs font-bold hover:bg-purple-200 transition-colors flex items-center gap-1 mx-auto"
                                            >
                                                {debatingId === row.node_id ? <RefreshCw className="animate-spin" size={12}/> : <Scale size={12}/>}
                                                Board
                                            </button>
                                        ) : ['ORDER', 'HIKE', 'MARKDOWN'].includes(row.decision) && (
                                            <div className="flex justify-center gap-2">
                                                <button onClick={() => handleApprove(row)} disabled={processingId === row.node_id} className="p-1.5 bg-green-50 text-green-600 rounded hover:bg-green-100">
                                                    <CheckCircle size={16}/>
                                                </button>
                                                <button className="p-1.5 bg-red-50 text-red-600 rounded hover:bg-red-100">
                                                    <XCircle size={16}/>
                                                </button>
                                            </div>
                                        )}
                                    </td>
                                </tr>
                            )})}
                        </tbody>
                    </table>
                </div>
              </>
          )}
      </div>
    </div>
  );
};

export default DecisionWorkbench;
