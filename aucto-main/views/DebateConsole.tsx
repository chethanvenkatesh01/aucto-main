import React, { useState, useEffect } from 'react';
import { 
  Users, Gavel, AlertTriangle, ArrowRight, 
  MessageSquare, TrendingUp, Shield, DollarSign,
  CheckCircle2, XCircle, BrainCircuit, RefreshCw
} from 'lucide-react';
import { api } from '../services/api';
import { PlanningRow } from '../types';

// --- TYPES ---
interface DebateSession {
  id: string;
  topic: string;
  severity: 'High' | 'Medium' | 'Low';
  status: 'Pending' | 'Active' | 'Resolved';
  context: PlanningRow;
  transcript: any[];
  verdict?: string;
}

// --- AGENT AVATAR COMPONENT ---
const AgentAvatar = ({ role, size = 'md' }: { role: string, size?: 'sm' | 'md' | 'lg' }) => {
  const colors: Record<string, string> = {
    'Sales_Agent': 'bg-emerald-100 text-emerald-700 border-emerald-300',
    'Finance_Agent': 'bg-amber-100 text-amber-700 border-amber-300',
    'Brand_Agent': 'bg-purple-100 text-purple-700 border-purple-300',
    'CEO_Verdict': 'bg-slate-800 text-white border-slate-600',
    'System': 'bg-slate-100 text-slate-500 border-slate-200'
  };
  
  const icons: Record<string, any> = {
    'Sales_Agent': TrendingUp,
    'Finance_Agent': DollarSign,
    'Brand_Agent': Shield,
    'CEO_Verdict': Gavel,
    'System': BrainCircuit
  };

  const Icon = icons[role] || Users;
  const style = colors[role] || colors['System'];
  
  const sizeClasses = {
    sm: 'w-6 h-6 text-xs',
    md: 'w-10 h-10 text-sm',
    lg: 'w-16 h-16 text-xl'
  };

  return (
    <div className={`rounded-full flex items-center justify-center border shadow-sm ${style} ${sizeClasses[size]}`}>
      <Icon size={size === 'sm' ? 12 : size === 'md' ? 18 : 28} />
    </div>
  );
};

export const DebateConsole: React.FC = () => {
  const [candidates, setCandidates] = useState<DebateSession[]>([]);
  const [activeSession, setActiveSession] = useState<DebateSession | null>(null);
  const [loading, setLoading] = useState(false);
  const [debating, setDebating] = useState(false);

  // 1. SCAN FOR CONFLICTS ON MOUNT
  useEffect(() => {
    loadConflicts();
  }, []);

  const loadConflicts = async () => {
    setLoading(true);
    try {
      // Fetch the full plan to find "DEBATE_POSSIBLE" rows
      const rows: PlanningRow[] = await api.orchestration.getFinancialPlan('LY', 'category', 'Month');
      
      const conflicts = rows
        .filter(r => r.decision === 'DEBATE_POSSIBLE' || r.metrics.gm_pct.WP < 30) // Filter logic
        .map(r => ({
          id: r.id,
          topic: `${r.hierarchy.dept} Strategy`,
          severity: r.metrics.gm_pct.WP < 20 ? 'High' : 'Medium',
          status: 'Pending',
          context: r,
          transcript: []
        } as DebateSession));

      setCandidates(conflicts);
      if (conflicts.length > 0 && !activeSession) {
        setActiveSession(conflicts[0]);
      }
    } catch (e) {
      console.error("Failed to load conflicts", e);
    } finally {
      setLoading(false);
    }
  };

  // 2. CONVENE THE COUNCIL
  const handleConvene = async () => {
    if (!activeSession) return;
    setDebating(true);
    
    try {
      // Call the "Weaponized" Backend
      const res = await api.orchestration.conveneCouncil(activeSession.context.hierarchy.dept, 'FINANCIAL');
      
      // Update the active session with the transcript
      const updatedSession = {
        ...activeSession,
        status: 'Resolved' as const,
        transcript: res.transcript || [],
        verdict: res.decision
      };

      setActiveSession(updatedSession);
      
      // Update list
      setCandidates(prev => prev.map(c => c.id === updatedSession.id ? updatedSession : c));

    } catch (e) {
      alert("Council failed to convene. Check backend.");
    } finally {
      setDebating(false);
    }
  };

  return (
    <div className="flex h-full bg-slate-50 overflow-hidden">
      
      {/* --- LEFT PANEL: CONFLICT QUEUE --- */}
      <div className="w-80 bg-white border-r border-slate-200 flex flex-col">
        <div className="p-4 border-b border-slate-100">
          <h2 className="font-bold text-slate-800 flex items-center gap-2">
            <AlertTriangle className="text-amber-500" size={18}/>
            Active Conflicts
            <span className="bg-amber-100 text-amber-700 text-xs px-2 py-0.5 rounded-full">{candidates.length}</span>
          </h2>
        </div>
        
        <div className="overflow-y-auto flex-1">
          {loading ? (
            <div className="p-8 text-center text-slate-400 text-sm">Scanning Grid...</div>
          ) : candidates.length === 0 ? (
            <div className="p-8 text-center text-slate-400 text-sm">System Nominal. No conflicts detected.</div>
          ) : (
            candidates.map(c => (
              <button
                key={c.id}
                onClick={() => setActiveSession(c)}
                className={`w-full text-left p-4 border-b border-slate-50 hover:bg-slate-50 transition-colors ${activeSession?.id === c.id ? 'bg-indigo-50/50 border-l-4 border-l-indigo-500' : 'border-l-4 border-l-transparent'}`}
              >
                <div className="flex justify-between items-start mb-1">
                  <span className="font-bold text-slate-700 text-sm">{c.topic}</span>
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${c.severity === 'High' ? 'bg-red-100 text-red-600' : 'bg-amber-100 text-amber-600'}`}>
                    {c.severity}
                  </span>
                </div>
                <div className="text-xs text-slate-500 line-clamp-2">
                  {c.context.rationale}
                </div>
                <div className="mt-2 flex items-center gap-2 text-[10px] text-slate-400">
                  <span className={c.status === 'Resolved' ? 'text-emerald-600 font-bold' : ''}>{c.status}</span>
                  <span>•</span>
                  <span>ID: {c.id}</span>
                </div>
              </button>
            ))
          )}
        </div>
      </div>

      {/* --- CENTER PANEL: THE COUNCIL CHAMBER --- */}
      <div className="flex-1 flex flex-col min-w-0 bg-slate-50/50">
        {activeSession ? (
          <>
            {/* HEADER */}
            <div className="p-4 bg-white border-b border-slate-200 shadow-sm flex justify-between items-center">
              <div>
                <h1 className="text-xl font-black text-slate-900">{activeSession.topic}</h1>
                <p className="text-xs text-slate-500 font-mono mt-1">
                  CONTEXT: {activeSession.context.rationale}
                </p>
              </div>
              <div>
                {activeSession.status === 'Pending' ? (
                  <button 
                    onClick={handleConvene}
                    disabled={debating}
                    className="flex items-center gap-2 px-6 py-2.5 bg-slate-900 text-white font-bold rounded-lg hover:bg-slate-800 transition-all shadow-md disabled:opacity-50"
                  >
                    {debating ? <RefreshCw className="animate-spin" size={18}/> : <Gavel size={18}/>}
                    {debating ? 'Convening...' : 'Convene Council'}
                  </button>
                ) : (
                  <div className="flex items-center gap-2 px-4 py-2 bg-emerald-50 text-emerald-700 font-bold rounded-lg border border-emerald-200">
                    <CheckCircle2 size={18}/> Verdict Reached
                  </div>
                )}
              </div>
            </div>

            {/* TRANSCRIPT */}
            <div className="flex-1 p-8 overflow-y-auto space-y-6">
              {activeSession.transcript.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-slate-300 opacity-50">
                  <Users size={64} className="mb-4"/>
                  <p className="text-lg font-bold">Chamber Empty</p>
                  <p className="text-sm">Convene the council to begin System 3 deliberation.</p>
                </div>
              ) : (
                activeSession.transcript.map((t, i) => (
                  <div key={i} className={`flex gap-4 max-w-3xl ${t.agent === 'CEO_Verdict' ? 'mx-auto w-full border-t-2 border-slate-200 pt-8 mt-8' : ''}`}>
                    <div className="shrink-0 mt-1">
                      <AgentAvatar role={t.agent} size={t.agent === 'CEO_Verdict' ? 'lg' : 'md'} />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-baseline gap-2 mb-1">
                        <span className="font-bold text-slate-800 text-sm">{t.agent.replace('_', ' ')}</span>
                        <span className="text-[10px] text-slate-400 font-mono">{new Date().toLocaleTimeString()}</span>
                      </div>
                      <div className={`p-4 rounded-2xl text-sm leading-relaxed shadow-sm ${
                        t.agent === 'CEO_Verdict' ? 'bg-slate-800 text-white' : 
                        t.agent === 'Sales_Agent' ? 'bg-emerald-50 text-slate-800 border border-emerald-100' :
                        t.agent === 'Finance_Agent' ? 'bg-amber-50 text-slate-800 border border-amber-100' :
                        'bg-white text-slate-700 border border-slate-200'
                      }`}>
                        {t.arg || t.verdict}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-400">
            Select a conflict to view details
          </div>
        )}
      </div>

      {/* --- RIGHT PANEL: CONTEXT DATA --- */}
      {activeSession && (
        <div className="w-80 bg-white border-l border-slate-200 p-6 flex flex-col gap-6">
          <div>
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Metric Impact</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg border border-slate-100">
                <span className="text-xs font-bold text-slate-600">Revenue (WP)</span>
                <span className="font-mono font-bold text-slate-900">${activeSession.context.metrics.sales_amt.WP.toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg border border-slate-100">
                <span className="text-xs font-bold text-slate-600">Margin %</span>
                <div className="flex items-center gap-2">
                    <span className={`font-mono font-bold ${activeSession.context.metrics.gm_pct.WP < 30 ? 'text-red-600' : 'text-slate-900'}`}>
                        {activeSession.context.metrics.gm_pct.WP.toFixed(1)}%
                    </span>
                    {activeSession.context.metrics.gm_pct.WP < 30 && <AlertTriangle size={12} className="text-red-500"/>}
                </div>
              </div>
              <div className="flex justify-between items-center p-3 bg-slate-50 rounded-lg border border-slate-100">
                <span className="text-xs font-bold text-slate-600">Inventory</span>
                <span className="font-mono font-bold text-slate-900">${activeSession.context.metrics.inventory.WP.toLocaleString()}</span>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-xs font-bold text-slate-400 uppercase tracking-widest mb-4">Required Actions</h3>
            {activeSession.status === 'Resolved' ? (
                <div className="space-y-2">
                    <button className="w-full py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-bold shadow-sm transition-colors flex items-center justify-center gap-2">
                        <CheckCircle2 size={14}/> Execute Verdict
                    </button>
                    <button className="w-full py-2 bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 rounded-lg text-xs font-bold transition-colors">
                        Appeal to Human (System 2)
                    </button>
                </div>
            ) : (
                <div className="text-center p-4 bg-slate-50 rounded-lg border border-slate-200 border-dashed text-xs text-slate-400">
                    Waiting for Council Decision...
                </div>
            )}
          </div>
        </div>
      )}

    </div>
  );
};