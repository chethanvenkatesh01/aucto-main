
import React, { useEffect, useRef, useState } from 'react';
import { X, Terminal, CheckCircle, Circle, AlertTriangle, Play, RefreshCw, Server, Shield } from 'lucide-react';
import { AuctobotLog, Plane } from '../types';

interface AuctobotConsoleProps {
  isOpen: boolean;
  onClose: () => void;
}

const MOCK_LOGS: AuctobotLog[] = [
  { id: '1', timestamp: '03:00:00', plane: Plane.DATA, action: 'Triggered Weekly Cycle (Cron: 0 3 * * MON)', status: 'success' },
  { id: '2', timestamp: '03:00:05', plane: Plane.DATA, action: 'Ingesting Sales Data (Snowflake)', status: 'success', details: '14k rows ingested, 0 errors' },
  { id: '3', timestamp: '03:00:12', plane: Plane.DATA, action: 'Running GreatExpectations Suite', status: 'success', details: 'All contracts passed' },
  { id: '4', timestamp: '03:01:45', plane: Plane.INTELLIGENCE, action: 'Retraining Demand Model v3.4', status: 'success', details: 'Accuracy improved by 0.4%' },
  // Changed Plane.DECISION to Plane.ORCHESTRATION
  { id: '5', timestamp: '03:05:20', plane: Plane.ORCHESTRATION, action: 'Generating Decision Package', status: 'success', details: 'Optimized pricing for 1,200 SKUs' },
  { id: '6', timestamp: '03:05:25', plane: Plane.GOVERNANCE, action: 'Checking Guardrails (OPA)', status: 'warning', details: '3 SKUs hit Max Price Cap' },
  { id: '7', timestamp: '03:05:30', plane: Plane.GOVERNANCE, action: 'Calculating DQS', status: 'success', details: 'Score: 92.4 (Auto-Approval Threshold: 90)' },
  // Changed Plane.DECISION to Plane.ORCHESTRATION
  { id: '8', timestamp: '03:06:00', plane: Plane.ORCHESTRATION, action: 'Executing Actions via ERP API', status: 'running', details: 'Pushing updates...' },
];

const AuctobotConsole: React.FC<AuctobotConsoleProps> = ({ isOpen, onClose }) => {
  const [logs, setLogs] = useState<AuctobotLog[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen && logs.length === 0) {
      // Replay logs for effect
      let delay = 0;
      MOCK_LOGS.forEach((log, index) => {
        delay += index === 0 ? 0 : 800;
        setTimeout(() => {
          setLogs(prev => [...prev, log]);
        }, delay);
      });
    }
  }, [isOpen]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 w-full max-w-4xl rounded-xl shadow-2xl border border-slate-700 overflow-hidden flex flex-col max-h-[80vh]">
        {/* Header */}
        <div className="bg-slate-950 p-4 border-b border-slate-800 flex justify-between items-center">
          <div className="flex items-center space-x-3">
            <div className="h-8 w-8 bg-indigo-600 rounded flex items-center justify-center">
              <Terminal className="text-white" size={18} />
            </div>
            <div>
              <h3 className="font-bold text-white">Auctobot Orchestrator</h3>
              <div className="flex items-center space-x-2 text-xs">
                <span className="text-emerald-400 flex items-center"><Circle size={6} fill="currentColor" className="mr-1" /> Online</span>
                <span className="text-slate-500">|</span>
                <span className="text-slate-400">Cycle ID: WK44-2023</span>
              </div>
            </div>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
            <X size={20} />
          </button>
        </div>

        {/* Visualization Pipeline */}
        <div className="bg-slate-900 p-6 border-b border-slate-800 flex justify-between items-center relative">
          {/* Connector Line */}
          <div className="absolute top-1/2 left-10 right-10 h-0.5 bg-slate-800 -z-0"></div>
          
          {[
            { label: 'Data', icon: Server, active: true },
            { label: 'Model', icon: RefreshCw, active: true },
            { label: 'Decision', icon: Play, active: true },
            { label: 'Govern', icon: Shield, active: true },
            { label: 'Exec', icon: CheckCircle, active: logs.length >= 8 }
          ].map((step, idx) => (
            <div key={idx} className="relative z-10 flex flex-col items-center space-y-2">
              <div className={`w-12 h-12 rounded-full border-2 flex items-center justify-center transition-all duration-500 ${
                step.active 
                  ? 'bg-indigo-900 border-indigo-500 text-indigo-400 shadow-[0_0_15px_rgba(99,102,241,0.5)]' 
                  : 'bg-slate-800 border-slate-700 text-slate-600'
              }`}>
                <step.icon size={20} />
              </div>
              <span className={`text-xs font-medium ${step.active ? 'text-indigo-300' : 'text-slate-600'}`}>{step.label}</span>
            </div>
          ))}
        </div>

        {/* Logs Console */}
        <div className="flex-1 bg-black/40 p-4 overflow-y-auto font-mono text-sm" ref={scrollRef}>
          <div className="space-y-1">
            {logs.map((log) => (
              <div key={log.id} className="flex space-x-3 items-start hover:bg-white/5 p-1 rounded">
                <span className="text-slate-500 shrink-0">[{log.timestamp}]</span>
                <span className={`font-bold shrink-0 w-24 ${
                  log.plane === Plane.DATA ? 'text-blue-400' :
                  log.plane === Plane.INTELLIGENCE ? 'text-purple-400' :
                  // Changed Plane.DECISION to Plane.ORCHESTRATION
                  log.plane === Plane.ORCHESTRATION ? 'text-amber-400' :
                  log.plane === Plane.GOVERNANCE ? 'text-red-400' : 'text-emerald-400'
                }`}>{log.plane.split(' ')[0]}</span>
                <div className="flex-1">
                  <span className="text-slate-200">{log.action}</span>
                  {log.details && (
                    <span className="text-slate-500 ml-2 border-l border-slate-700 pr-2 italic">
                      {log.details}
                    </span>
                  )}
                </div>
                <div>
                  {log.status === 'success' && <span className="text-emerald-500">OK</span>}
                  {log.status === 'warning' && <span className="text-amber-500">WARN</span>}
                  {log.status === 'running' && <span className="text-blue-400 animate-pulse">...</span>}
                </div>
              </div>
            ))}
            {logs.length === MOCK_LOGS.length && (
              <div className="text-emerald-500 pt-4">
                {'>'} Cycle Completed Successfully. 
                <span className="animate-pulse">_</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AuctobotConsole;
