import React, { useState, useEffect } from 'react';
import { 
  Truck, 
  AlertTriangle, 
  ArrowRight, 
  MapPin, 
  Activity, 
  CheckCircle, 
  RefreshCw,
  Box 
} from 'lucide-react';
import { api } from '../services/api';

// --- DATA CONTRACT (Matches backend response from orchestrator.py) ---
export interface TransferRequest {
  id: string;
  sku: string;
  source: string;
  dest: string;
  items: number;
  status: string;   // 'High' | 'Standard' | 'Low'
  decision: string; // 'APPROVED' | 'HOLD'
  rationale: string;
}

export const AllocationGrid: React.FC = () => {
  const [routes, setRoutes] = useState<TransferRequest[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load Live Allocation Data from Kernel
  useEffect(() => {
      fetchRoutes();
  }, []);

  const fetchRoutes = async () => {
      setLoading(true);
      setError(null);
      try {
          // Calls /orchestration/allocation (The Profit-Linked Engine)
          const data = await api.orchestration.getAllocation();
          setRoutes(data);
      } catch (e) {
          console.error("Allocation Engine Failed", e);
          setError("Failed to connect to Logistics Engine. Check backend status.");
      } finally {
          setLoading(false);
      }
  };

  const getPriorityBadge = (priority: string) => {
    switch (priority) {
      case 'High':
        return <span className="bg-rose-100 text-rose-700 px-2 py-1 rounded text-[10px] font-bold border border-rose-200">URGENT</span>;
      case 'Standard':
        return <span className="bg-blue-50 text-blue-700 px-2 py-1 rounded text-[10px] font-bold border border-blue-200">STANDARD</span>;
      default:
        return <span className="bg-slate-100 text-slate-600 px-2 py-1 rounded text-[10px] font-bold border border-slate-200">LOW</span>;
    }
  };

  const getDecisionBadge = (decision: string) => {
      return decision === 'APPROVED' 
        ? <span className="flex items-center gap-1 text-emerald-600 font-bold text-xs"><CheckCircle size={14}/> Approved</span>
        : <span className="flex items-center gap-1 text-amber-600 font-bold text-xs"><AlertTriangle size={14}/> Hold</span>;
  };

  return (
    <div className="h-full flex flex-col space-y-6 animate-in fade-in">
       
       {/* 1. KPI HEADER */}
       <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
           <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
               <div>
                   <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">Network Health</p>
                   <h3 className="text-2xl font-black text-slate-800 mt-1">98.4%</h3>
               </div>
               <div className="w-12 h-12 bg-emerald-50 rounded-full flex items-center justify-center text-emerald-600">
                   <Activity size={24} />
               </div>
           </div>
           <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
               <div>
                   <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">Active Transfers</p>
                   <h3 className="text-2xl font-black text-indigo-600 mt-1">{routes.filter(r => r.decision === 'APPROVED').length}</h3>
               </div>
               <div className="w-12 h-12 bg-indigo-50 rounded-full flex items-center justify-center text-indigo-600">
                   <Truck size={24} />
               </div>
           </div>
           <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between">
               <div>
                   <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">Pending Review</p>
                   <h3 className="text-2xl font-black text-amber-500 mt-1">{routes.filter(r => r.decision !== 'APPROVED').length}</h3>
               </div>
               <div className="w-12 h-12 bg-amber-50 rounded-full flex items-center justify-center text-amber-600">
                   <AlertTriangle size={24} />
               </div>
           </div>
       </div>

       {/* 2. MAIN GRID */}
       <div className="flex-1 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col">
           <div className="p-4 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
               <div>
                   <h3 className="font-bold text-slate-800 flex items-center gap-2">
                       <MapPin className="text-indigo-600" size={18}/>
                       Intelligent Allocation Matrix
                   </h3>
                   <p className="text-xs text-slate-500 mt-1">
                       Profit-Arbitrage Logic: Transfers are only approved if Margin Gain &gt; Shipping Cost.
                   </p>
               </div>
               <button 
                   onClick={fetchRoutes} 
                   disabled={loading}
                   className="flex items-center gap-2 px-4 py-2 bg-white hover:bg-slate-50 border border-slate-200 rounded-lg text-xs font-bold text-slate-700 transition-all shadow-sm active:scale-95"
               >
                   <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
                   {loading ? "Optimizing..." : "Refresh Logistics"}
               </button>
           </div>

           <div className="flex-1 overflow-auto">
               <table className="w-full text-left">
                   <thead className="bg-slate-50 sticky top-0 z-10 text-[10px] uppercase font-bold text-slate-500 tracking-wider">
                       <tr>
                           <th className="p-4 border-b border-slate-200">Transfer ID</th>
                           <th className="p-4 border-b border-slate-200">Product / SKU</th>
                           <th className="p-4 border-b border-slate-200">Route</th>
                           <th className="p-4 border-b border-slate-200 text-center">Units</th>
                           <th className="p-4 border-b border-slate-200">Priority</th>
                           <th className="p-4 border-b border-slate-200">Status</th>
                           <th className="p-4 border-b border-slate-200">Economic Rationale</th>
                       </tr>
                   </thead>
                   <tbody className="divide-y divide-slate-100">
                       {loading && routes.length === 0 ? (
                           <tr>
                               <td colSpan={7} className="p-12 text-center text-slate-400">
                                   <div className="flex flex-col items-center gap-2">
                                       <RefreshCw className="animate-spin text-indigo-400" size={24} />
                                       <span>Calculating Margin Arbitrage...</span>
                                   </div>
                               </td>
                           </tr>
                       ) : error ? (
                           <tr>
                               <td colSpan={7} className="p-12 text-center text-rose-500 font-bold bg-rose-50/50">
                                   <div className="flex flex-col items-center gap-2">
                                       <AlertTriangle size={24}/>
                                       {error}
                                   </div>
                               </td>
                           </tr>
                       ) : routes.length === 0 ? (
                           <tr>
                               <td colSpan={7} className="p-12 text-center text-slate-400">
                                   System Balanced. No profitable transfers identified (Savings &lt; Shipping Cost).
                               </td>
                           </tr>
                       ) : (
                           routes.map((r) => (
                               <tr key={r.id} className="hover:bg-slate-50/80 transition-colors group text-sm">
                                   <td className="p-4 font-mono text-slate-500 text-xs">{r.id}</td>
                                   <td className="p-4 font-bold text-slate-800 flex items-center gap-2">
                                       <Box size={14} className="text-slate-400"/>
                                       {r.sku}
                                   </td>
                                   <td className="p-4">
                                       <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
                                           <span className="bg-slate-100 px-2 py-1 rounded text-slate-500 border border-slate-200">{r.source}</span>
                                           <ArrowRight size={12} className="text-slate-300"/>
                                           <span className="bg-indigo-50 text-indigo-700 px-2 py-1 rounded border border-indigo-100">{r.dest}</span>
                                       </div>
                                   </td>
                                   <td className="p-4 text-center font-mono font-bold">{r.items.toLocaleString()}</td>
                                   <td className="p-4">{getPriorityBadge(r.status)}</td>
                                   <td className="p-4">{getDecisionBadge(r.decision)}</td>
                                   <td className="p-4 text-xs text-slate-500 max-w-[200px] truncate" title={r.rationale}>
                                       {r.rationale}
                                   </td>
                               </tr>
                           ))
                       )}
                   </tbody>
               </table>
           </div>
       </div>
    </div>
  );
};
