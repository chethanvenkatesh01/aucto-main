import React, { useState, useEffect, useMemo } from 'react';
import { 
  Package, 
  CheckCircle2, 
  XCircle, 
  TrendingUp, 
  AlertTriangle, 
  RefreshCw,
  ArrowRight,
  ArrowUpDown,
  Search,
  Filter
} from 'lucide-react';
import { api } from '../services/api';

// --- DATA CONTRACT (Matches Optimized Orchestrator) ---
export interface AssortmentRow {
  skuId: string;
  name: string;
  velocity: string; // 'High' | 'Low'
  decision: string; // 'KEEP' | 'DROP' | 'EXPAND' | 'DEBATE_POSSIBLE'
  rationale: string;
  metrics: {
    npv: number; // Projected Net Present Value (90d)
  };
}

// --- OPTIMIZED ROW COMPONENT (Prevents Re-renders) ---
const MatrixRow = React.memo(({ row, onAction }: { row: AssortmentRow, onAction: (id: string) => void }) => {
  
  const getStatusBadge = (decision: string) => {
    switch (decision) {
      case 'EXPAND':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-black tracking-wide bg-emerald-100 text-emerald-700 border border-emerald-200 shadow-sm">
            <TrendingUp size={12} /> GROW
          </span>
        );
      case 'DROP':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-black tracking-wide bg-rose-100 text-rose-700 border border-rose-200 shadow-sm">
            <XCircle size={12} /> KILL
          </span>
        );
      case 'DEBATE_POSSIBLE':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-black tracking-wide bg-amber-100 text-amber-700 border border-amber-200 shadow-sm">
            <AlertTriangle size={12} /> REVIEW
          </span>
        );
      default: 
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-black tracking-wide bg-slate-100 text-slate-600 border border-slate-200">
            <CheckCircle2 size={12} /> MAINTAIN
          </span>
        );
    }
  };

  return (
    <tr className="hover:bg-slate-50/80 transition-colors group border-b border-slate-50 last:border-0">
      {/* PRODUCT INFO */}
      <td className="px-6 py-3 w-[30%]">
        <div className="flex flex-col">
          <span className="text-xs font-bold text-slate-800 truncate" title={row.name}>{row.name}</span>
          <span className="text-[10px] text-slate-400 font-mono mt-0.5">{row.skuId}</span>
        </div>
      </td>

      {/* VELOCITY */}
      <td className="px-6 py-3 text-center w-[10%]">
        <span className={`text-[10px] font-bold px-2 py-1 rounded border ${
          row.velocity === 'High' 
            ? 'bg-indigo-50 text-indigo-600 border-indigo-100' 
            : 'bg-slate-50 text-slate-400 border-slate-200'
        }`}>
          {row.velocity}
        </span>
      </td>

      {/* NPV (PROFIT PHYSICS) */}
      <td className="px-6 py-3 text-right w-[15%]">
        <div className={`text-sm font-mono font-bold flex items-center justify-end gap-1 ${
          row.metrics?.npv >= 0 ? 'text-emerald-600' : 'text-rose-600'
        }`}>
          {row.metrics?.npv < 0 && <TrendingUp className="rotate-180" size={10} />}
          ${row.metrics?.npv?.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 }) || '0'}
        </div>
      </td>

      {/* DECISION */}
      <td className="px-6 py-3 text-center w-[15%]">
        {getStatusBadge(row.decision)}
      </td>

      {/* RATIONALE */}
      <td className="px-6 py-3 w-[30%]">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-slate-500 leading-tight truncate" title={row.rationale}>
            {row.rationale}
          </span>
          {row.decision === 'DEBATE_POSSIBLE' && (
            <button 
              onClick={() => onAction(row.skuId)}
              className="p-1.5 text-indigo-600 hover:bg-indigo-50 rounded-lg opacity-0 group-hover:opacity-100 transition-all"
              title="Send to Debate Council"
            >
              <ArrowRight size={14} />
            </button>
          )}
        </div>
      </td>
    </tr>
  );
});

export const AssortmentMatrix: React.FC = () => {
  // --- STATE ---
  const [data, setData] = useState<AssortmentRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Filtering & Sorting
  const [search, setSearch] = useState('');
  const [sortCol, setSortCol] = useState<'npv' | 'velocity' | 'decision'>('npv');
  const [sortAsc, setSortAsc] = useState(false);

  // --- ACTIONS ---
  const loadAssortment = async () => {
    setLoading(true);
    setError(null);
    try {
      // Calls the Optimized Backend Engine
      const result = await api.orchestration.getAssortment();
      setData(result);
    } catch (e) {
      console.error("Assortment Engine Failure", e);
      setError("Engine Offline");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadAssortment(); }, []);

  const handleAction = (id: string) => {
    // Placeholder for System 3 invocation
    console.log("Convening Council for", id);
  };

  // --- MEMOIZED SORTING/FILTERING (Frontend Physics) ---
  const processedData = useMemo(() => {
    let res = [...data];

    // 1. Filter
    if (search) {
      const lower = search.toLowerCase();
      res = res.filter(r => r.name.toLowerCase().includes(lower) || r.skuId.toLowerCase().includes(lower));
    }

    // 2. Sort
    res.sort((a, b) => {
      let valA, valB;
      if (sortCol === 'npv') {
        valA = a.metrics?.npv || 0;
        valB = b.metrics?.npv || 0;
      } else if (sortCol === 'velocity') {
        valA = a.velocity === 'High' ? 2 : 1;
        valB = b.velocity === 'High' ? 2 : 1;
      } else {
        valA = a.decision;
        valB = b.decision;
      }
      
      if (valA < valB) return sortAsc ? -1 : 1;
      if (valA > valB) return sortAsc ? 1 : -1;
      return 0;
    });

    return res;
  }, [data, search, sortCol, sortAsc]);

  // --- UI HELPERS ---
  const SortIcon = ({ col }: { col: string }) => (
    <button 
      onClick={() => {
        if (sortCol === col) setSortAsc(!sortAsc);
        else { setSortCol(col as any); setSortAsc(false); } // Default desc for new col
      }}
      className={`ml-1 p-1 rounded hover:bg-slate-200 ${sortCol === col ? 'text-indigo-600' : 'text-slate-400'}`}
    >
      <ArrowUpDown size={12} />
    </button>
  );

  return (
    <div className="flex flex-col h-full bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden animate-in fade-in duration-500">
      
      {/* 1. HEADER CONTROL PLANE */}
      <div className="flex justify-between items-center p-4 border-b border-slate-100 bg-slate-50/50">
        <div className="flex items-center gap-4">
          <div>
            <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2">
              <Package className="text-indigo-600" size={18} />
              Rationalization Matrix
            </h3>
            <p className="text-[10px] text-slate-500 mt-0.5 font-medium">
              Profit Physics: {data.length} SKUs Analyzed
            </p>
          </div>
          
          {/* Search Bar */}
          <div className="relative group">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 group-focus-within:text-indigo-500 transition-colors" size={14} />
            <input 
              type="text" 
              placeholder="Filter SKUs..." 
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9 pr-4 py-1.5 bg-white border border-slate-200 rounded-lg text-xs font-medium focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 outline-none w-48 transition-all"
            />
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1 text-[10px] text-slate-500 bg-white border border-slate-200 px-2 py-1 rounded-md">
            <Filter size={10} />
            <span>Sort: <strong>{sortCol.toUpperCase()}</strong></span>
          </div>
          <button 
            onClick={loadAssortment}
            disabled={loading}
            className="p-2 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg transition-colors border border-transparent hover:border-indigo-100"
            title="Re-run Engine"
          >
            <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
          </button>
        </div>
      </div>

      {/* 2. TABLE HEADER (Sticky) */}
      <div className="bg-slate-50 border-b border-slate-200 pr-2"> {/* pr-2 for scrollbar offset */}
        <table className="w-full text-left table-fixed">
          <thead>
            <tr className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
              <th className="px-6 py-3 w-[30%]">Product / SKU</th>
              <th className="px-6 py-3 w-[10%] text-center">
                <div className="flex items-center justify-center">
                  Velocity <SortIcon col="velocity"/>
                </div>
              </th>
              <th className="px-6 py-3 w-[15%] text-right">
                <div className="flex items-center justify-end">
                  Proj. Profit (NPV) <SortIcon col="npv"/>
                </div>
              </th>
              <th className="px-6 py-3 w-[15%] text-center">
                <div className="flex items-center justify-center">
                  Decision <SortIcon col="decision"/>
                </div>
              </th>
              <th className="px-6 py-3 w-[30%]">Economic Rationale</th>
            </tr>
          </thead>
        </table>
      </div>

      {/* 3. SCROLLABLE CONTENT (Virtualization-Ready) */}
      <div className="flex-1 overflow-y-auto custom-scrollbar bg-white relative">
        {loading && data.length === 0 ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/80 z-10">
            <RefreshCw className="animate-spin text-indigo-500 mb-2" size={24} />
            <span className="text-xs font-bold text-slate-500">Calculating Economic Value...</span>
          </div>
        ) : error ? (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-rose-500">
            <AlertTriangle size={32} className="mb-2" />
            <span className="text-sm font-bold">Analysis Failed</span>
            <span className="text-xs">{error}</span>
          </div>
        ) : processedData.length === 0 ? (
          <div className="p-12 text-center text-slate-400 text-sm">
            No SKUs match your filter.
          </div>
        ) : (
          <table className="w-full text-left table-fixed">
            <tbody className="divide-y divide-slate-50">
              {processedData.map((row) => (
                <MatrixRow key={row.skuId} row={row} onAction={handleAction} />
              ))}
            </tbody>
          </table>
        )}
      </div>
      
      {/* 4. FOOTER */}
      <div className="p-3 bg-slate-50 border-t border-slate-200 text-[10px] text-slate-500 flex justify-between items-center px-6">
        <div className="flex items-center gap-2">
          <AlertTriangle size={12} className="text-amber-500" />
          <span><strong>Guardrails Active:</strong> Negative NPV items flagged for auto-liquidation.</span>
        </div>
        <div className="font-mono bg-white px-2 py-0.5 rounded border border-slate-200">
          Showing {processedData.length} / {data.length}
        </div>
      </div>

    </div>
  );
};
