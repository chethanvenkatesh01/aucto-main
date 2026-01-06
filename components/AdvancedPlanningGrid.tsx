import React, { useState, useMemo, useRef, CSSProperties } from 'react';
// FIX: Namespace Import to catch both ESM and CJS exports
import * as ReactWindow from 'react-window';
import { 
  Lock, Unlock, ChevronRight, ChevronDown, 
  Calendar, Layers, RefreshCw, TrendingUp, TrendingDown,
  AlertTriangle
} from 'lucide-react';
import { usePlanningEngine } from '../hooks/usePlanningEngine';
import { PlanningContext, PlanNode, VersionType } from '../types';

// --- ROBUST INTEROP FIX ---
// This checks both locations where the component might be hidden by the bundler
const Grid = ReactWindow.VariableSizeGrid || (ReactWindow as any).default?.VariableSizeGrid;

// --- CONFIGURATION ---
const CELL_WIDTH = 100;
const HIERARCHY_COL_WIDTH = 250;
const ROW_HEIGHT = 40;
const HEADER_HEIGHT = 80;

// Metrics to display per period
const METRICS = [
  { id: 'sales_units', label: 'Sales U', fmt: 'number' },
  { id: 'sales_amt',   label: 'Sales $', fmt: 'currency' },
  { id: 'aur',         label: 'AUR',     fmt: 'currency' },
  { id: 'receipts',    label: 'Rec U',   fmt: 'number' },
  { id: 'inventory',   label: 'Inv U',   fmt: 'number' }
];

// Display Versions (Includes calculated 'var')
type DisplayVersion = VersionType | 'var';
const DISPLAY_VERSIONS: DisplayVersion[] = ['wp', 'ly', 'var', 'op'];

interface AdvancedPlanningGridProps {
  context: PlanningContext;
}

export const AdvancedPlanningGrid: React.FC<AdvancedPlanningGridProps> = ({ context }) => {
  // --- 1. ENGINE CONNECTION ---
  const { tree, loading, updateCell, toggleLock, refresh } = usePlanningEngine(context);
  
  // --- 2. LOCAL STATE ---
  const [currency, setCurrency] = useState<'USD' | 'EUR'>('USD');
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const gridRef = useRef<any>(null); 
  const headerRef = useRef<HTMLDivElement>(null);

  // FX Rates (Mock)
  const FX_RATE = currency === 'EUR' ? 0.85 : 1.0;

  // --- 3. DATA FLATTENER (Tree -> Grid Rows) ---
  const flatRows = useMemo(() => {
    const rows: { node: PlanNode; depth: number }[] = [];
    const traverse = (nodes: PlanNode[], depth: number) => {
      nodes.forEach(node => {
        rows.push({ node, depth });
        const isExpanded = depth === 0 || expandedNodes.has(node.id);
        if (isExpanded && node.children && node.children.length > 0) {
          traverse(node.children, depth + 1);
        }
      });
    };
    if (tree) traverse(tree, 0);
    return rows;
  }, [tree, expandedNodes]);

  // --- 4. COLUMN GENERATOR (Time x Metrics x Versions) ---
  const columns = useMemo(() => {
    if (flatRows.length === 0) return [];
    
    // Safety check for empty data
    const firstNodeData = flatRows[0].node.data;
    if (!firstNodeData) return [];

    const periodIds = Object.keys(firstNodeData).sort(); 
    
    const cols: { period: string; metric: typeof METRICS[0]; version: DisplayVersion }[] = [];
    
    periodIds.forEach(pid => {
      METRICS.forEach(metric => {
        DISPLAY_VERSIONS.forEach(ver => {
          cols.push({ period: pid, metric, version: ver });
        });
      });
    });
    
    return cols;
  }, [flatRows]);

  // --- 5. HANDLERS ---
  const toggleNode = (nodeId: string) => {
    setExpandedNodes(prev => {
      const next = new Set(prev);
      if (next.has(nodeId)) next.delete(nodeId);
      else next.add(nodeId);
      return next;
    });
    // Force grid re-measure since row count changed
    if (gridRef.current && gridRef.current.resetAfterRowIndex) {
        gridRef.current.resetAfterRowIndex(0);
    }
  };

  const handleScroll = ({ scrollLeft }: { scrollLeft: number }) => {
    if (headerRef.current) {
      headerRef.current.scrollLeft = scrollLeft;
    }
  };

  // --- 6. RENDERERS ---

  // A. THE CELL (The Atomic Unit)
  const Cell = ({ columnIndex, rowIndex, style }: { columnIndex: number; rowIndex: number; style: CSSProperties }) => {
    const row = flatRows[rowIndex];
    const col = columns[columnIndex];
    
    if (!row || !col) return <div style={style} />;

    const { period, metric, version } = col;
    
    // --- VALUE RESOLUTION LOGIC ---
    let displayVal = 0;
    let isLocked = false;
    let formatted = '-';
    let isVar = version === 'var';

    const periodData = row.node.data[period];
    if (!periodData) return <div style={style} className="bg-slate-50 border-r border-slate-100" />;

    if (isVar) {
        // CALCULATED: Variance % = (WP - LY) / LY
        // @ts-ignore
        const wpRaw = periodData[metric.id]?.['wp']?.value || 0;
        // @ts-ignore
        const lyRaw = periodData[metric.id]?.['ly']?.value || 0;
        
        if (lyRaw !== 0) {
            displayVal = (wpRaw - lyRaw) / lyRaw;
            formatted = displayVal.toLocaleString(undefined, { style: 'percent', minimumFractionDigits: 1 });
            if (displayVal > 0) formatted = `+${formatted}`;
        } else {
            formatted = '-';
        }
    } else {
        // STANDARD: Fetch from Data
        // @ts-ignore
        const cellData = periodData[metric.id]?.[version as VersionType];
        if (cellData) {
            displayVal = cellData.value * FX_RATE;
            isLocked = cellData.locked;
            formatted = displayVal.toLocaleString(undefined, {
                minimumFractionDigits: metric.fmt === 'currency' ? 0 : 0,
                maximumFractionDigits: 0
            });
        }
    }
    
    const isEditable = version === 'wp' && context.anchorLevel === row.node.level;
    
    // --- STYLING LOGIC ---
    let bgClass = "bg-white";
    let textClass = "text-slate-900";
    
    if (version === 'ly') { bgClass = "bg-slate-50"; textClass = "text-slate-500"; }
    if (version === 'op') { bgClass = "bg-slate-50"; textClass = "text-indigo-900 font-medium"; }
    if (version === 'wp') { bgClass = "bg-white"; textClass = "text-slate-900 font-bold"; }
    if (isLocked) { bgClass = "bg-amber-50"; textClass = "text-amber-900"; }
    
    // Variance Coloring
    if (isVar) {
        bgClass = "bg-white";
        if (displayVal > 0) textClass = "text-emerald-600 font-bold";
        else if (displayVal < 0) textClass = "text-rose-600 font-bold";
        else textClass = "text-slate-300";
    }

    return (
      <div 
        style={style} 
        className={`border-r border-b border-slate-200 flex items-center justify-end px-2 text-xs relative group ${bgClass} ${textClass}`}
      >
        {isEditable ? (
          <>
            <input 
              className={`w-full text-right bg-transparent outline-none ${isLocked ? 'cursor-not-allowed' : 'cursor-text'}`}
              value={formatted}
              disabled={isLocked} 
              onChange={(e) => {
                const val = parseFloat(e.target.value.replace(/,/g, ''));
                if (!isNaN(val)) {
                  updateCell(row.node.id, period, version as VersionType, metric.id as any, val / FX_RATE);
                }
              }}
            />
            {/* Lock Trigger */}
            <button 
              onClick={() => toggleLock(row.node.id, period, version as VersionType, metric.id)}
              className={`absolute left-1 top-1/2 -translate-y-1/2 p-0.5 rounded opacity-0 group-hover:opacity-100 transition-opacity ${isLocked ? 'text-amber-600 opacity-100' : 'text-slate-300 hover:text-indigo-500'}`}
            >
              {isLocked ? <Lock size={10} /> : <Unlock size={10} />}
            </button>
          </>
        ) : (
          <span className="flex items-center gap-1">
             {isVar && displayVal > 0 && <TrendingUp size={10}/>}
             {isVar && displayVal < 0 && <TrendingDown size={10}/>}
             {formatted}
          </span>
        )}
      </div>
    );
  };

  // --- 7. MAIN RENDER ---
  // Final Guard for robustness
  if (!Grid) {
      return (
        <div className="flex h-full items-center justify-center bg-slate-50 p-10">
            <div className="text-center text-rose-500">
                <AlertTriangle size={32} className="mx-auto mb-2" />
                <h3 className="text-lg font-bold">Grid Engine Error</h3>
                <p className="text-sm">Unable to load virtualization engine.</p>
                <p className="text-xs mt-1">Please ensure 'react-window' is installed correctly.</p>
            </div>
        </div>
      );
  }

  if (loading) {
      return (
          <div className="h-full flex flex-col items-center justify-center bg-white">
              <RefreshCw className="animate-spin text-indigo-600 mb-4" size={32}/>
              <h3 className="text-lg font-bold text-slate-800">Generating 5-Year Plan...</h3>
              <p className="text-slate-500">Aggregating {flatRows.length * columns.length} data points</p>
          </div>
      );
  }

  return (
    <div className="flex flex-col h-full bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">
      
      {/* TOOLBAR */}
      <div className="h-14 border-b border-slate-200 bg-white flex items-center justify-between px-4 shrink-0 z-20 relative shadow-sm">
        <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-indigo-900 font-bold">
                <Calendar size={18} className="text-indigo-600"/>
                <span>{context.startYear} - {context.startYear + context.horizonYears} Horizon</span>
            </div>
            <div className="h-4 w-px bg-slate-200"></div>
            <div className="flex items-center gap-2 text-xs text-slate-500">
                <Layers size={14}/>
                <span>Level: <strong className="text-slate-700">{context.anchorLevel}</strong></span>
            </div>
        </div>
        
        <div className="flex items-center gap-3">
            <div className="flex bg-slate-100 p-1 rounded-lg">
                <button 
                    onClick={() => setCurrency('USD')} 
                    className={`px-3 py-1 rounded-md text-xs font-bold transition-all ${currency === 'USD' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-400'}`}
                >USD ($)</button>
                <button 
                    onClick={() => setCurrency('EUR')} 
                    className={`px-3 py-1 rounded-md text-xs font-bold transition-all ${currency === 'EUR' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-400'}`}
                >EUR (€)</button>
            </div>
            
            <button onClick={() => refresh()} className="p-2 hover:bg-slate-100 rounded-lg text-slate-400 hover:text-indigo-600">
                <RefreshCw size={18}/>
            </button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden relative">
        
        {/* HIERARCHY TREE (Frozen) */}
        <div className="flex flex-col border-r border-slate-200 shadow-[4px_0_12px_-4px_rgba(0,0,0,0.1)] z-10 bg-white" style={{ width: HIERARCHY_COL_WIDTH }}>
            <div className="h-[80px] bg-slate-50 border-b border-slate-200 flex items-center px-4 font-bold text-slate-600 text-xs uppercase tracking-wider">
                Hierarchy
            </div>
            <div className="flex-1 overflow-hidden hover:overflow-y-auto custom-scrollbar">
                {flatRows.map((row, idx) => (
                    <div 
                        key={`${row.node.id}-${idx}`} 
                        className="flex items-center border-b border-slate-100 hover:bg-slate-50 transition-colors text-sm text-slate-700 select-none cursor-pointer"
                        style={{ height: ROW_HEIGHT, paddingLeft: row.depth * 16 + 12 }}
                        onClick={() => toggleNode(row.node.id)}
                    >
                        {row.node.children && row.node.children.length > 0 ? (
                            expandedNodes.has(row.node.id) ? <ChevronDown size={14} className="mr-1 text-slate-400"/> : <ChevronRight size={14} className="mr-1 text-slate-400"/>
                        ) : <div className="w-[14px] mr-1"/>}
                        <span className={`truncate ${row.depth === 0 ? 'font-bold text-indigo-900' : ''}`}>
                            {row.node.name}
                        </span>
                    </div>
                ))}
            </div>
        </div>

        {/* DATA GRID (Virtualized) */}
        <div className="flex-1 flex flex-col overflow-hidden">
            
            {/* STICKY HEADERS */}
            <div 
                ref={headerRef} 
                className="overflow-hidden bg-slate-50 border-b border-slate-200 select-none"
                style={{ height: HEADER_HEIGHT }}
            >
                <div className="flex" style={{ width: columns.length * CELL_WIDTH }}>
                    {columns.map((col, i) => {
                        const prevCol = columns[i-1];
                        const isNewPeriod = !prevCol || prevCol.period !== col.period;
                        const isNewMetric = !prevCol || prevCol.metric.id !== col.metric.id || isNewPeriod;
                        
                        return (
                            <div key={i} className="flex flex-col h-full border-r border-slate-200 bg-slate-50" style={{ width: CELL_WIDTH }}>
                                <div className={`h-6 flex items-center justify-center text-[10px] font-bold text-slate-500 uppercase ${isNewPeriod ? 'border-l border-slate-300' : ''}`}>
                                    {isNewPeriod ? col.period : ''}
                                </div>
                                <div className={`h-6 flex items-center justify-center text-[10px] text-slate-400 bg-slate-100 border-y border-slate-200 ${isNewMetric ? 'border-l border-slate-200' : ''}`}>
                                    {isNewMetric ? col.metric.label : ''}
                                </div>
                                <div className={`flex-1 flex items-center justify-center text-[9px] font-bold ${
                                    col.version === 'wp' ? 'text-indigo-600 bg-indigo-50/50' : 
                                    col.version === 'op' ? 'text-emerald-600' : 
                                    col.version === 'var' ? 'text-slate-600 bg-slate-100' : 'text-slate-400'
                                }`}>
                                    {col.version.toUpperCase()}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            <div className="flex-1">
                <Grid
                    ref={gridRef}
                    columnCount={columns.length}
                    columnWidth={() => CELL_WIDTH}
                    height={800}
                    rowCount={flatRows.length}
                    rowHeight={() => ROW_HEIGHT}
                    width={1200}
                    onScroll={handleScroll}
                    className="custom-scrollbar"
                >
                    {Cell}
                </Grid>
            </div>
        </div>
      </div>
      
      {/* FOOTER */}
      <div className="h-8 bg-slate-50 border-t border-slate-200 flex items-center justify-between px-4 text-[10px] text-slate-500">
          <div className="flex items-center gap-4">
              <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-indigo-600"></div> WP (Editable)</div>
              <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-emerald-600"></div> OP (Budget)</div>
              <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-slate-400"></div> LY (History)</div>
              <div className="flex items-center gap-1 font-bold text-slate-600">VAR %</div>
          </div>
          <div>
              Physics Engine: <strong>v10.20</strong> • Currency: <strong>{currency}</strong>
          </div>
      </div>
    </div>
  );
};
