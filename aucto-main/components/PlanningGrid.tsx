
import React, { useState } from 'react';
import { Save, AlertCircle, TrendingUp, DollarSign } from 'lucide-react';

export interface GridRow {
  id: string;
  category: string;
  metric: 'Sales' | 'Inventory' | 'Receipts' | 'Margin %';
  jan: string | number;
  feb: string | number;
  mar: string | number;
  total: string | number;
  status: 'LOCKED' | 'EDITABLE' | 'PROPOSED';
}

interface PlanningGridProps {
  data: GridRow[];
  onPropose: (data: GridRow[]) => void;
}

export const PlanningGrid: React.FC<PlanningGridProps> = ({ data, onPropose }) => {
  const [gridData, setGridData] = useState<GridRow[]>(data);

  const handleCellEdit = (id: string, field: string, value: string) => {
    setGridData(prev => prev.map(row => 
      row.id === id ? { ...row, [field]: value, status: 'PROPOSED' } : row
    ));
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden flex flex-col h-full animate-in fade-in">
      {/* Header */}
      <div className="p-4 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
        <div>
           <h3 className="font-bold text-slate-800 flex items-center">
             <DollarSign size={16} className="mr-2 text-indigo-600" />
             Open-to-Buy (OTB) Workbench
           </h3>
           <p className="text-xs text-slate-500 mt-1">Q1 2024 Budget Allocation • Fiscal Year 24</p>
        </div>
        <button 
          onClick={() => onPropose(gridData)} 
          className="flex items-center space-x-2 bg-indigo-600 text-white px-4 py-2 rounded-lg text-xs font-bold hover:bg-indigo-700 shadow-md shadow-indigo-200 transition-all hover:-translate-y-0.5"
        >
           <Save size={14} /> <span>Submit Proposal to Boardroom</span>
        </button>
      </div>

      {/* Grid */}
      <div className="overflow-auto flex-1 custom-scrollbar">
        <table className="w-full text-xs text-right font-mono border-collapse">
           <thead className="bg-slate-50 text-slate-500 font-sans sticky top-0 z-10 shadow-sm">
              <tr>
                <th className="p-3 text-left border-b border-r sticky left-0 bg-slate-50 w-48">Category</th>
                <th className="p-3 text-left border-b border-r w-32">Metric</th>
                <th className="p-3 border-b border-slate-200 w-32">Jan '24 (Actual)</th>
                <th className="p-3 border-b border-amber-200 bg-amber-50/50 text-amber-700 w-32">Feb '24 (Plan)</th>
                <th className="p-3 border-b border-amber-200 bg-amber-50/50 text-amber-700 w-32">Mar '24 (Plan)</th>
                <th className="p-3 border-b border-slate-200 text-indigo-700 font-bold bg-indigo-50/30 w-32">Q1 Total</th>
              </tr>
           </thead>
           <tbody className="divide-y divide-slate-100">
              {gridData.map((row) => (
                <tr key={row.id} className="hover:bg-slate-50 transition-colors group">
                   <td className="p-3 text-left font-bold text-slate-700 sticky left-0 bg-white border-r group-hover:bg-slate-50">{row.category}</td>
                   <td className="p-3 text-left text-slate-500 border-r">{row.metric}</td>
                   
                   {/* Locked / Historic Data */}
                   <td className="p-3 text-slate-400">{row.jan}</td>
                   
                   {/* Editable Cells */}
                   <td className="p-0 border border-transparent focus-within:border-indigo-500 relative bg-amber-50/10">
                      <input 
                        type="text" 
                        value={row.feb}
                        onChange={(e) => handleCellEdit(row.id, 'feb', e.target.value)}
                        className="w-full h-full p-3 text-right bg-transparent outline-none focus:bg-white text-slate-800 font-bold"
                      />
                   </td>
                   <td className="p-0 border border-transparent focus-within:border-indigo-500 relative bg-amber-50/10">
                      <input 
                        type="text" 
                        value={row.mar}
                        onChange={(e) => handleCellEdit(row.id, 'mar', e.target.value)}
                        className="w-full h-full p-3 text-right bg-transparent outline-none focus:bg-white text-slate-800 font-bold"
                      />
                   </td>
                   
                   {/* Calculated Total */}
                   <td className="p-3 font-bold text-indigo-600 bg-indigo-50/10">{row.total}</td>
                </tr>
              ))}
           </tbody>
        </table>
      </div>

      {/* Footer / Context Legend */}
      <div className="p-2 bg-amber-50 text-amber-800 text-[10px] flex items-center justify-center border-t border-amber-100">
         <AlertCircle size={12} className="mr-1.5" />
         <span className="font-medium">Cells highlighted in Amber trigger System 2 validation logic upon submission.</span>
      </div>
    </div>
  );
};
