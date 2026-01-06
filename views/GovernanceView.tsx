import React, { useState, useEffect } from 'react';
import { 
  Shield, CheckCircle, AlertTriangle, Edit2, Save, X, 
  Lock, Unlock, Globe, Box, Plus, Search, RefreshCw
} from 'lucide-react';
import { api } from '../services/api';

const GovernanceView: React.FC = () => {
  // State for Policies
  const [policies, setPolicies] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Edit State
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [editValue, setEditValue] = useState<string>('');

  // --- 1. FETCH POLICIES ---
  const fetchPolicies = async () => {
      setLoading(true);
      setError(null);
      try {
          // This endpoint now connects to the Policy Engine (v9.3)
          const res = await fetch('http://localhost:8000/governance/policies');
          if (!res.ok) throw new Error("Failed to connect to Governance Engine");
          
          const data = await res.json();
          // Sort: Global first, then others
          const sorted = data.sort((a: any, b: any) => {
              if (a.entity_id === 'GLOBAL' && b.entity_id !== 'GLOBAL') return -1;
              if (a.entity_id !== 'GLOBAL' && b.entity_id === 'GLOBAL') return 1;
              return a.key.localeCompare(b.key);
          });
          
          setPolicies(sorted);
      } catch (e: any) {
          console.error("Governance Sync Failed", e);
          setError(e.message);
      } finally {
          setLoading(false);
      }
  };

  useEffect(() => {
      fetchPolicies();
  }, []);

  // --- 2. EDIT HANDLERS ---
  const handleEdit = (policy: any) => {
      // Create a unique composite key for editing state
      setEditingKey(policy.key + policy.entity_id);
      setEditValue(policy.value.toString());
  };

  const handleCancel = () => {
      setEditingKey(null);
      setEditValue('');
  };

  const handleSave = async (policy: any) => {
      try {
          const numVal = parseFloat(editValue);
          if (isNaN(numVal)) {
              alert("Please enter a valid number");
              return;
          }

          // Write back to the Policy Engine
          const res = await fetch('http://localhost:8000/governance/policies', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                  key: policy.key,
                  value: numVal,
                  entity_id: policy.entity_id
              })
          });

          if (res.ok) {
              setEditingKey(null);
              fetchPolicies(); // Refresh to see the 'source' change to DATABASE
          } else {
              alert("Failed to save policy override.");
          }
      } catch (e) {
          alert("Network error while saving.");
      }
  };

  return (
    <div className="p-8 h-full bg-slate-50 text-slate-900 overflow-y-auto font-sans">
      <div className="mb-8 flex justify-between items-end">
        <div>
            <h2 className="text-3xl font-bold text-slate-900 flex items-center gap-3">
            <Shield className="text-emerald-600" /> Governance & Guardrails
            </h2>
            <p className="text-slate-500 mt-2">
                Configure the safety constraints for System 1 (Auto-Pilot) and System 2 (Recommendations).
            </p>
        </div>
        <button 
            onClick={fetchPolicies} 
            className="p-2 text-slate-400 hover:text-indigo-600 transition-colors"
            title="Refresh Policies"
        >
            <RefreshCw size={18} className={loading ? "animate-spin" : ""} />
        </button>
      </div>

      {error && (
          <div className="mb-6 p-4 bg-red-50 text-red-700 rounded-xl border border-red-100 flex items-center gap-2">
              <AlertTriangle size={20}/>
              <span><strong>Connection Error:</strong> {error}. Is the backend running?</span>
          </div>
      )}

      {/* POLICY GRID */}
      <div className="grid grid-cols-1 gap-6">
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
              <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
                  <h3 className="font-bold text-slate-700 flex items-center gap-2">
                      <Globe size={18}/> Active Policies
                  </h3>
                  <button className="text-xs font-bold text-indigo-600 hover:bg-indigo-50 px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1">
                      <Plus size={14}/> Add Override
                  </button>
              </div>
              
              <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                      <thead className="bg-slate-50 text-slate-500 font-bold uppercase text-[11px] tracking-wider">
                          <tr>
                              <th className="p-4 pl-6">Scope</th>
                              <th className="p-4">Rule Key</th>
                              <th className="p-4">Threshold Value</th>
                              <th className="p-4">Source</th>
                              <th className="p-4 text-right pr-6">Action</th>
                          </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                          {policies.map((p, i) => {
                              const isEditing = editingKey === p.key + p.entity_id;
                              const isGlobal = p.entity_id === 'GLOBAL';
                              
                              return (
                                  <tr key={i} className="hover:bg-slate-50 transition-colors group">
                                      <td className="p-4 pl-6">
                                          {isGlobal ? (
                                              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-100 text-slate-600 text-xs font-bold border border-slate-200">
                                                  <Globe size={12}/> Global
                                              </span>
                                          ) : (
                                              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-indigo-50 text-indigo-600 text-xs font-bold border border-indigo-100">
                                                  <Box size={12}/> {p.entity_id}
                                              </span>
                                          )}
                                      </td>
                                      
                                      <td className="p-4 font-mono font-bold text-slate-700 text-xs tracking-wide">
                                          {p.key}
                                          <div className="text-[10px] font-normal text-slate-400 mt-0.5">
                                              {getPolicyDescription(p.key)}
                                          </div>
                                      </td>
                                      
                                      <td className="p-4">
                                          {isEditing ? (
                                              <div className="flex items-center gap-2">
                                                  <input 
                                                      type="number" 
                                                      value={editValue}
                                                      onChange={(e) => setEditValue(e.target.value)}
                                                      className="w-28 p-2 text-sm border border-indigo-300 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none shadow-sm"
                                                      autoFocus
                                                  />
                                              </div>
                                          ) : (
                                              <span className={`font-mono text-lg font-bold ${p.key.includes('SPEND') ? 'text-slate-800' : 'text-emerald-600'}`}>
                                                  {p.key.includes('SPEND') ? '$' : ''}
                                                  {p.value.toLocaleString()}
                                                  {p.key.includes('PCT') || p.key.includes('TRIGGER') || p.key.includes('DEPTH') ? '%' : ''}
                                              </span>
                                          )}
                                      </td>
                                      
                                      <td className="p-4">
                                          {p.source === 'DATABASE' ? (
                                              <span className="flex items-center gap-1.5 text-xs font-medium text-emerald-700 bg-emerald-50 px-2 py-1 rounded w-fit">
                                                  <CheckCircle size={12}/> Active DB
                                              </span>
                                          ) : (
                                              <span className="flex items-center gap-1.5 text-xs text-slate-400 bg-slate-100 px-2 py-1 rounded w-fit">
                                                  <Lock size={12}/> Code Default
                                              </span>
                                          )}
                                      </td>
                                      
                                      <td className="p-4 pr-6 text-right">
                                          {isEditing ? (
                                              <div className="flex justify-end gap-2">
                                                  <button 
                                                      onClick={() => handleSave(p)} 
                                                      className="p-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 shadow-sm transition-all hover:scale-105"
                                                      title="Save Changes"
                                                  >
                                                      <Save size={16}/>
                                                  </button>
                                                  <button 
                                                      onClick={handleCancel} 
                                                      className="p-2 bg-white border border-slate-200 text-slate-600 rounded-lg hover:bg-slate-50 transition-all"
                                                      title="Cancel"
                                                  >
                                                      <X size={16}/>
                                                  </button>
                                              </div>
                                          ) : (
                                              <button 
                                                  onClick={() => handleEdit(p)} 
                                                  className="p-2 text-slate-400 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg transition-all opacity-0 group-hover:opacity-100"
                                                  title="Edit Policy"
                                              >
                                                  <Edit2 size={16}/>
                                              </button>
                                          )}
                                      </td>
                                  </tr>
                              );
                          })}
                          
                          {policies.length === 0 && !loading && (
                              <tr>
                                  <td colSpan={5} className="p-8 text-center text-slate-400">
                                      No policies found. System is running on internal defaults.
                                  </td>
                              </tr>
                          )}
                      </tbody>
                  </table>
              </div>
          </div>
      </div>
    </div>
  );
};

// Helper to explain cryptic keys to the user
const getPolicyDescription = (key: string) => {
    const map: Record<string, string> = {
        "MAX_AUTO_SPEND": "Max value allowed for System 1 auto-orders.",
        "MIN_MARGIN_PCT": "Minimum gross margin allowed for pricing.",
        "MAX_PRICE_HIKE_PCT": "Max single-step price increase allowed.",
        "MAX_MARKDOWN_DEPTH": "Deepest discount allowed without human approval.",
        "SYSTEM_3_TRIGGER": "Confidence score below this triggers Council Debate."
    };
    return map[key] || "Custom business rule.";
};

export default GovernanceView;
