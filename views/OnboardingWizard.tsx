import React, { useState, useEffect } from 'react';
import { 
  Layers, 
  MapPin, 
  CheckCircle, 
  ArrowRight, 
  Save, 
  Edit2, 
  Plus, 
  Trash2,
  AlertTriangle 
} from 'lucide-react';
import { api } from '../services/api';
import { CartridgeManifest } from '../types';

interface OnboardingWizardProps {
  onNavigate: (view: string) => void;
  cartridge: CartridgeManifest | null;
}

export const OnboardingWizard: React.FC<OnboardingWizardProps> = ({ onNavigate, cartridge }) => {
  const [step, setStep] = useState(1);
  const [loading, setLoading] = useState(false);
  
  // Dynamic Vocabulary (Defaults to Retail terms if no cartridge)
  const termProduct = cartridge?.vocabulary.product || 'Product';
  const termLocation = cartridge?.vocabulary.location || 'Location';

  // Smart Defaults
  const defaultProdRoot = cartridge?.dataSchema?.find(d => d.type === 'MASTER')?.mandatoryFields[0] || 'Category';
  const defaultLocRoot = cartridge?.dataSchema?.find(d => d.id === 'NETWORK')?.mandatoryFields[0] || 'Region';

  // Product Tree State
  const [prodLevels, setProdLevels] = useState<string[]>([defaultProdRoot, termProduct]);
  const [isProdLocked, setIsProdLocked] = useState(false);

  // Location Tree State
  const [locLevels, setLocLevels] = useState<string[]>([defaultLocRoot, termLocation]);
  const [isLocLocked, setIsLocLocked] = useState(false);

  // --- INITIALIZATION (Sync with Backend) ---
  useEffect(() => {
    const fetchState = async () => {
      try {
        // Check if structures already exist in the Profit Kernel
        const existingProd = await api.ontology.getStructure('PRODUCT');
        if (existingProd && existingProd.length > 0) {
            setProdLevels(existingProd);
            setIsProdLocked(true); 
        }

        const existingLoc = await api.ontology.getStructure('LOCATION');
        if (existingLoc && existingLoc.length > 0) {
            setLocLevels(existingLoc);
            setIsLocLocked(true);
        }
        
        // Auto-advance logic
        if (existingProd.length > 0 && (!existingLoc || existingLoc.length === 0)) {
            setStep(2);
        }
        if (existingProd.length > 0 && existingLoc && existingLoc.length > 0) {
            setStep(3);
        }

      } catch (e) {
        console.error("Wizard sync failed", e);
      }
    };
    fetchState();
  }, []);

  // --- HANDLERS ---

  const handleAddLevel = (setter: any, current: string[]) => {
    setter([...current, `Level ${current.length + 1}`]);
  };

  const handleLevelChange = (setter: any, current: string[], index: number, val: string) => {
    const next = [...current];
    next[index] = val;
    setter(next);
  };

  const handleRemoveLevel = (setter: any, current: string[], index: number) => {
    if (current.length <= 1) {
        alert("You must have at least one hierarchy level.");
        return; 
    }
    setter(current.filter((_, i) => i !== index));
  };

  const saveProductStructure = async () => {
    if (prodLevels.some(l => !l.trim())) {
        alert("Please name all levels.");
        return;
    }
    setLoading(true);
    try {
      // Calls POST /ontology/structure (Backend Memory)
      await api.ontology.defineStructure(prodLevels);
      setIsProdLocked(true);
      setStep(2);
    } catch (e) {
      alert("Failed to save Product Structure. Check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  const saveLocationStructure = async () => {
    if (locLevels.some(l => !l.trim())) {
        alert("Please name all levels.");
        return;
    }
    setLoading(true);
    try {
      // Calls POST /ontology/structure (Backend Memory)
      await api.ontology.defineLocationStructure(locLevels);
      setIsLocLocked(true);
      setStep(3); 
    } catch (e) {
      alert("Failed to save Location Structure.");
    } finally {
      setLoading(false);
    }
  };

  // --- UI RENDERER ---
  
  const renderLevelInput = (levels: string[], setLevels: any, locked: boolean) => (
    <div className="space-y-3">
      {levels.map((lvl, idx) => (
        <div 
            key={idx} 
            className="flex items-center gap-3 animate-in fade-in slide-in-from-left-4 duration-300" 
            style={{animationDelay: `${idx * 50}ms`}}
        >
            <div className="flex flex-col items-center gap-1 w-16">
                <span className={`text-[10px] font-bold uppercase tracking-wider ${locked ? 'text-slate-300' : 'text-indigo-400'}`}>
                    Level {idx + 1}
                </span>
            </div>
            
            <div className="flex-1 relative">
                <input 
                    value={lvl}
                    disabled={locked}
                    onChange={(e) => handleLevelChange(setLevels, levels, idx, e.target.value)}
                    className={`w-full p-4 border rounded-xl font-bold text-lg shadow-sm transition-all outline-none focus:ring-2 focus:ring-indigo-500 ${
                        locked 
                        ? 'bg-slate-50 text-slate-500 border-slate-200 cursor-not-allowed' 
                        : 'bg-white border-slate-300 text-slate-800'
                    }`}
                    placeholder={`e.g. ${idx === 0 ? 'Group' : 'Item'}`}
                />
                
                {/* Visual Connector Line */}
                {idx < levels.length - 1 && (
                    <div className="absolute left-6 -bottom-6 w-0.5 h-6 bg-slate-200 -z-10"></div>
                )}
            </div>

            {!locked && (
                <button 
                    onClick={() => handleRemoveLevel(setLevels, levels, idx)} 
                    className="p-3 text-slate-300 hover:text-rose-500 hover:bg-rose-50 rounded-xl transition-colors"
                    title="Remove Level"
                >
                    <Trash2 size={18}/>
                </button>
            )}
        </div>
      ))}
      
      {!locked && (
          <div className="pl-[4.5rem]">
            <button 
                onClick={() => handleAddLevel(setLevels, levels)} 
                className="flex items-center gap-2 text-sm font-bold text-indigo-600 hover:text-indigo-800 hover:bg-indigo-50 px-4 py-2 rounded-lg transition-all mt-2 border border-dashed border-indigo-200 hover:border-indigo-300 w-full justify-center"
            >
                <Plus size={16} /> Add Hierarchy Level
            </button>
          </div>
      )}
    </div>
  );

  return (
    <div className="max-w-4xl mx-auto py-12 px-6 h-full overflow-y-auto custom-scrollbar">
      
      <div className="mb-10 text-center">
        <h1 className="text-3xl font-black text-slate-900 tracking-tight">System Configuration</h1>
        <p className="text-slate-500 mt-2 text-lg">
            Define the ontology for the <strong>{cartridge?.name || 'Enterprise'}</strong> domain.
        </p>
      </div>

      {/* PROGRESS STEPPER */}
      <div className="flex justify-center mb-12">
          {[1, 2, 3].map(s => (
              <div key={s} className={`flex items-center ${s < 3 ? 'w-32' : ''}`}>
                  <div className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-lg transition-all border-4 ${
                      step >= s || (s === 1 && isProdLocked) || (s === 2 && isLocLocked)
                      ? 'bg-indigo-600 border-indigo-100 text-white shadow-lg shadow-indigo-200' 
                      : 'bg-white border-slate-200 text-slate-300'
                  }`}>
                      {(s === 1 && isProdLocked) || (s === 2 && isLocLocked) ? <CheckCircle size={24}/> : s}
                  </div>
                  {s < 3 && <div className={`flex-1 h-1 mx-2 rounded ${step > s ? 'bg-indigo-600' : 'bg-slate-200'}`} />}
              </div>
          ))}
      </div>

      <div className="bg-white rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-200 overflow-hidden min-h-[500px]">
        
        {/* --- STEP 1: PRODUCT HIERARCHY --- */}
        {step === 1 && (
            <div className="p-10 animate-in slide-in-from-right duration-500">
                <div className="flex items-start justify-between mb-8">
                    <div className="flex gap-5">
                        <div className="p-4 bg-indigo-50 text-indigo-600 rounded-2xl h-fit border border-indigo-100"><Layers size={32}/></div>
                        <div>
                            <h2 className="text-2xl font-bold text-slate-900">{termProduct} Hierarchy</h2>
                            <p className="text-slate-500 mt-1">Define levels from broadest to specific (e.g. Division → {termProduct}).</p>
                        </div>
                    </div>
                    {isProdLocked && (
                        <button 
                            onClick={() => setIsProdLocked(false)} 
                            className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-xl text-sm font-bold transition-colors"
                        >
                            <Edit2 size={16}/> Edit
                        </button>
                    )}
                </div>
                
                {renderLevelInput(prodLevels, setProdLevels, isProdLocked)}

                <div className="mt-12 pt-6 border-t border-slate-100 flex justify-end">
                    {isProdLocked ? (
                        <button onClick={() => setStep(2)} className="flex items-center gap-2 bg-indigo-600 text-white px-8 py-4 rounded-xl font-bold hover:bg-indigo-700 shadow-lg shadow-indigo-200 transition-all hover:-translate-y-0.5">
                            Continue to {termLocation}s <ArrowRight size={20}/>
                        </button>
                    ) : (
                        <button onClick={saveProductStructure} disabled={loading} className="bg-slate-900 text-white px-8 py-4 rounded-xl font-bold hover:bg-slate-800 shadow-lg transition-all flex items-center gap-2">
                            {loading ? 'Saving...' : <>Save Hierarchy <Save size={20}/></>}
                        </button>
                    )}
                </div>
            </div>
        )}

        {/* --- STEP 2: LOCATION HIERARCHY --- */}
        {step === 2 && (
            <div className="p-10 animate-in slide-in-from-right duration-500">
                <div className="flex items-start justify-between mb-8">
                    <div className="flex gap-5">
                        <div className="p-4 bg-emerald-50 text-emerald-600 rounded-2xl h-fit border border-emerald-100"><MapPin size={32}/></div>
                        <div>
                            <h2 className="text-2xl font-bold text-slate-900">{termLocation} Hierarchy</h2>
                            <p className="text-slate-500 mt-1">Define levels for your physical network (e.g. Region → {termLocation}).</p>
                        </div>
                    </div>
                    {isLocLocked && (
                        <button 
                            onClick={() => setIsLocLocked(false)} 
                            className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-600 rounded-xl text-sm font-bold transition-colors"
                        >
                            <Edit2 size={16}/> Edit
                        </button>
                    )}
                </div>

                {renderLevelInput(locLevels, setLocLevels, isLocLocked)}

                <div className="mt-12 pt-6 border-t border-slate-100 flex justify-between items-center">
                    <button onClick={() => setStep(1)} className="text-slate-400 font-bold hover:text-slate-600 px-4">Back</button>
                    {isLocLocked ? (
                        <button onClick={() => setStep(3)} className="flex items-center gap-2 bg-emerald-600 text-white px-8 py-4 rounded-xl font-bold hover:bg-emerald-700 shadow-lg shadow-emerald-200 transition-all hover:-translate-y-0.5">
                            Finish Setup <ArrowRight size={20}/>
                        </button>
                    ) : (
                        <button onClick={saveLocationStructure} disabled={loading} className="bg-slate-900 text-white px-8 py-4 rounded-xl font-bold hover:bg-slate-800 shadow-lg transition-all flex items-center gap-2">
                            {loading ? 'Saving...' : <>Confirm Structure <Save size={20}/></>}
                        </button>
                    )}
                </div>
            </div>
        )}

        {/* --- STEP 3: CONFIRMATION --- */}
        {step === 3 && (
            <div className="p-16 text-center animate-in zoom-in-95 duration-500 flex flex-col items-center justify-center h-full">
                <div className="w-24 h-24 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center mb-6 shadow-emerald-200 shadow-lg">
                    <CheckCircle size={48} />
                </div>
                <h2 className="text-3xl font-black text-slate-900">Configuration Locked</h2>
                <p className="text-slate-500 mt-4 mb-10 max-w-lg text-lg leading-relaxed">
                    Your Enterprise Ontology is ready. The Data Plane is now configured to recognize 
                    <strong> {prodLevels.join(' > ')}</strong> and 
                    <strong> {locLevels.join(' > ')}</strong>.
                </p>
                
                <div className="flex gap-4">
                    <button 
                        onClick={() => { setIsLocLocked(false); setStep(2); }} 
                        className="px-6 py-3 text-slate-500 font-bold hover:bg-slate-50 rounded-xl border border-transparent hover:border-slate-200 transition-all"
                    >
                        Review Config
                    </button>
                    
                    <button 
                        onClick={() => onNavigate('data')} 
                        className="px-8 py-4 bg-indigo-600 text-white font-bold rounded-xl shadow-xl shadow-indigo-200 flex items-center gap-2 animate-pulse hover:bg-indigo-700 hover:scale-105 transition-all"
                    >
                        Go to Data Plane <ArrowRight size={18} />
                    </button>
                </div>
            </div>
        )}

      </div>
    </div>
  );
};
