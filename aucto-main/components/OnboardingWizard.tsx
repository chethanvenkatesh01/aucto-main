import React, { useState } from 'react';
import { Layers, CheckCircle, ArrowRight, MapPin, Database, TrendingUp } from 'lucide-react';
import { api } from '../services/api';
import { IngestionMapper } from '../components/IngestionMapper';

interface WizardProps {
    onComplete: () => void;
}

export const OnboardingWizard: React.FC<WizardProps> = ({ onComplete }) => {
    const [step, setStep] = useState(1);
    
    // Structure State
    const [prodStructure, setProdStructure] = useState<string[]>(['Division', 'Department', 'SKU']);
    const [locStructure, setLocStructure] = useState<string[]>(['Region', 'District', 'Store']);
    
    const [statusMsg, setStatusMsg] = useState<string | null>(null);

    // --- STEP 1: PRODUCT HIERARCHY ---
    const handleSaveProdStructure = async () => {
        try {
            await api.ontology.defineStructure(prodStructure);
            setStep(2); 
            setStatusMsg(null);
        } catch (err: any) { setStatusMsg(err.message); }
    };

    // --- STEP 2: LOCATION HIERARCHY ---
    const handleSaveLocStructure = async () => {
        try {
            // We'll update API in next step to support this
            await api.ontology.defineLocationStructure(locStructure);
            setStep(3);
            setStatusMsg(null);
        } catch (err: any) { setStatusMsg(err.message); }
    };

    // --- HELPER: Dynamic Level Input ---
    const LevelInput = ({ list, setList, leafName, colorClass }: any) => (
        <div className="flex items-center gap-2 overflow-x-auto pb-6 mb-6">
            {list.map((lvl: string, idx: number) => (
                <React.Fragment key={idx}>
                    <input 
                        value={lvl}
                        onChange={(e) => {
                            const copy = [...list]; copy[idx] = e.target.value; setList(copy);
                        }}
                        disabled={lvl === leafName}
                        className={`px-4 py-3 rounded-lg font-bold text-center w-40 border-2 transition-all outline-none focus:ring-2 ${
                            lvl === leafName 
                                ? 'bg-slate-800 text-white border-slate-800' 
                                : `bg-white border-slate-200 text-slate-700 ${colorClass}`
                        }`}
                    />
                    {idx < list.length - 1 && <ArrowRight className="text-slate-300 shrink-0" />}
                </React.Fragment>
            ))}
            <button 
                onClick={() => { 
                    const copy = [...list]; 
                    // Insert before leaf
                    copy.splice(copy.length-1, 0, 'New Level'); 
                    setList(copy); 
                }}
                className="ml-4 px-3 py-1 text-xs font-bold text-indigo-600 bg-indigo-50 rounded hover:bg-indigo-100 transition-colors whitespace-nowrap"
            >
                + Insert Level
            </button>
        </div>
    );

    return (
        <div className="h-full bg-slate-50 p-10 overflow-auto font-sans">
            <div className="max-w-5xl mx-auto">
                {/* HEADER */}
                <div className="mb-10 text-center">
                    <h1 className="text-3xl font-bold text-slate-900">Enterprise Onboarding</h1>
                    <p className="text-slate-500 mt-2">Configure Domain Ontology & Data Plane</p>
                </div>

                {/* PROGRESS BAR */}
                <div className="flex justify-between mb-12 max-w-4xl mx-auto relative">
                    <div className="absolute top-4 left-0 w-full h-1 bg-slate-200 -z-0"></div>
                    {[1, 2, 3, 4, 5].map((s) => (
                        <div key={s} className="relative z-10 flex flex-col items-center gap-2 bg-slate-50 px-4">
                             <div className={`w-10 h-10 rounded-full flex items-center justify-center font-bold border-2 transition-all shadow-sm ${
                                step >= s ? 'bg-indigo-600 border-indigo-600 text-white' : 'bg-white border-slate-300 text-slate-300'
                             }`}>
                                {step > s ? <CheckCircle size={20}/> : s}
                             </div>
                             <span className={`text-[10px] font-bold uppercase tracking-wider ${step >= s ? 'text-indigo-600' : 'text-slate-400'}`}>
                                {s===1?'Prod Tree':s===2?'Loc Tree':s===3?'Catalog':s===4?'Network': 'Sales'}
                             </span>
                        </div>
                    ))}
                </div>

                <div className="bg-white rounded-2xl shadow-xl border border-slate-200 p-8 min-h-[500px]">
                    
                    {/* STEP 1: PRODUCT TREE */}
                    {step === 1 && (
                        <div className="animate-in fade-in slide-in-from-bottom-4">
                            <h2 className="text-xl font-bold mb-2 flex items-center gap-2 text-slate-800">
                                <Layers className="text-indigo-600"/> Product Hierarchy
                            </h2>
                            <p className="text-sm text-slate-500 mb-8">Define how you categorize items. <strong className="text-slate-700">SKU</strong> is enforced as the leaf.</p>
                            
                            <LevelInput 
                                list={prodStructure} 
                                setList={setProdStructure} 
                                leafName="SKU" 
                                colorClass="focus:border-indigo-500" 
                            />
                            
                            <div className="flex justify-end mt-12">
                                <button 
                                    onClick={handleSaveProdStructure} 
                                    className="bg-indigo-600 text-white px-8 py-3 rounded-xl font-bold hover:bg-indigo-700 transition-all shadow-lg shadow-indigo-200"
                                >
                                    Save Product Tree
                                </button>
                            </div>
                        </div>
                    )}

                    {/* STEP 2: LOCATION TREE */}
                    {step === 2 && (
                        <div className="animate-in fade-in slide-in-from-bottom-4">
                            <h2 className="text-xl font-bold mb-2 flex items-center gap-2 text-slate-800">
                                <MapPin className="text-emerald-600"/> Location Hierarchy
                            </h2>
                            <p className="text-sm text-slate-500 mb-8">Define your physical network. <strong className="text-slate-700">Store</strong> is enforced as the leaf.</p>
                            
                            <LevelInput 
                                list={locStructure} 
                                setList={setLocStructure} 
                                leafName="Store" 
                                colorClass="focus:border-emerald-500" 
                            />

                            <div className="flex justify-end mt-12">
                                <button 
                                    onClick={handleSaveLocStructure} 
                                    className="bg-emerald-600 text-white px-8 py-3 rounded-xl font-bold hover:bg-emerald-700 transition-all shadow-lg shadow-emerald-200"
                                >
                                    Save Location Tree
                                </button>
                            </div>
                        </div>
                    )}

                    {/* STEP 3: CATALOG INGEST */}
                    {step === 3 && (
                        <IngestionMapper 
                            title="Ingest Product Catalog"
                            description={`Upload CSV containing: ${prodStructure.join(', ')}`}
                            targetFields={prodStructure}
                            optionalFields={['Cost', 'Price', 'Color', 'Size', 'Brand', 'Supplier']}
                            onSubmit={(f, m) => api.ontology.ingestCatalog(f, m)}
                            onComplete={() => setStep(4)}
                        />
                    )}

                    {/* STEP 4: NETWORK INGEST */}
                    {step === 4 && (
                        <IngestionMapper 
                            title="Ingest Store Network"
                            description={`Upload CSV containing: ${locStructure.join(', ')}`}
                            targetFields={locStructure}
                            optionalFields={['City', 'State', 'SqFt', 'Open_Date', 'Manager']}
                            onSubmit={(f, m) => api.ontology.ingestLocations(f, m)}
                            onComplete={() => setStep(5)}
                        />
                    )}

                    {/* STEP 5: SALES INGEST */}
                    {step === 5 && (
                        <IngestionMapper 
                            title="Ingest Sales History"
                            description="Upload Transaction Data. Map to Date, SKU, Store, and Metrics."
                            targetFields={['Date', 'SKU', 'Store', 'Qty']}
                            optionalFields={['Revenue', 'Cost', 'Currency', 'Discount']}
                            onSubmit={(f, m) => api.ontology.ingestSales(f, m)}
                            onComplete={onComplete}
                        />
                    )}

                    {statusMsg && (
                         <div className="mt-6 p-4 bg-red-50 text-red-700 rounded-xl border border-red-100 font-medium">
                            Error: {statusMsg}
                         </div>
                    )}

                </div>
            </div>
        </div>
    );
};

export default OnboardingWizard;