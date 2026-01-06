import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar.tsx';
import { OnboardingWizard } from './views/OnboardingWizard.tsx'; 
import { CommandCenter } from './views/CommandCenter.tsx'; 
import { DebateConsole } from './views/DebateConsole.tsx'; 
import DecisionWorkbench from './views/DecisionWorkbench.tsx';
import IntelligenceView from './views/IntelligenceView.tsx';
import DataPlaneView from './views/DataPlaneView.tsx';
import GovernanceView from './views/GovernanceView.tsx';
import LearningLoopView from './views/LearningLoopView.tsx';
import CartridgeRegistry from './views/CartridgeRegistry.tsx';
import JarvisAssistant from './components/JarvisAssistant.tsx';
import AuctobotConsole from './components/AuctobotConsole.tsx';
import SettingsModal from './components/SettingsModal.tsx';
// --- NEW ENGINE IMPORTS ---
import { AdvancedPlanningGrid } from './components/AdvancedPlanningGrid.tsx';
import { PlanSetupModal } from './components/PlanSetupModal.tsx';
// --------------------------
import { AssortmentMatrix } from './components/AssortmentMatrix.tsx';
import { ForecastWorkbench } from './components/ForecastWorkbench.tsx';
import { AllocationGrid } from './components/AllocationGrid.tsx';
import { 
  RETAIL_CARTRIDGE_V1, LOGISTICS_CARTRIDGE 
} from './constants.ts';
import { 
  DataContract, SKUData, RevenuePoint, ModelMetric, CrossValidationResult, 
  Plane, AuditLogEntry, LearningEvent, DecisionState, CartridgeManifest, 
  EnterpriseConfig, PlanningContext 
} from './types.ts';
import { api } from './services/api.ts';
import { Layers, Calendar, ChevronRight, Loader2, Settings } from 'lucide-react';

const App: React.FC = () => {
  // --- STATE (PRESERVED) ---
  const [activeCartridge, setActiveCartridge] = useState<CartridgeManifest | null>(null);
  const [currentView, setCurrentView] = useState('registry'); 
  const [isAuctobotOpen, setIsAuctobotOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [cartridges, setCartridges] = useState<CartridgeManifest[]>([
    RETAIL_CARTRIDGE_V1, 
    LOGISTICS_CARTRIDGE
  ]);
  
  // --- LIVE DATA STATE (PRESERVED) ---
  const [dataContracts, setDataContracts] = useState<DataContract[]>([]);
  const [skus, setSkus] = useState<SKUData[]>([]); 
  const [models, setModels] = useState<ModelMetric[]>([]); 
  const [alerts, setAlerts] = useState<any[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditLogEntry[]>([]);
  const [learningEvents, setLearningEvents] = useState<LearningEvent[]>([]);
  const [revenueData, setRevenueData] = useState<RevenuePoint[]>([]);
  const [cvStats, setCvStats] = useState<CrossValidationResult | null>(null);
  
  // --- PROFIT KERNEL STATE ---
  const [assortmentData, setAssortmentData] = useState<any[]>([]); 
  const [budgetContext, setBudgetContext] = useState<SKUData[]>([]);
  
  // --- CONFIGURATION STATE ---
  const [enterpriseConfig, setEnterpriseConfig] = useState<EnterpriseConfig | null>(null);
  const [isInitializing, setIsInitializing] = useState(false);

  // --- NEW PLANSMART STATE ---
  const [planningContext, setPlanningContext] = useState<PlanningContext | null>(null);
  const [isPlanModalOpen, setIsPlanModalOpen] = useState(false);

  // 1. BOOT SEQUENCE (PRESERVED)
  useEffect(() => {
    const bootOS = async () => {
        try {
            const config = await api.getEnterpriseConfig(); 
            setEnterpriseConfig(config);
            
            setAlerts([{
                id: 'SYS-INIT',
                plane: Plane.GOVERNANCE,
                message: 'Auctorian Profit Kernel v10.19 Online.',
                severity: 'low',
                timestamp: 'Just now'
            }]);
        } catch (err) {
            console.warn("Kernel Boot Warning:", err);
        }
    };
    bootOS();
  }, []);

  // 2. LIVE DATA POLL (PRESERVED)
  useEffect(() => {
    if (!activeCartridge) return;

    const fetchData = async () => {
        try {
            // FIX: Use api.graph.getObjects instead of api.getGraphObjects
            const products = await api.graph.getObjects('PRODUCT'); 
            
            // FIX: Add safety check for array response
            if (Array.isArray(products)) {
                setSkus(products.map((p: any) => ({
                    id: p.obj_id,
                    name: p.name,
                    // FIX: Safe JSON parsing for attributes
                    category: p.attributes ? (typeof p.attributes === 'string' ? JSON.parse(p.attributes).category : p.attributes.category) : 'General',
                    cost: p.attributes ? (typeof p.attributes === 'string' ? JSON.parse(p.attributes).cost : p.attributes.cost) : 0,
                    price: 0,
                    inventory: 0 
                })));
            }

            if (api.ml && api.ml.getMetrics) {
                const mlMetrics = await api.ml.getMetrics();
                if (mlMetrics) setModels([mlMetrics]);
            }

            const stats = await api.getSystemStats(); 
            if (stats) {
                setAuditLogs([{
                    id: 'LOG-LIVE',
                    user: 'System',
                    action: 'DECISION_CYCLE',
                    target: `${stats.decisions || 0} Decisions Processed`,
                    timestamp: stats.last_updated || new Date().toISOString(),
                    status: 'Success'
                }]);
            }

        } catch (e) {
            console.error("Live Data Fetch Failed", e);
        }
    };

    fetchData();
    const interval = setInterval(fetchData, 30000); 
    return () => clearInterval(interval);
  }, [activeCartridge]);

  // 3. ASSORTMENT ENGINE HOOK (PRESERVED)
  useEffect(() => {
    if (currentView === 'assortment-planner') {
        const loadAssortment = async () => {
            try {
                const data = await api.orchestration.getAssortment(); 
                setAssortmentData(data);
            } catch (e) {
                console.error("Assortment Engine Offline", e);
            }
        };
        loadAssortment();
    }
  }, [currentView]);

  // 4. SCOPE GATING (UPDATED)
  useEffect(() => {
    if (currentView === 'merch-financial-planner' && !planningContext) {
        setIsPlanModalOpen(true);
    }
  }, [currentView, planningContext]);

  // --- HANDLERS ---
  const handleMountCartridge = (id: string) => {
    const target = cartridges.find(c => c.id === id);
    if (target) {
        setCartridges(prev => prev.map(c => ({
            ...c,
            status: c.id === id ? 'ACTIVE' : (c.status === 'ACTIVE' ? 'INSTALLED' : c.status)
        })) as CartridgeManifest[]);
        
        setActiveCartridge(target);
        setCurrentView('command-center'); 
    }
  };

  const handleNavigate = (view: string) => {
    setCurrentView(view);
  };

  const handlePlanSubmit = (context: PlanningContext) => {
      setPlanningContext(context);
      setIsPlanModalOpen(false);
  };

  const handleRunAutoML = async () => {
      try {
          await api.ml.train(); 
          const newMetrics = await api.ml.getMetrics();
          setModels([newMetrics]);
      } catch (err) {
          console.error("AutoML Failed", err);
      }
  };

  const appState = {
    view: currentView,
    activeAlerts: alerts,
    activeModels: models,
    dataStats: { joinedSKUs: skus.length },
    cartridge: activeCartridge
  };

  return (
    <div className="flex h-screen bg-slate-900 font-sans text-slate-100 overflow-hidden">
      
      {activeCartridge && (
          <Sidebar 
            currentView={currentView} 
            onChangeView={handleNavigate} 
            onOpenAuctobot={() => setIsAuctobotOpen(true)}
            onOpenSettings={() => setIsSettingsOpen(true)}
            onLoadDemo={() => {}} 
            onResetData={() => window.location.reload()}
            isDemoMode={false}
            activeCartridge={activeCartridge}
          />
      )}

      <main className="flex-1 flex flex-col relative bg-slate-50 text-slate-900">
        
        <header className="h-16 bg-white/80 backdrop-blur-md border-b border-slate-200 flex items-center px-8 justify-between shrink-0 z-10">
          <div className="flex items-center text-sm font-medium text-slate-500">
             <span className="text-indigo-600 font-bold tracking-tight">AUCTORIAN</span>
             <span className="mx-2 opacity-30">/</span>
             <span className="text-slate-900 font-bold">
                 {activeCartridge ? activeCartridge.name.toUpperCase() : 'KERNEL BOOTLOADER'}
             </span>
          </div>
          <div className="flex items-center space-x-6">
            
            {activeCartridge && (
                <button 
                    onClick={() => setCurrentView('onboarding')}
                    className="flex items-center gap-2 text-xs font-bold text-indigo-600 bg-indigo-50 px-3 py-1.5 rounded-lg hover:bg-indigo-100 transition-colors border border-indigo-100"
                >
                    <Settings size={14} /> Ontology Setup
                </button>
            )}

            <div className="flex flex-col items-end">
              <span className="text-[10px] font-bold text-indigo-600 tracking-widest uppercase">Profit Kernel Active</span>
              <span className="text-[9px] text-slate-400 font-mono">NODE: PRIMARY</span>
            </div>
            {activeCartridge ? (
               <div className="h-10 w-10 rounded-xl bg-slate-900 flex items-center justify-center text-white font-bold border border-slate-700 shadow-xl">
                 <span className="text-xs">OS</span>
               </div>
            ) : (
                <div className="h-10 w-10 rounded-xl bg-indigo-600 flex items-center justify-center text-white font-bold border border-indigo-500 shadow-xl animate-pulse">
                 <Loader2 className="animate-spin" size={20}/>
               </div>
            )}
          </div>
        </header>

        <div className="flex-1 overflow-auto custom-scrollbar bg-[#F8FAFC]">
          <div className="h-full">
            
            {currentView === 'registry' && (
                <div className="p-10 max-w-7xl mx-auto">
                    <CartridgeRegistry 
                        cartridges={cartridges} 
                        onMount={handleMountCartridge} 
                        onInstall={() => {}} 
                    />
                </div>
            )}

            {currentView === 'command-center' && activeCartridge && (
               <div className="p-6 h-full">
                 <CommandCenter /> 
               </div>
            )}

            {currentView === 'debate' && (
                <DebateConsole />
            )}
            
            {/* MERCH PLANNER VIEW */}
            {currentView === 'merch-financial-planner' && activeCartridge && (
                <div className="p-6 h-full flex flex-col space-y-6 max-w-7xl mx-auto animate-in fade-in">
                    {planningContext ? (
                        <>
                            <div className="flex items-center justify-between pb-4 border-b border-slate-200">
                                <div>
                                    <h2 className="text-xl font-bold text-slate-900">Merchandise Financial Planner</h2>
                                    <div className="flex items-center space-x-2 text-sm text-slate-500 mt-1">
                                        <Layers size={14} className="text-indigo-600"/>
                                        <span className="font-mono bg-slate-100 px-2 rounded text-xs">
                                            {planningContext.aggregateLevel} &gt; {planningContext.anchorLevel}
                                        </span>
                                        <span className="mx-2 text-slate-300">|</span>
                                        <Calendar size={14} className="text-slate-400" />
                                        <span className="font-medium text-slate-600">
                                            {planningContext.startYear} - {planningContext.startYear + planningContext.horizonYears} ({planningContext.timeBucket})
                                        </span>
                                    </div>
                                </div>
                                <button 
                                    onClick={() => { setPlanningContext(null); setIsPlanModalOpen(true); }}
                                    className="text-xs font-bold text-indigo-600 hover:text-indigo-700 bg-indigo-50 px-3 py-2 rounded-lg transition-colors"
                                >
                                    Reconfigure Scope
                                </button>
                            </div>

                            <div className="flex-1 min-h-[600px]">
                                <AdvancedPlanningGrid context={planningContext} />
                            </div>
                        </>
                    ) : (
                        <div className="h-full flex items-center justify-center text-slate-400">
                            <div className="text-center">
                                <Loader2 className="animate-spin mx-auto mb-2" />
                                <p>Waiting for Plan Configuration...</p>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {currentView === 'assortment-planner' && activeCartridge && (
                <div className="p-6 h-full flex flex-col space-y-6 max-w-7xl mx-auto">
                    <div className="flex-1 min-h-[400px]">
                        <AssortmentMatrix 
                            data={assortmentData} 
                            onSimulate={() => {}} 
                        />
                    </div>
                </div>
            )}

            {currentView === 'intelligence' && activeCartridge && (
               <div className="p-10 max-w-7xl mx-auto">
                    <IntelligenceView />
               </div>
            )}

            {currentView === 'governance' && activeCartridge && (
              <div className="p-10 max-w-7xl mx-auto">
                <GovernanceView 
                  logs={auditLogs} 
                  skus={skus} 
                  activeSOPs={activeCartridge?.sopConfig}
                />
              </div>
            )}
            
            {currentView === 'learning' && activeCartridge && <div className="p-10 max-w-7xl mx-auto"><LearningLoopView events={learningEvents} revenueData={revenueData} championAccuracy={models.length > 0 ? models[0].accuracy : 0} totalValue={0} /></div>}
            
            {currentView === 'data' && <div className="p-8 h-full"><DataPlaneView cartridge={activeCartridge} /></div>}
            {currentView === 'onboarding' && <OnboardingWizard onNavigate={handleNavigate} cartridge={activeCartridge} />}
            {currentView === 'allocation-hub' && <div className="p-6 h-full max-w-7xl mx-auto"><AllocationGrid /></div>}
            
            {currentView === 'nucleus-forecast' && <div className="p-6 h-full max-w-7xl mx-auto"><ForecastWorkbench scope={null} /></div>}
            
            {(currentView === 'price-optimizer' || currentView === 'markdown-manager') && <DecisionWorkbench contextTitle={currentView === 'price-optimizer' ? "Pricing" : "Markdown"} endpoint={`/orchestration/${currentView === 'price-optimizer' ? 'price_optimize' : 'markdown'}`} />}

            {!activeCartridge && currentView !== 'registry' && (
                <div className="h-full flex flex-col items-center justify-center">
                    <h2 className="text-2xl font-bold text-slate-400">Kernel Idle</h2>
                    <button onClick={() => setCurrentView('registry')} className="mt-4 text-indigo-600 font-bold hover:underline">Return to Registry</button>
                </div>
            )}
          </div>
        </div>

        <JarvisAssistant contextData={appState} />
        <AuctobotConsole isOpen={isAuctobotOpen} onClose={() => setIsAuctobotOpen(false)} />
        <SettingsModal isOpen={isSettingsOpen} onClose={() => setIsSettingsOpen(false)} />
        
        {/* MODAL (Safe Close Logic) */}
        <PlanSetupModal 
            isOpen={isPlanModalOpen}
            onClose={() => {
                setIsPlanModalOpen(false);
                if (!planningContext && currentView === 'merch-financial-planner') {
                    setCurrentView('command-center'); 
                }
            }}
            onSubmit={handlePlanSubmit}
            config={enterpriseConfig || { hierarchy: [], timeBuckets: [] }} 
        />

        {isInitializing && (
             <div className="fixed inset-0 z-[200] bg-slate-900/80 backdrop-blur-md flex flex-col items-center justify-center space-y-6">
                <div className="w-24 h-24 border-8 border-indigo-500/20 border-t-indigo-500 rounded-full animate-spin"></div>
                <div className="text-center">
                    <h2 className="text-2xl font-bold text-white tracking-widest uppercase">Initializing Remote Context</h2>
                    <p className="text-indigo-300 font-mono text-xs mt-2">SECURE_TUNNEL: ACTIVE • SCHEDULING_RESOURCES...</p>
                </div>
             </div>
        )}
      </main>
    </div>
  );
};

export default App;
