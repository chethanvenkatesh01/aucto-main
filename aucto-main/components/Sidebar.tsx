import React from 'react';
import { 
  LayoutDashboard, 
  Database, 
  Brain, 
  Scale, 
  GitBranch, 
  TrendingUp, 
  Truck, 
  Tags, 
  ShoppingCart, 
  PieChart, 
  Activity, 
  Zap,
  Settings,
  LogOut,
  ChevronRight,
  Target,
  Gavel,           // <--- NEW: For Council Chamber
  Layers,
  ShieldCheck,
  RefreshCw
} from 'lucide-react';
import { CartridgeManifest } from '../types';

interface SidebarProps {
  currentView: string;
  onChangeView: (view: string) => void;
  onOpenAuctobot: () => void;
  onOpenSettings: () => void;
  onLoadDemo: () => void;
  onResetData: () => void;
  isDemoMode: boolean;
  activeCartridge: CartridgeManifest | null;
}

const Sidebar: React.FC<SidebarProps> = ({ 
  currentView, 
  onChangeView, 
  activeCartridge,
  onOpenAuctobot,
  onOpenSettings,
  onResetData
}) => {

  // --- NAVIGATION HIERARCHY ---
  const menuGroups = [
    {
      label: "Overview",
      items: [
        { id: 'command-center', label: 'Command Center', icon: LayoutDashboard },
        { id: 'intelligence', label: 'Intelligence Plane', icon: Brain },
      ]
    },
    {
      label: "Strategy & Plan",
      items: [
        { id: 'merch-financial-planner', label: 'Financial Plan', icon: Target },
        { id: 'assortment-planner', label: 'Assortment Strategy', icon: PieChart },
        { id: 'debate', label: 'Council Chamber', icon: Gavel, badge: '3' }, // <--- NEW ITEM
      ]
    },
    {
      label: "Execution",
      items: [
        { id: 'nucleus-forecast', label: 'Demand Forecast', icon: Activity },
        { id: 'price-optimizer', label: 'Price Optimizer', icon: Tags },
        { id: 'markdown-manager', label: 'Markdown Manager', icon: TrendingUp },
        { id: 'allocation-hub', label: 'Allocation Hub', icon: Truck },
      ]
    },
    {
      label: "Governance",
      items: [
        { id: 'data-plane', label: 'Data Contracts', icon: Database },
        { id: 'governance', label: 'Audit & Policy', icon: ShieldCheck },
        { id: 'learning-loop', label: 'Learning Loop', icon: RefreshCw },
      ]
    }
  ];

  return (
    <nav className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col h-full shrink-0 transition-all duration-300">
      
      {/* 1. HEADER BRANDING */}
      <div className="h-16 flex items-center px-6 border-b border-slate-800/50">
        <div className="flex items-center gap-2 text-indigo-500">
           <Layers size={24} strokeWidth={2.5} />
           <span className="text-lg font-black tracking-tight text-white">AUCTORIAN</span>
        </div>
      </div>

      {/* 2. NAVIGATION LIST */}
      <div className="flex-1 overflow-y-auto py-6 px-3 space-y-8 custom-scrollbar">
        {menuGroups.map((group, groupIdx) => (
          <div key={groupIdx}>
            <h3 className="px-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">
              {group.label}
            </h3>
            <div className="space-y-0.5">
              {group.items.map((item) => {
                const isActive = currentView === item.id;
                const Icon = item.icon;
                
                return (
                  <button
                    key={item.id}
                    onClick={() => onChangeView(item.id)}
                    className={`
                      w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200 group
                      ${isActive 
                        ? 'bg-indigo-600/10 text-indigo-400 shadow-[inset_3px_0_0_0_#6366f1]' 
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                      }
                    `}
                  >
                    <Icon 
                      size={18} 
                      className={isActive ? 'text-indigo-500' : 'text-slate-500 group-hover:text-slate-300'} 
                    />
                    
                    <span className="flex-1 text-left">{item.label}</span>
                    
                    {/* Optional Badge (e.g., for Active Conflicts) */}
                    {item.badge && (
                      <span className="bg-indigo-500 text-white text-[10px] font-bold px-1.5 py-0.5 rounded-md shadow-sm">
                        {item.badge}
                      </span>
                    )}

                    {isActive && <ChevronRight size={14} className="text-indigo-500/50" />}
                  </button>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* 3. FOOTER ACTIONS */}
      <div className="p-4 border-t border-slate-800 bg-slate-900/50 space-y-3">
        
        {/* Active Cartridge Indicator */}
        {activeCartridge && (
            <div className="px-3 py-2 bg-slate-800 rounded border border-slate-700 flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]"></div>
                <div className="flex-1 overflow-hidden">
                    <div className="text-[10px] font-bold text-slate-400 uppercase">Cartridge Active</div>
                    <div className="text-xs font-bold text-slate-200 truncate">{activeCartridge.name}</div>
                </div>
            </div>
        )}

        <div className="grid grid-cols-2 gap-2">
            <button 
                onClick={onOpenSettings}
                className="flex items-center justify-center gap-2 py-2 bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white rounded-lg transition-colors border border-slate-700"
                title="System Settings"
            >
                <Settings size={16} />
                <span className="text-xs font-bold">Config</span>
            </button>
            <button 
                onClick={onOpenAuctobot}
                className="flex items-center justify-center gap-2 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg transition-colors shadow-lg hover:shadow-indigo-500/20"
                title="Ask Jarvis"
            >
                <Brain size={16} />
                <span className="text-xs font-bold">Jarvis</span>
            </button>
        </div>

        <button 
            onClick={onResetData}
            className="w-full flex items-center justify-center gap-2 py-2 text-rose-500/70 hover:text-rose-400 hover:bg-rose-950/20 rounded-lg transition-all text-xs font-bold"
        >
            <LogOut size={14} />
            Reset Kernel
        </button>
      </div>
    </nav>
  );
};

export default Sidebar;
