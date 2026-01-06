import React, { useState, useEffect } from 'react';
import { 
  Activity, ShieldAlert, Zap, Globe, 
  TrendingUp, Anchor, DollarSign, Package,
  BrainCircuit, History, ArrowRight
} from 'lucide-react';
import { api } from '../services/api';

// --- TYPES ---
interface DashboardStats {
  revenue: number;
  margin: number;
  wos: number;
  receipts: number;
  autonomy_score: number;
  system_split: { sys_1: number; sys_2: number; sys_3: number };
  active_debates: number;
}

// --- MOCK DATA GENERATOR (To Simulate Live Feed) ---
const generateMockFeed = () => [
  { id: 1, type: 'SYS1', msg: 'Auto-Replenished 500 units of Puma Suede (Low Risk)', time: '10:02 AM' },
  { id: 2, type: 'SYS3', msg: 'Debate Triggered: Price Hike on "Velocity Nitro" vs Market', time: '09:55 AM' },
  { id: 3, type: 'SYS2', msg: 'Profit Guardrail blocked Air Freight request (ROI < 15%)', time: '09:48 AM' },
  { id: 4, type: 'SYS1', msg: 'Routine Allocation: 200 units to Flagship Store', time: '09:30 AM' },
];

// --- 1. KPI CARD COMPONENT ---
const KpiCard = ({ title, value, sub, icon: Icon, color }: any) => (
  <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-start justify-between relative overflow-hidden group">
    <div className={`absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity ${color}`}>
        <Icon size={64} />
    </div>
    <div>
      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-1">{title}</p>
      <h3 className="text-2xl font-black text-slate-800 font-mono">{value}</h3>
      <p className={`text-xs font-bold mt-1 flex items-center gap-1 ${color.replace('bg-', 'text-').replace('/10', '')}`}>
        {sub}
      </p>
    </div>
  </div>
);

// --- 2. AUTONOMY RADAR (Custom SVG Visualization) ---
const AutonomyRadar = ({ split }: { split: { sys_1: number; sys_2: number; sys_3: number } }) => {
    const total = split.sys_1 + split.sys_2 + split.sys_3 || 1;
    const s1 = (split.sys_1 / total) * 100; // Auto
    const s2 = (split.sys_2 / total) * 100; // Human
    const s3 = (split.sys_3 / total) * 100; // Debate

    return (
        <div className="relative w-full h-64 flex items-center justify-center">
            {/* Background Rings */}
            <div className="absolute w-48 h-48 border border-slate-100 rounded-full"></div>
            <div className="absolute w-32 h-32 border border-slate-100 rounded-full"></div>
            <div className="absolute w-16 h-16 border border-slate-100 rounded-full"></div>
            
            {/* Axis Lines */}
            <div className="absolute w-full h-px bg-slate-100"></div>
            <div className="absolute h-full w-px bg-slate-100"></div>

            {/* The Triangle Data Visualization */}
            <svg className="w-full h-full absolute top-0 left-0 overflow-visible">
                <polygon 
                    points={`150,${150 - s1*1.5} ${150 + s2*1.3},${150 + s2*0.8} ${150 - s3*1.3},${150 + s3*0.8}`}
                    className="fill-indigo-500/20 stroke-indigo-500 stroke-2"
                />
                {/* Points */}
                <circle cx="150" cy={150 - s1*1.5} r="4" className="fill-emerald-500" />
                <circle cx={150 + s2*1.3} cy={150 + s2*0.8} r="4" className="fill-amber-500" />
                <circle cx={150 - s3*1.3} cy={150 + s3*0.8} r="4" className="fill-purple-500" />
            </svg>

            {/* Labels */}
            <div className="absolute top-4 text-xs font-bold text-emerald-600 bg-emerald-50 px-2 py-1 rounded-full">SYS 1: Auto ({s1.toFixed(0)}%)</div>
            <div className="absolute bottom-10 right-4 text-xs font-bold text-amber-600 bg-amber-50 px-2 py-1 rounded-full">SYS 2: Human ({s2.toFixed(0)}%)</div>
            <div className="absolute bottom-10 left-4 text-xs font-bold text-purple-600 bg-purple-50 px-2 py-1 rounded-full">SYS 3: Debate ({s3.toFixed(0)}%)</div>
        </div>
    );
};

export const CommandCenter: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [feed, setFeed] = useState(generateMockFeed());

  useEffect(() => {
    // Simulating fetching "Weaponized" stats
    const fetchStats = async () => {
        // In real app: const data = await api.governance.getDashboardStats();
        // Mocking for "Profit Kernel" demo visualization
        setStats({
            revenue: 12500000,
            margin: 48.2, // True Margin
            wos: 11.5,    // Weeks of Supply
            receipts: 450000, // OTB
            autonomy_score: 82,
            system_split: { sys_1: 65, sys_2: 20, sys_3: 15 },
            active_debates: 3
        });
    };
    fetchStats();
  }, []);

  if (!stats) return <div className="p-8 text-slate-400">Initializing Profit Kernel...</div>;

  return (
    <div className="h-full flex flex-col gap-6 p-6 bg-slate-50/50 overflow-y-auto">
      
      {/* 1. STATUS HEADER */}
      <div className="flex justify-between items-end">
        <div>
            <h1 className="text-2xl font-black text-slate-900 tracking-tight flex items-center gap-3">
                <Activity className="text-indigo-600"/> 
                Executive Situation Room
            </h1>
            <p className="text-sm text-slate-500 mt-1 font-medium">
                Profit Kernel v10.13 • <span className="text-emerald-600 font-bold">System Nominal</span>
            </p>
        </div>
        <div className="flex gap-3">
            <div className="bg-white px-4 py-2 rounded-lg border border-slate-200 shadow-sm flex flex-col items-end">
                <span className="text-[10px] font-bold text-slate-400 uppercase">Autonomy Score</span>
                <span className="text-xl font-black text-indigo-600">{stats.autonomy_score}%</span>
            </div>
            <div className="bg-white px-4 py-2 rounded-lg border border-slate-200 shadow-sm flex flex-col items-end">
                <span className="text-[10px] font-bold text-slate-400 uppercase">Active Debates</span>
                <span className="text-xl font-black text-purple-600">{stats.active_debates}</span>
            </div>
        </div>
      </div>

      {/* 2. KPI DECK (The Profit Ticker) */}
      <div className="grid grid-cols-4 gap-4">
        <KpiCard 
            title="Net Revenue (Proj)" 
            value={`$${(stats.revenue / 1000000).toFixed(1)}M`} 
            sub="+4.2% vs Target"
            icon={DollarSign}
            color="bg-emerald-500 text-emerald-600"
        />
        <KpiCard 
            title="True Margin (Post-Frt)" 
            value={`${stats.margin}%`} 
            sub="Landed Cost Adjusted"
            icon={Anchor} // Anchor cost
            color="bg-indigo-500 text-indigo-600"
        />
        <KpiCard 
            title="Inventory Health" 
            value={`${stats.wos} Wks`} 
            sub="Optimal Range (10-12)"
            icon={Package}
            color="bg-amber-500 text-amber-600"
        />
        <KpiCard 
            title="Open-to-Buy (Receipts)" 
            value={`$${(stats.receipts / 1000).toFixed(0)}k`} 
            sub="Cash Outlay Required"
            icon={Zap}
            color="bg-purple-500 text-purple-600"
        />
      </div>

      {/* 3. MAIN DASHBOARD AREA */}
      <div className="flex gap-6 min-h-[400px]">
        
        {/* LEFT: AUTONOMY RADAR */}
        <div className="w-1/3 bg-white rounded-xl border border-slate-200 shadow-sm p-6 flex flex-col">
            <h3 className="font-bold text-slate-800 flex items-center gap-2 mb-6">
                <BrainCircuit size={18} className="text-slate-400"/>
                Decision Architecture
            </h3>
            <div className="flex-1 flex items-center justify-center">
                <AutonomyRadar split={stats.system_split} />
            </div>
            <div className="mt-6 text-xs text-slate-500 text-center">
                System 3 (Debate) engagement is <span className="font-bold text-purple-600">High</span> due to recent tariff volatility.
            </div>
        </div>

        {/* RIGHT: LIVE NEURAL FEED */}
        <div className="flex-1 bg-white rounded-xl border border-slate-200 shadow-sm p-0 flex flex-col overflow-hidden">
            <div className="p-4 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
                <h3 className="font-bold text-slate-800 flex items-center gap-2">
                    <History size={18} className="text-slate-400"/>
                    Live Neural Feed
                </h3>
                <span className="flex h-2 w-2 relative">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
            </div>
            <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {feed.map((item) => (
                    <div key={item.id} className="flex gap-3 items-start group">
                        <div className="w-16 text-[10px] font-mono text-slate-400 mt-1">{item.time}</div>
                        <div className="flex-1">
                            <div className={`text-xs font-bold px-2 py-0.5 rounded w-fit mb-1 ${
                                item.type === 'SYS1' ? 'bg-emerald-100 text-emerald-700' :
                                item.type === 'SYS3' ? 'bg-purple-100 text-purple-700' :
                                'bg-amber-100 text-amber-700'
                            }`}>
                                {item.type === 'SYS1' ? 'AUTOPILOT' : item.type === 'SYS3' ? 'COUNCIL DEBATE' : 'PROFIT GUARD'}
                            </div>
                            <div className="text-sm text-slate-700 font-medium">
                                {item.msg}
                            </div>
                        </div>
                        <button className="text-slate-300 hover:text-indigo-600 opacity-0 group-hover:opacity-100 transition-opacity">
                            <ArrowRight size={16}/>
                        </button>
                    </div>
                ))}
            </div>
        </div>
      </div>

      {/* 4. GLOBAL STATUS BAR */}
      <div className="bg-slate-900 text-slate-300 rounded-lg p-3 flex justify-between items-center text-xs font-mono">
        <div className="flex gap-6">
            <span className="flex items-center gap-2"><div className="w-2 h-2 bg-emerald-500 rounded-full"></div> ML Engine: ONLINE (98.4% Acc)</span>
            <span className="flex items-center gap-2"><div className="w-2 h-2 bg-emerald-500 rounded-full"></div> Gemini-1.5: CONNECTED</span>
            <span className="flex items-center gap-2"><div className="w-2 h-2 bg-amber-500 rounded-full"></div> Inventory Feed: 400ms Latency</span>
        </div>
        <div>
            SESSION: ADMIN_ROOT • <span className="text-slate-500">SECURE CONNECTION</span>
        </div>
      </div>

    </div>
  );
};
