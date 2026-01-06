
import React, { useState } from 'react';
import { HardDrive, Download, Play, Box, CheckCircle, Cpu, Loader2, Lock, LayoutGrid, Terminal } from 'lucide-react';
import { CartridgeManifest } from '../types';

interface CartridgeRegistryProps {
  cartridges: CartridgeManifest[];
  onMount: (id: string) => void;
  onInstall: (id: string) => void;
}

const CartridgeRegistry: React.FC<CartridgeRegistryProps> = ({ cartridges, onMount, onInstall }) => {
  const [mountingId, setMountingId] = useState<string | null>(null);

  const handleMountClick = (id: string) => {
    setMountingId(id);
    // Simulate the "Physical Mounting" delay of a cartridge injection
    setTimeout(() => {
        onMount(id);
        setMountingId(null);
    }, 2000); 
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex justify-between items-end border-b border-slate-200/60 pb-6">
        <div>
          <h2 className="text-3xl font-bold text-slate-900 tracking-tight">Cartridge Registry</h2>
          <p className="text-slate-500 mt-1">Manage Domain Intelligence Modules (DIMs)</p>
        </div>
        <div className="flex items-center space-x-2 text-xs font-mono bg-slate-900 text-white px-4 py-2 rounded-xl shadow-lg border border-slate-700">
            <Cpu size={14} className="text-indigo-400" />
            <span>KERNEL: <span className="text-indigo-400">v4.2.1-STABLE</span></span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {cartridges.map(cartridge => (
          <div key={cartridge.id} className={`relative group rounded-3xl border transition-all duration-300 flex flex-col overflow-hidden ${
              cartridge.status === 'ACTIVE' 
              ? 'bg-slate-900 border-indigo-500 shadow-2xl shadow-indigo-500/20 scale-[1.02]' 
              : 'bg-white border-slate-200 hover:border-indigo-300 hover:shadow-xl'
          }`}>
            
            {/* Cartridge Header / Art */}
            <div className={`h-32 p-6 flex items-start justify-between relative overflow-hidden ${
                cartridge.status === 'ACTIVE' ? 'bg-gradient-to-br from-indigo-900 to-slate-900' : 'bg-slate-50'
            }`}>
                 <div className="relative z-10">
                    <div className={`w-12 h-12 rounded-xl flex items-center justify-center text-white shadow-lg mb-4 ${
                         cartridge.status === 'ACTIVE' ? 'bg-indigo-500' : 'bg-slate-900'
                    }`}>
                        <Box size={24} />
                    </div>
                    <span className={`px-2 py-1 rounded text-[10px] font-bold uppercase tracking-widest border ${
                        cartridge.status === 'ACTIVE' 
                        ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' 
                        : cartridge.status === 'INSTALLED'
                        ? 'bg-indigo-50 text-indigo-600 border-indigo-100'
                        : 'bg-slate-200 text-slate-500 border-slate-300'
                    }`}>
                        {cartridge.status}
                    </span>
                 </div>
                 {/* Decorative background pattern */}
                 <div className="absolute right-0 top-0 opacity-10 transform translate-x-1/4 -translate-y-1/4">
                    <LayoutGrid size={160} />
                 </div>
            </div>

            {/* Cartridge Body */}
            <div className="p-6 flex-1 flex flex-col">
                <div className="mb-6">
                    <h3 className={`text-lg font-bold mb-1 ${cartridge.status === 'ACTIVE' ? 'text-white' : 'text-slate-900'}`}>
                        {cartridge.name}
                    </h3>
                    <p className={`text-xs leading-relaxed ${cartridge.status === 'ACTIVE' ? 'text-slate-400' : 'text-slate-500'}`}>
                        {cartridge.description}
                    </p>
                </div>

                {/* Specs Grid */}
                <div className={`grid grid-cols-2 gap-3 mb-6 p-3 rounded-xl border ${
                    cartridge.status === 'ACTIVE' ? 'bg-white/5 border-white/10' : 'bg-slate-50 border-slate-100'
                }`}>
                    <div>
                        <span className="text-[9px] uppercase text-slate-500 font-bold block">Agents</span>
                        <span className={`text-xs font-mono font-bold ${cartridge.status === 'ACTIVE' ? 'text-indigo-400' : 'text-slate-800'}`}>
                            {cartridge.capabilities.agents.length} Units
                        </span>
                    </div>
                    <div>
                        <span className="text-[9px] uppercase text-slate-500 font-bold block">SOP Vectors</span>
                        <span className={`text-xs font-mono font-bold ${cartridge.status === 'ACTIVE' ? 'text-indigo-400' : 'text-slate-800'}`}>
                            {cartridge.capabilities.sops.toLocaleString()} Nodes
                        </span>
                    </div>
                </div>

                <div className="mt-auto">
                    {cartridge.status === 'ACTIVE' ? (
                        <div className="w-full py-3 rounded-xl bg-emerald-500/10 border border-emerald-500/50 text-emerald-400 font-bold text-xs flex items-center justify-center space-x-2">
                             <Terminal size={16} className="animate-pulse" />
                             <span>RUNTIME MOUNTED</span>
                        </div>
                    ) : mountingId === cartridge.id ? (
                        <div className="w-full py-3 rounded-xl bg-indigo-50 text-indigo-600 font-bold text-xs flex items-center justify-center space-x-2 border border-indigo-100">
                             <Loader2 size={16} className="animate-spin" />
                             <span>INJECTING SCHEMA...</span>
                        </div>
                    ) : cartridge.status === 'INSTALLED' ? (
                        <button 
                            onClick={() => handleMountClick(cartridge.id)}
                            className="w-full py-3 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-bold text-xs flex items-center justify-center space-x-2 shadow-lg shadow-indigo-200 hover:-translate-y-0.5 transition-all"
                        >
                             <Play size={16} fill="currentColor" />
                             <span>MOUNT CARTRIDGE</span>
                        </button>
                    ) : (
                        <button 
                            onClick={() => onInstall(cartridge.id)}
                            className="w-full py-3 rounded-xl border-2 border-slate-200 text-slate-600 font-bold text-xs flex items-center justify-center space-x-2 hover:border-slate-400 hover:text-slate-800 transition-all"
                        >
                             <Download size={16} />
                             <span>INSTALL FROM REPO</span>
                        </button>
                    )}
                </div>
            </div>
            
            {/* Version Footer */}
            <div className={`px-6 py-3 border-t text-[10px] font-mono flex justify-between ${
                cartridge.status === 'ACTIVE' ? 'border-white/10 text-slate-500 bg-black/20' : 'border-slate-100 text-slate-400 bg-slate-50/50'
            }`}>
                <span>VER: {cartridge.version}</span>
                <span>CORE: {cartridge.coreVersion}</span>
            </div>

          </div>
        ))}
        
        {/* Placeholder for "Coming Soon" */}
        <div className="rounded-3xl border-2 border-dashed border-slate-200 flex flex-col items-center justify-center p-8 text-slate-400 min-h-[400px]">
            <Lock size={40} className="mb-4 opacity-50" />
            <span className="font-bold text-sm">Marketplace Locked</span>
            <span className="text-xs mt-1">Connect to Auctorian Cloud</span>
        </div>
      </div>
    </div>
  );
};

export default CartridgeRegistry;
