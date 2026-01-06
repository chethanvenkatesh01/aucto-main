import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Activity, RefreshCw, GitMerge, ArrowUpRight, TrendingUp } from 'lucide-react';
import MetricCard from '../components/MetricCard';
import { LearningEvent, RevenuePoint } from '../types';

interface LearningLoopViewProps {
  events: LearningEvent[];
  revenueData: RevenuePoint[];
  championAccuracy: number;
  totalValue: number;
}

const LearningLoopView: React.FC<LearningLoopViewProps> = ({ events, revenueData, championAccuracy, totalValue }) => {
  
  const chartData = revenueData.map(d => ({
    ...d,
    lower: d.projected * 0.9,
    upper: d.projected * 1.1
  }));

  const retrainCount = events.filter(e => e.type === 'Retrain').length;
  const driftCount = events.filter(e => e.type === 'Drift').length;

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="flex justify-between items-end border-b border-slate-200/60 pb-6">
        <div>
          <h2 className="text-3xl font-bold text-slate-900 tracking-tight">Learning Plane</h2>
          <p className="text-slate-500 mt-1">Outcome Tracking & Feedback Loop</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <MetricCard title="Champion Accuracy" value={championAccuracy > 0 ? `${championAccuracy.toFixed(1)}%` : '--'} change="Model Health" trend="up" icon={Activity} />
        <MetricCard title="Auto-Retrains" value={retrainCount.toString()} change="Active Cycles" trend="neutral" icon={RefreshCw} />
        <MetricCard title="Drift Alerts" value={driftCount.toString()} change="Monitored" trend={driftCount > 0 ? 'down' : 'neutral'} icon={GitMerge} />
        <MetricCard title="Value Captured" value={`$${(totalValue / 1000).toFixed(1)}k`} change="Cumulative" trend="up" icon={ArrowUpRight} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        <div className="lg:col-span-2 bg-white rounded-2xl border border-slate-200/60 shadow-sm p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="font-bold text-slate-900">Actuals vs. Forecast</h3>
            <span className="text-xs font-bold text-indigo-600 bg-indigo-50 px-3 py-1 rounded-full flex items-center border border-indigo-100">
                <TrendingUp size={12} className="mr-1.5" /> Performance
            </span>
          </div>
          <div className="h-80 w-full">
            {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                    <defs>
                    <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#818cf8" stopOpacity={0.2}/>
                        <stop offset="95%" stopColor="#818cf8" stopOpacity={0}/>
                    </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                    <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 11}} dy={10} />
                    <YAxis axisLine={false} tickLine={false} tick={{fill: '#94a3b8', fontSize: 11}} />
                    <Tooltip 
                        contentStyle={{backgroundColor: 'rgba(255, 255, 255, 0.95)', borderRadius: '12px', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'}}
                    />
                    <Area type="monotone" dataKey="upper" stackId="2" stroke="none" fill="#f8fafc" />
                    <Area type="monotone" dataKey="lower" stackId="2" stroke="none" fill="#fff" />
                    <Area type="monotone" dataKey="projected" stroke="#6366f1" strokeWidth={2} strokeDasharray="4 4" fill="none" name="Forecast" />
                    <Area type="monotone" dataKey="revenue" stroke="#10b981" strokeWidth={3} fillOpacity={1} fill="url(#colorPredicted)" name="Actual Revenue" />
                </AreaChart>
                </ResponsiveContainer>
            ) : (
                <div className="h-full flex flex-col items-center justify-center text-slate-300">
                    <Activity size={48} className="mb-4 opacity-30" />
                    <p>No Sales Data</p>
                </div>
            )}
          </div>
        </div>

        <div className="bg-white rounded-2xl border border-slate-200/60 shadow-sm overflow-hidden flex flex-col">
          <div className="p-5 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center">
            <h3 className="font-bold text-slate-900">Learning Timeline</h3>
          </div>
          <div className="flex-1 overflow-auto p-6 space-y-6 relative">
            <div className="absolute left-9 top-6 bottom-6 w-0.5 bg-slate-100"></div>
            {events.length === 0 ? (
                <div className="text-center py-10 text-slate-400 text-sm">No events recorded.</div>
            ) : (
                events.map((event) => (
                <div key={event.id} className="relative pl-10 animate-in slide-in-from-left-4 fade-in">
                    <div className={`absolute left-0 top-1 h-6 w-6 rounded-full flex items-center justify-center ring-4 ring-white z-10 ${
                        event.type === 'Retrain' ? 'bg-indigo-100 text-indigo-600' : 
                        event.type === 'Drift' ? 'bg-amber-100 text-amber-600' : 'bg-emerald-100 text-emerald-600'
                    }`}>
                        <div className={`w-2 h-2 rounded-full ${
                            event.type === 'Retrain' ? 'bg-indigo-600' : 
                            event.type === 'Drift' ? 'bg-amber-600' : 'bg-emerald-600'
                        }`}></div>
                    </div>
                    <div className="flex justify-between items-start">
                        <span className="text-xs font-bold text-slate-800 uppercase tracking-wide">{event.type}</span>
                        <span className="text-[10px] text-slate-400 font-mono">{event.date}</span>
                    </div>
                    <p className="text-sm text-slate-600 mt-1 leading-relaxed">{event.description}</p>
                    <div className="mt-2 inline-flex items-center text-xs font-bold text-slate-500 bg-slate-50 px-2.5 py-1 rounded border border-slate-100">
                    {event.automated && <RefreshCw size={10} className="mr-1.5 text-slate-400" />}
                    {event.impactMetric}
                    </div>
                </div>
                ))
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

export default LearningLoopView;