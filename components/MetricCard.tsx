import React from 'react';
import { ArrowUpRight, ArrowDownRight, Minus } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string;
  change?: string;
  trend?: 'up' | 'down' | 'neutral';
  icon?: React.ElementType;
  loading?: boolean;
}

const MetricCard: React.FC<MetricCardProps> = ({ title, value, change, trend, icon: Icon, loading }) => {
  return (
    <div className="bg-white rounded-xl border border-slate-200/60 shadow-sm hover:shadow-lg hover:-translate-y-1 transition-all duration-300 p-6 relative overflow-hidden group">
      {/* Subtle Background Decoration */}
      <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-br from-indigo-50 to-transparent rounded-bl-full opacity-50 group-hover:scale-110 transition-transform duration-500"></div>
      
      <div className="relative z-10 flex justify-between items-start mb-4">
        <h3 className="text-slate-500 text-xs font-semibold uppercase tracking-wider">{title}</h3>
        {Icon && (
            <div className="p-2 bg-slate-50 rounded-lg group-hover:bg-indigo-50 group-hover:text-indigo-600 transition-colors">
                <Icon className="text-slate-400 group-hover:text-indigo-600" size={18} />
            </div>
        )}
      </div>
      
      <div className="relative z-10 flex items-baseline space-x-2">
        {loading ? (
           <div className="h-8 flex items-center w-full">
             <div className="w-2/3 h-6 bg-slate-100 rounded animate-pulse"></div>
           </div>
        ) : (
          <h2 className="text-3xl font-bold text-slate-900 tracking-tight">{value}</h2>
        )}
      </div>
      
      {change && (
        <div className={`relative z-10 flex items-center mt-3 text-xs font-medium ${
          trend === 'up' ? 'text-emerald-600' : trend === 'down' ? 'text-rose-600' : 'text-slate-500'
        }`}>
          <span className={`flex items-center px-1.5 py-0.5 rounded ${
              trend === 'up' ? 'bg-emerald-50' : trend === 'down' ? 'bg-rose-50' : 'bg-slate-50'
          }`}>
              {trend === 'up' && <ArrowUpRight size={12} className="mr-1" />}
              {trend === 'down' && <ArrowDownRight size={12} className="mr-1" />}
              {trend === 'neutral' && <Minus size={12} className="mr-1" />}
              {change}
          </span>
          <span className="text-slate-400 font-normal ml-2">vs last period</span>
        </div>
      )}
    </div>
  );
};

export default MetricCard;