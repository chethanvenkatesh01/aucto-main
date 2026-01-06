
import React, { useState, useEffect } from 'react';
import { X, Save, Globe, Key, Bell, ShieldCheck, Server, Mail, Smartphone, Check, AlertCircle } from 'lucide-react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
  const [activeTab, setActiveTab] = useState<'general' | 'integrations' | 'notifications'>('general');
  const [tenantName, setTenantName] = useState('Enterprise Tenant');
  const [notifications, setNotifications] = useState({
    email: true,
    slack: false,
    mobile: true
  });
  
  // State for connected services to allow toggling
  const [services, setServices] = useState([
    { id: 'snowflake', name: 'Snowflake Data Warehouse', icon: Server, color: 'text-blue-500', connected: true },
    { id: 'mlflow', name: 'MLflow Tracking Server', icon: Server, color: 'text-orange-500', connected: true }
  ]);
  
  const [isSaving, setIsSaving] = useState(false);

  // Load from local storage on mount
  useEffect(() => {
    const savedTenant = localStorage.getItem('auctorian_tenant');
    if (savedTenant) setTenantName(savedTenant);
  }, []);

  const handleSave = () => {
    setIsSaving(true);
    // Simulate API call / Persistence
    setTimeout(() => {
        localStorage.setItem('auctorian_tenant', tenantName);
        setIsSaving(false);
        onClose();
    }, 800);
  };

  // Handler to toggle service connection
  const toggleService = (id: string) => {
      setServices(prev => prev.map(s => s.id === id ? { ...s, connected: !s.connected } : s));
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-sm p-4 animate-in fade-in duration-200">
      <div className="bg-white w-full max-w-2xl rounded-2xl shadow-2xl border border-slate-200 overflow-hidden flex flex-col max-h-[85vh] animate-in zoom-in-95 duration-200">
        
        {/* Header */}
        <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-white shrink-0">
          <div>
            <h3 className="text-xl font-bold text-slate-900">System Settings</h3>
            <p className="text-sm text-slate-500 mt-1">Manage configuration and integrations</p>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-600 p-1 rounded-full hover:bg-slate-100 transition-colors">
              <X size={20} />
          </button>
        </div>

        <div className="flex flex-1 overflow-hidden">
            {/* Sidebar Tabs */}
            <div className="w-48 bg-slate-50 border-r border-slate-100 p-4 space-y-2">
                <button 
                    onClick={() => setActiveTab('general')}
                    className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                        activeTab === 'general' ? 'bg-white text-indigo-600 shadow-sm ring-1 ring-indigo-100' : 'text-slate-600 hover:bg-slate-100'
                    }`}
                >
                    <Globe size={16} />
                    <span>General</span>
                </button>
                <button 
                    onClick={() => setActiveTab('integrations')}
                    className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                        activeTab === 'integrations' ? 'bg-white text-indigo-600 shadow-sm ring-1 ring-indigo-100' : 'text-slate-600 hover:bg-slate-100'
                    }`}
                >
                    <Server size={16} />
                    <span>Integrations</span>
                </button>
                <button 
                    onClick={() => setActiveTab('notifications')}
                    className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                        activeTab === 'notifications' ? 'bg-white text-indigo-600 shadow-sm ring-1 ring-indigo-100' : 'text-slate-600 hover:bg-slate-100'
                    }`}
                >
                    <Bell size={16} />
                    <span>Notifications</span>
                </button>
            </div>

            {/* Content Area */}
            <div className="flex-1 p-8 overflow-y-auto bg-white">
                {activeTab === 'general' && (
                    <div className="space-y-6">
                        <div className="space-y-3">
                            <label className="text-sm font-bold text-slate-700">Tenant Name</label>
                            <input 
                                type="text" 
                                value={tenantName}
                                onChange={(e) => setTenantName(e.target.value)}
                                className="w-full px-4 py-2 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none text-slate-900 placeholder:text-slate-400 transition-colors"
                                placeholder="e.g. Acme Corp"
                            />
                            <p className="text-xs text-slate-500">This name will appear on all audit logs and reports.</p>
                        </div>
                        <div className="pt-4 border-t border-slate-100">
                            <div className="flex items-start space-x-3 bg-indigo-50 p-4 rounded-xl border border-indigo-100">
                                <ShieldCheck size={20} className="text-indigo-600 shrink-0 mt-0.5" />
                                <div>
                                    <h4 className="text-sm font-bold text-indigo-900">Security Mode</h4>
                                    <p className="text-xs text-indigo-700 mt-1">Role-Based Access Control (RBAC) is enforced. Only Admins can change system settings.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'integrations' && (
                    <div className="space-y-6">
                        <div className="space-y-4">
                            <h4 className="text-sm font-bold text-slate-900">Connected Services</h4>
                            {services.map((service) => (
                                <div key={service.id} className={`flex items-center justify-between p-3 border rounded-xl transition-all ${
                                    service.connected ? 'bg-slate-50 border-slate-200' : 'bg-white border-slate-100 opacity-75 hover:opacity-100'
                                }`}>
                                    <div className="flex items-center space-x-3">
                                        <div className={`w-8 h-8 rounded-lg border flex items-center justify-center ${
                                            service.connected ? 'bg-white border-slate-200' : 'bg-slate-50 border-slate-100'
                                        }`}>
                                            <service.icon size={16} className={service.connected ? service.color : 'text-slate-400'} />
                                        </div>
                                        <div>
                                            <div className={`text-sm font-semibold ${service.connected ? 'text-slate-800' : 'text-slate-500'}`}>{service.name}</div>
                                            <div className="text-[10px] font-bold flex items-center mt-0.5">
                                                {service.connected ? (
                                                    <span className="text-emerald-600 flex items-center">
                                                        <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full mr-1.5"></span> Connected
                                                    </span>
                                                ) : (
                                                    <span className="text-slate-400 flex items-center">
                                                        <span className="w-1.5 h-1.5 bg-slate-300 rounded-full mr-1.5"></span> Disconnected
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                    <button 
                                        onClick={() => toggleService(service.id)}
                                        className={`text-xs font-medium px-3 py-1.5 rounded-lg border transition-all ${
                                            service.connected 
                                            ? 'text-rose-600 border-rose-200 hover:bg-rose-50' 
                                            : 'text-indigo-600 border-indigo-200 hover:bg-indigo-50'
                                        }`}
                                    >
                                        {service.connected ? 'Disconnect' : 'Connect'}
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {activeTab === 'notifications' && (
                    <div className="space-y-6">
                        <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
                            <div className="flex items-center space-x-3">
                                <div className="p-2 bg-white rounded-lg shadow-sm text-slate-500">
                                    <Mail size={18} />
                                </div>
                                <div>
                                    <h4 className="text-sm font-bold text-slate-800">Email Alerts</h4>
                                    <p className="text-xs text-slate-500">Receive weekly digests and critical system alerts.</p>
                                </div>
                            </div>
                            <label className="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" checked={notifications.email} onChange={() => setNotifications(p => ({...p, email: !p.email}))} className="sr-only peer" />
                                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
                            </label>
                        </div>

                        <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
                            <div className="flex items-center space-x-3">
                                <div className="p-2 bg-white rounded-lg shadow-sm text-slate-500">
                                    <Bell size={18} />
                                </div>
                                <div>
                                    <h4 className="text-sm font-bold text-slate-800">Slack Integration</h4>
                                    <p className="text-xs text-slate-500">Post Auctobot cycle summaries to #ops-alerts.</p>
                                </div>
                            </div>
                            <label className="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" checked={notifications.slack} onChange={() => setNotifications(p => ({...p, slack: !p.slack}))} className="sr-only peer" />
                                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
                            </label>
                        </div>

                        <div className="flex items-center justify-between p-4 bg-slate-50 rounded-xl border border-slate-100">
                            <div className="flex items-center space-x-3">
                                <div className="p-2 bg-white rounded-lg shadow-sm text-slate-500">
                                    <Smartphone size={18} />
                                </div>
                                <div>
                                    <h4 className="text-sm font-bold text-slate-800">Mobile Push</h4>
                                    <p className="text-xs text-slate-500">Critical intervention requests only.</p>
                                </div>
                            </div>
                            <label className="relative inline-flex items-center cursor-pointer">
                                <input type="checkbox" checked={notifications.mobile} onChange={() => setNotifications(p => ({...p, mobile: !p.mobile}))} className="sr-only peer" />
                                <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
                            </label>
                        </div>
                    </div>
                )}
            </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-100 flex justify-end bg-slate-50 shrink-0">
            <button 
                onClick={handleSave}
                disabled={isSaving}
                className="bg-indigo-600 text-white px-6 py-2.5 rounded-xl text-sm font-bold hover:bg-indigo-700 transition-all shadow-lg shadow-indigo-200 hover:-translate-y-0.5 flex items-center space-x-2 disabled:opacity-70 disabled:shadow-none"
            >
                {isSaving ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></div> : <Save size={18} />}
                <span>{isSaving ? 'Saving...' : 'Save Changes'}</span>
            </button>
        </div>

      </div>
    </div>
  );
};

export default SettingsModal;
