import React, { useState, useEffect } from 'react';
import { 
  Database, 
  Upload, 
  FileText, 
  CheckCircle, 
  AlertTriangle, 
  Server, 
  Activity, 
  Layers, 
  RefreshCw, 
  ShieldCheck 
} from 'lucide-react';
import { api } from '../services/api';
import { DataContract } from '../types';

const DataPlaneView: React.FC = () => {
  // --- STATE ---
  const [stats, setStats] = useState<any>({ objects: 0, events: 0, status: 'Loading...' });
  const [contracts, setContracts] = useState<DataContract[]>([]); // New: Stores active contracts
  
  // Upload State
  const [file, setFile] = useState<File | null>(null);
  const [uploadMode, setUploadMode] = useState<'EVENT' | 'OBJECT'>('EVENT');
  const [entityName, setEntityName] = useState<string>('COMP_PRICE');
  const [mappingStr, setMappingStr] = useState<string>('{\n  "Target_ID": "SKU",\n  "Date": "Date",\n  "Value": "Price",\n  "Location_ID": "Store"\n}');
  const [uploadStatus, setUploadStatus] = useState<string>('');
  const [isUploading, setIsUploading] = useState(false);
  
  // --- ACTIONS ---

  const refreshData = async () => {
    try {
        // 1. Fetch Graph Stats
        const statsData = await api.graph.getStats();
        setStats(statsData);

        // 2. Fetch Active Contracts (Fixes the blank screen issue)
        const contractsData = await api.ingest.getContracts();
        setContracts(contractsData);
    } catch (e) {
        console.error("Data Plane Refresh Failed", e);
        setStats({ objects: 0, events: 0, status: 'Offline' });
    }
  };

  // Initial Load
  useEffect(() => {
    refreshData();
  }, []);

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setUploadStatus('Processing...');
    
    try {
        // Validate JSON Mapping
        let mapping;
        try { 
            mapping = JSON.parse(mappingStr); 
        } catch (e) { 
            throw new Error("Invalid JSON Mapping format"); 
        }

        // Call the Universal Ingest API
        const config = { 
            type: uploadMode, 
            entity_name: entityName.toUpperCase(), 
            mapping: mapping 
        };
        
        const res = await api.ingest.uploadUniversal(file, config);

        if (res.status === 'success' || res.status === 'warning') {
            setUploadStatus(`Success! Ingested ${res.count || 0} records.`);
            setFile(null);
            // Refresh to show the new contract immediately
            await refreshData();
        } else {
            setUploadStatus(`Error: ${res.message || 'Unknown error'}`);
        }
    } catch (e: any) {
        setUploadStatus(`Upload Failed: ${e.message}`);
    } finally {
        setIsUploading(false);
    }
  };

  // Templates for quick setup
  const loadTemplate = (type: 'SALES' | 'PRICE' | 'PRODUCT') => {
      if (type === 'SALES') {
          setUploadMode('EVENT'); 
          setEntityName('SALES_QTY');
          setMappingStr('{\n  "Target_ID": "SKU",\n  "Date": "Date",\n  "Value": "Qty",\n  "Location_ID": "Store"\n}');
      } else if (type === 'PRICE') {
          setUploadMode('EVENT'); 
          setEntityName('COMP_PRICE');
          setMappingStr('{\n  "Target_ID": "SKU",\n  "Date": "ObservedDate",\n  "Value": "Price"\n}');
      } else if (type === 'PRODUCT') {
          setUploadMode('OBJECT'); 
          setEntityName('PRODUCT');
          setMappingStr('{\n  "ID": "SKU",\n  "Name": "Description",\n  "Category": "Dept"\n}');
      }
  };

  return (
    <div className="p-8 h-full bg-slate-50 text-slate-900 overflow-y-auto font-sans">
      
      {/* 1. HEADER */}
      <div className="mb-8 flex justify-between items-end">
        <div>
            <h2 className="text-3xl font-bold text-slate-900 flex items-center gap-3">
            <Database className="text-indigo-600" /> Universal Data Plane
            </h2>
            <p className="text-slate-500 mt-2">
                Manage the Knowledge Graph and Active Data Contracts.
            </p>
        </div>
        <button 
            onClick={refreshData} 
            className="p-2 text-slate-400 hover:text-indigo-600 transition-colors bg-white border border-slate-200 rounded-lg shadow-sm"
            title="Refresh Data"
        >
            <RefreshCw size={18} />
        </button>
      </div>

      {/* 2. KPI CARDS */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <div className="flex items-center gap-3 mb-2">
                <div className="p-2 bg-blue-50 rounded-lg text-blue-600"><Layers size={20}/></div>
                <div className="font-bold text-slate-500 text-sm">Universal Objects</div>
            </div>
            <div className="text-3xl font-bold text-slate-800">{stats.objects?.toLocaleString() || 0}</div>
        </div>
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <div className="flex items-center gap-3 mb-2">
                <div className="p-2 bg-emerald-50 rounded-lg text-emerald-600"><Activity size={20}/></div>
                <div className="font-bold text-slate-500 text-sm">Universal Events</div>
            </div>
            <div className="text-3xl font-bold text-slate-800">{stats.events?.toLocaleString() || 0}</div>
        </div>
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
            <div className="flex items-center gap-3 mb-2">
                <div className="p-2 bg-indigo-50 rounded-lg text-indigo-600"><Server size={20}/></div>
                <div className="font-bold text-slate-500 text-sm">Graph Status</div>
            </div>
            <div className="text-xl font-bold text-emerald-600 flex items-center gap-2">
                <CheckCircle size={20}/> {stats.status || "Active"}
            </div>
        </div>
      </div>

      {/* 3. ACTIVE DATA CONTRACTS TABLE (New Section) */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden mb-8 animate-in fade-in slide-in-from-bottom-4">
          <div className="p-4 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center">
              <h3 className="font-bold text-slate-700 flex items-center gap-2">
                  <ShieldCheck size={18} className="text-indigo-600"/> Active Data Contracts
              </h3>
              <span className="text-xs font-mono text-slate-400">{contracts.length} Streams Active</span>
          </div>
          
          <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                  <thead className="bg-slate-50 text-slate-500 font-bold uppercase text-[10px] tracking-wider">
                      <tr>
                          <th className="p-4 border-b border-slate-100">Contract ID</th>
                          <th className="p-4 border-b border-slate-100">Dataset Name</th>
                          <th className="p-4 border-b border-slate-100">Type</th>
                          <th className="p-4 border-b border-slate-100">Quality Score</th>
                          <th className="p-4 border-b border-slate-100 text-right">Records</th>
                          <th className="p-4 border-b border-slate-100 text-center">Status</th>
                      </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                      {contracts.length === 0 ? (
                          <tr>
                              <td colSpan={6} className="p-12 text-center text-slate-400">
                                  <div className="flex flex-col items-center gap-2">
                                      <Database size={24} className="opacity-20"/>
                                      <span>No active data streams found. Use the Ingestion Pipeline below.</span>
                                  </div>
                              </td>
                          </tr>
                      ) : (
                          contracts.map(c => (
                              <tr key={c.id} className="hover:bg-slate-50 transition-colors">
                                  <td className="p-4 font-mono text-xs text-slate-500">{c.id}</td>
                                  <td className="p-4 font-bold text-slate-700">{c.datasetName}</td>
                                  <td className="p-4">
                                      <span className={`px-2 py-1 rounded text-[10px] font-bold border ${
                                          c.detectedType === 'EVENT' 
                                            ? 'bg-emerald-50 text-emerald-700 border-emerald-100' 
                                            : 'bg-blue-50 text-blue-700 border-blue-100'
                                      }`}>
                                          {c.detectedType}
                                      </span>
                                  </td>
                                  <td className="p-4">
                                      <div className="flex items-center gap-2">
                                          <div className="w-16 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                              <div className="h-full bg-emerald-500 rounded-full" style={{width: `${c.qualityScore}%`}}></div>
                                          </div>
                                          <span className="text-xs font-bold text-emerald-600">{c.qualityScore}%</span>
                                      </div>
                                  </td>
                                  <td className="p-4 font-mono text-right">{c.rowCount.toLocaleString()}</td>
                                  <td className="p-4 text-center">
                                      <span className="px-2 py-1 bg-emerald-50 text-emerald-700 rounded text-[10px] font-bold border border-emerald-200 uppercase">Active</span>
                                  </td>
                              </tr>
                          ))
                      )}
                  </tbody>
              </table>
          </div>
      </div>

      {/* 4. UNIVERSAL UPLOADER */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="p-4 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center">
            <h3 className="font-bold text-slate-700 flex items-center gap-2">
                <Upload size={18} className="text-indigo-600"/> Ingestion Pipeline
            </h3>
            <div className="flex gap-2">
                <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider py-1">Load Template:</span>
                <button onClick={() => loadTemplate('SALES')} className="text-[10px] px-3 py-1 bg-white border border-slate-200 rounded-lg hover:bg-indigo-50 hover:text-indigo-600 hover:border-indigo-200 transition-colors font-medium text-slate-600">Sales</button>
                <button onClick={() => loadTemplate('PRICE')} className="text-[10px] px-3 py-1 bg-white border border-slate-200 rounded-lg hover:bg-indigo-50 hover:text-indigo-600 hover:border-indigo-200 transition-colors font-medium text-slate-600">Price</button>
                <button onClick={() => loadTemplate('PRODUCT')} className="text-[10px] px-3 py-1 bg-white border border-slate-200 rounded-lg hover:bg-indigo-50 hover:text-indigo-600 hover:border-indigo-200 transition-colors font-medium text-slate-600">Product</button>
            </div>
        </div>
        
        <div className="p-8 grid grid-cols-1 lg:grid-cols-2 gap-12">
            
            {/* LEFT: CONFIGURATION */}
            <div className="space-y-6">
                <div>
                    <label className="block text-xs font-bold text-slate-500 uppercase mb-2">1. Data Archetype</label>
                    <div className="flex gap-4">
                        <button 
                            onClick={() => { setUploadMode('EVENT'); setEntityName('COMP_PRICE'); }}
                            className={`flex-1 py-3 rounded-xl border font-bold text-sm transition-all flex items-center justify-center gap-2 ${uploadMode === 'EVENT' ? 'bg-indigo-600 text-white border-indigo-600 shadow-md' : 'bg-white text-slate-500 hover:bg-slate-50'}`}
                        >
                            <Activity size={16}/> Event Stream
                        </button>
                        <button 
                            onClick={() => { setUploadMode('OBJECT'); setEntityName('VENDOR'); }}
                            className={`flex-1 py-3 rounded-xl border font-bold text-sm transition-all flex items-center justify-center gap-2 ${uploadMode === 'OBJECT' ? 'bg-indigo-600 text-white border-indigo-600 shadow-md' : 'bg-white text-slate-500 hover:bg-slate-50'}`}
                        >
                            <Layers size={16}/> Reference Object
                        </button>
                    </div>
                </div>

                <div>
                    <label className="block text-xs font-bold text-slate-500 uppercase mb-2">2. Entity Definition</label>
                    <input 
                        type="text" 
                        value={entityName}
                        onChange={(e) => setEntityName(e.target.value)}
                        className="w-full p-3 bg-slate-50 border border-slate-200 rounded-xl font-mono text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all"
                        placeholder={uploadMode === 'EVENT' ? 'e.g. SALES_QTY' : 'e.g. PRODUCT'}
                    />
                </div>

                <div>
                    <label className="block text-xs font-bold text-slate-500 uppercase mb-2">3. Schema Mapping (JSON)</label>
                    <textarea 
                        value={mappingStr}
                        onChange={(e) => setMappingStr(e.target.value)}
                        className="w-full h-32 p-3 bg-slate-900 text-emerald-400 border border-slate-800 rounded-xl font-mono text-xs focus:ring-2 focus:ring-indigo-500 outline-none shadow-inner"
                    />
                </div>
            </div>

            {/* RIGHT: FILE & ACTION */}
            <div className="flex flex-col justify-center items-center border-l border-slate-100 pl-12">
                <label className="w-full h-40 border-2 border-dashed border-slate-300 rounded-2xl flex flex-col items-center justify-center cursor-pointer hover:border-indigo-400 hover:bg-indigo-50/10 transition-all group relative bg-slate-50/50">
                    <input type="file" className="hidden" onChange={(e) => setFile(e.target.files?.[0] || null)} />
                    <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center mb-3 shadow-sm group-hover:scale-110 transition-transform">
                        <FileText size={24} className="text-indigo-600" />
                    </div>
                    <div className="text-sm font-bold text-slate-600 group-hover:text-indigo-700">
                        {file ? file.name : "Drop CSV File Here"}
                    </div>
                    {file && <div className="absolute top-2 right-2 bg-indigo-600 text-white text-[10px] font-bold px-2 py-1 rounded-full animate-in zoom-in">Ready</div>}
                </label>

                {uploadStatus && (
                    <div className={`mt-4 w-full p-3 rounded-lg text-xs font-bold text-center flex items-center justify-center gap-2 animate-in fade-in slide-in-from-top-2 ${uploadStatus.includes('Success') ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-600'}`}>
                        {uploadStatus.includes('Error') || uploadStatus.includes('Failed') ? <AlertTriangle size={14}/> : <CheckCircle size={14}/>}
                        {uploadStatus}
                    </div>
                )}

                <button 
                    onClick={handleUpload}
                    disabled={!file || isUploading}
                    className="mt-6 w-full py-4 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white rounded-xl font-bold shadow-lg shadow-indigo-200/50 flex items-center justify-center gap-2 transition-all hover:scale-[1.02] active:scale-95"
                >
                    {isUploading ? <RefreshCw className="animate-spin" size={18}/> : <Upload size={18} />} 
                    {isUploading ? 'Ingesting Pipeline...' : 'Inject into Kernel'}
                </button>
            </div>
        </div>
      </div>
    </div>
  );
};

export default DataPlaneView;
