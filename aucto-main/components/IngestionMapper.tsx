import React, { useState, useRef } from 'react';
import { Upload, FileText, ArrowRight, AlertCircle, X } from 'lucide-react';

interface IngestionMapperProps {
  title: string;
  description: string;
  targetFields: string[];
  optionalFields: string[];
  // UPDATED: Expects object, not string
  onSubmit: (file: File, mapping: Record<string, string>) => Promise<any>;
  onComplete: (result: any) => void;
}

export const IngestionMapper: React.FC<IngestionMapperProps> = ({
  title,
  description,
  targetFields,
  optionalFields,
  onSubmit,
  onComplete
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
  const [mapping, setMapping] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      parseHeaders(selectedFile);
    }
  };

  const parseHeaders = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const firstLine = text.split('\n')[0];
      let delimiter = ',';
      if (firstLine.includes('|')) delimiter = '|';
      else if (firstLine.includes(';')) delimiter = ';';
      else if (firstLine.includes('\t')) delimiter = '\t';
      
      const headers = firstLine.split(delimiter).map(h => h.trim().replace(/^"|"$/g, ''));
      setCsvHeaders(headers);
      
      const initialMapping: Record<string, string> = {};
      [...targetFields, ...optionalFields].forEach(field => {
        const match = headers.find(h => h.toLowerCase() === field.toLowerCase());
        if (match) initialMapping[field] = match;
      });
      setMapping(initialMapping);
    };
    reader.readAsText(file.slice(0, 5000));
  };

  const updateMapping = (sysField: string, csvHeader: string) => {
    setMapping(prev => ({ ...prev, [sysField]: csvHeader }));
  };

  const handleSubmit = async () => {
    if (!file) return;
    setIsSubmitting(true);
    setError(null);

    const missing = targetFields.filter(f => !mapping[f]);
    if (missing.length > 0) {
      setError(`Missing mandatory mappings: ${missing.join(', ')}`);
      setIsSubmitting(false);
      return;
    }

    try {
      // FIXED: Pass raw object 'mapping', do NOT stringify here
      const result = await onSubmit(file, mapping);
      onComplete(result);
    } catch (err: any) {
      setError(err.message || "Ingestion Failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  const reset = () => {
    setFile(null);
    setCsvHeaders([]);
    setMapping({});
    setError(null);
  };

  return (
    <div className="h-full flex flex-col">
      <div className="mb-6">
        <h4 className="text-lg font-bold text-slate-900">{title}</h4>
        <p className="text-sm text-slate-500">{description}</p>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 text-red-700 text-sm rounded-lg flex items-center gap-2">
          <AlertCircle size={16} /> {error}
        </div>
      )}

      {!file && (
        <div 
          onClick={() => fileInputRef.current?.click()}
          className="flex-1 border-2 border-dashed border-slate-300 rounded-xl flex flex-col items-center justify-center cursor-pointer hover:bg-slate-50 transition-colors p-8"
        >
          <div className="w-16 h-16 bg-indigo-50 text-indigo-600 rounded-full flex items-center justify-center mb-4">
            <Upload size={32} />
          </div>
          <p className="font-medium text-slate-700">Click to Upload CSV</p>
          <p className="text-xs text-slate-400 mt-1">Supports .csv, .txt</p>
          <input type="file" ref={fileInputRef} className="hidden" accept=".csv,.txt" onChange={handleFileChange} />
        </div>
      )}

      {file && (
        <div className="flex-1 flex flex-col">
          <div className="flex justify-between items-center p-3 bg-slate-100 rounded-lg mb-6">
            <div className="flex items-center gap-2 text-sm font-medium text-slate-700">
              <FileText size={16} /> {file.name}
            </div>
            <button onClick={reset} className="text-slate-400 hover:text-red-500"><X size={16}/></button>
          </div>

          <div className="flex-1 overflow-auto pr-2 space-y-6">
            <div>
              <h5 className="text-xs font-bold text-indigo-600 uppercase tracking-wide mb-3">Mandatory Fields</h5>
              <div className="space-y-3">
                {targetFields.map(field => (
                  <div key={field} className="grid grid-cols-2 gap-4 items-center">
                    <label className="text-sm font-medium text-slate-700">{field}</label>
                    <select 
                      className={`text-sm border rounded-lg p-2 ${!mapping[field] ? 'border-red-300 bg-red-50' : 'border-slate-300'}`}
                      value={mapping[field] || ''}
                      onChange={(e) => updateMapping(field, e.target.value)}
                    >
                      <option value="">-- Select Column --</option>
                      {csvHeaders.map(h => <option key={h} value={h}>{h}</option>)}
                    </select>
                  </div>
                ))}
              </div>
            </div>

            {optionalFields.length > 0 && (
              <div>
                <h5 className="text-xs font-bold text-slate-400 uppercase tracking-wide mb-3 mt-4">Optional Attributes</h5>
                <div className="space-y-3">
                  {optionalFields.map(field => (
                    <div key={field} className="grid grid-cols-2 gap-4 items-center">
                      <label className="text-sm text-slate-500">{field}</label>
                      <select 
                        className="text-sm border border-slate-200 rounded-lg p-2 text-slate-600"
                        value={mapping[field] || ''}
                        onChange={(e) => updateMapping(field, e.target.value)}
                      >
                        <option value="">(Skip)</option>
                        {csvHeaders.map(h => <option key={h} value={h}>{h}</option>)}
                      </select>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          <div className="pt-6 mt-4 border-t border-slate-100">
            <button 
              onClick={handleSubmit}
              disabled={isSubmitting}
              className="w-full bg-indigo-600 text-white font-bold py-3 rounded-xl hover:bg-indigo-700 transition-all flex justify-center items-center gap-2 disabled:opacity-50"
            >
              {isSubmitting ? <span>Ingesting...</span> : <><span>Run Ingestion</span><ArrowRight size={18} /></>}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
