import axios from 'axios';
import { 
  DataContract, 
  EnterpriseConfig, 
  PlanningRow, 
  PlanningScope,
  MetricValue,
  PlanNode 
} from '../types';

// Points to the FastAPI Kernel
// Change this if you are running on a different port or host
const API_URL = 'http://localhost:8000';

// Standard JSON Client
const client = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// File Upload Client
const uploadClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

export const api = {
  
  // =========================================================
  // 1. SYSTEM & CONFIGURATION
  // =========================================================
  
  async getEnterpriseConfig(): Promise<EnterpriseConfig> {
    try {
        const res = await client.get('/ontology/structure');
        // Map backend "levels" to frontend "hierarchy"
        return {
            hierarchy: (res.data.levels || []).map((l: string) => ({
                id: l.toLowerCase().replace(' ', '_'),
                label: l,
                isPlanningEnabled: ['Category', 'Department'].includes(l)
            })),
            timeBuckets: ['Week', 'Month', 'Quarter']
        };
    } catch (e) {
        // Fallback for offline mode or initial boot
        return {
            hierarchy: [
                { id: 'division', label: 'Division', isPlanningEnabled: false },
                { id: 'category', label: 'Category', isPlanningEnabled: true },
                { id: 'sku', label: 'SKU', isPlanningEnabled: true }
            ],
            timeBuckets: ['Week', 'Month', 'Quarter']
        };
    }
  },

  async getSystemStats(): Promise<any> {
    try {
        const res = await client.get('/ontology/stats');
        return res.data;
    } catch (e) {
        return null;
    }
  },

  async getHierarchyOptions(levelId: string, context: Record<string, string>): Promise<string[]> {
    try {
        // Map frontend IDs (e.g., 'division') back to Backend Types (e.g., 'Division')
        const type = levelId.charAt(0).toUpperCase() + levelId.slice(1);
        const objects = await client.get(`/graph/objects/${type}`);
        return objects.data.map((o: any) => o.name);
    } catch (e) {
        return [];
    }
  },

  // [CRITICAL FIX] Top-Level Alias for Backward Compatibility
  // This prevents crashes in components that haven't updated to api.graph.getObjects
  getGraphObjects: async (type: string) => {
      try {
          const res = await client.get(`/graph/objects/${type}`);
          return res.data;
      } catch (e) { return []; }
  },

  // =========================================================
  // 2. ORCHESTRATION (THE PROFIT KERNEL)
  // =========================================================
  
  orchestration: {
    // --- PlanSmart Engine (5-Year Horizon) ---
    getPlanningTree: async (hierarchy: string, startYear: number = 2025, horizonYears: number = 1): Promise<PlanNode[]> => {
        try {
            const res = await client.get('/orchestration/plan/tree', {
                params: { 
                    hierarchy, 
                    start_year: startYear, 
                    horizon_years: horizonYears 
                }
            });
            return res.data;
        } catch (e) {
            console.error("Planning Tree Fetch Failed", e);
            return [];
        }
    },

    // --- Tactical Engines ---
    runReplenishment: async () => (await client.post('/orchestration/run')).data,
    runPricing: async () => (await client.post('/orchestration/price_optimize')).data,
    runMarkdown: async () => (await client.post('/orchestration/markdown')).data,
    
    getAssortment: async () => (await client.get('/orchestration/assortment')).data,
    runAssortment: async () => (await client.get('/orchestration/assortment')).data,
    
    getAllocation: async () => (await client.get('/orchestration/allocation')).data,
    runAllocation: async () => (await client.get('/orchestration/allocation')).data,

    // Aliases for compatibility
    runPriceOptimization: async () => (await client.post('/orchestration/price_optimize')).data,
    runMarkdownSimulation: async () => (await client.post('/orchestration/markdown')).data,

    // --- Financial Planning ---
    getFinancialPlan: async (compare = 'LY', groupBy = 'category', bucket = 'Month'): Promise<PlanningRow[]> => {
        try {
            const res = await client.get('/orchestration/financial_plan', {
                params: { compare, group_by: groupBy, time_bucket: bucket }
            });
            return res.data;
        } catch (e) { return []; }
    },

    simulateFinancialPlan: async (payload: { compare: string, scenarios: any, group_by: string, time_bucket: string }): Promise<PlanningRow[]> => {
        const res = await client.post('/orchestration/financial_plan/simulate', payload);
        return res.data;
    },

    // --- System 3 (Debate Engine) ---
    conveneCouncil: async (nodeId: string, mode: string) => {
        const res = await client.post('/orchestration/convene_council', { node_id: nodeId, mode });
        return res.data;
    }
  },

  // =========================================================
  // 3. INTELLIGENCE & ML (GLASS BOX & SELF-CORRECTING)
  // =========================================================
  
  ml: {
      getMetrics: async () => {
          try {
              const res = await client.get('/ml/metrics');
              return res.data;
          } catch (e) { return { model: 'Offline', status: 'Checking...' }; }
      },
      
      // [GLASS BOX] Triggers the full Cleanse -> Backtest -> Compete -> Vectorize pipeline
      triggerDemandModeling: async () => {
          const res = await client.post('/ml/train');
          return res.data;
      },
      
      // [GLASS BOX] Fetches the Explanation (Audit Log) for a SKU
      getForecastExplanation: async (skuId: string) => {
          try {
              const res = await client.get(`/ml/explain/${skuId}`);
              return res.data;
          } catch (e) {
              return { status: "Explanation Unavailable" };
          }
      },

      // [SELF-CORRECTING] Fetches the WMAPE Accuracy Matrix for a Node
      getForecastAccuracy: async (nodeId: string | null) => {
          try {
              const target = nodeId || "GLOBAL";
              const res = await client.get(`/ml/accuracy/${target}`);
              return res.data;
          } catch (e) {
              return { node_id: nodeId, scores: {}, children_scores: [] };
          }
      },
      
      // Legacy Aliases
      train: async () => {
          const res = await client.post('/ml/train');
          return res.data;
      },
      
      predict: async (sku: string, days = 7) => {
          try {
            const res = await client.get(`/ml/predict?sku=${sku}&days=${days}`);
            return res.data;
          } catch (e) { return { forecast: [] }; }
      }
  },

  // Legacy Namespace (Restored for Backward Compatibility)
  intelligence: {
    trainModel: async () => {
      try {
        const res = await client.post('/ml/train');
        return res.data;
      } catch (e) { 
        console.error("Training Error", e); 
        throw e; 
      }
    },
    getMetrics: async () => {
        try {
            const res = await client.get('/ml/metrics');
            return res.data;
        } catch (e) { return { r2_score: 0, status: 'Offline' }; }
    },
    getForecast: async (sku: string, days: number = 7) => {
        try {
            const res = await client.get(`/ml/predict?sku=${sku}&days=${days}`);
            return res.data;
        } catch (e) { return { forecast: [] }; }
    }
  },

  // =========================================================
  // 4. GOVERNANCE
  // =========================================================
  
  governance: {
      getPolicies: async () => {
          const res = await client.get('/governance/policies');
          return res.data;
      },
      updatePolicy: async (key: string, value: number) => {
          const res = await client.post('/governance/policies', { key, value });
          return res.data;
      }
  },

  // =========================================================
  // 5. INGESTION & GRAPH (DATA PLANE)
  // =========================================================
  
  ingest: {
    uploadUniversal: async (file: File, config: any) => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('config', JSON.stringify(config));
      const res = await uploadClient.post('/ingest/universal', formData);
      return res.data;
    },
    uploadSales: async (file: File, map: any, pre: string = 'SALES') => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('mapping', JSON.stringify(map));
      formData.append('prefix', pre);
      const res = await uploadClient.post('/ingest/sales', formData);
      return res.data;
    },
    getContracts: async (): Promise<DataContract[]> => {
      try {
        const res = await client.get('/ingest/contracts');
        return res.data;
      } catch (e) { 
        return []; 
      }
    }
  },

  graph: {
    getObjects: async (type: string) => {
        try {
            return (await client.get(`/graph/objects/${type}`)).data;
        } catch (e) { return []; }
    },
    getEvents: async (type: string, target?: string) => {
      const url = target ? `/graph/events/${type}?target=${target}` : `/graph/events/${type}`;
      try {
          return (await client.get(url)).data;
      } catch (e) { return []; }
    },
    getStats: async () => {
        try {
            return (await client.get('/ontology/stats')).data;
        } catch (e) { return { objects: 0, events: 0, status: 'Offline' }; }
    }
  },

  ontology: {
    getStructure: async (type: string = 'PRODUCT') => {
      try {
          const res = await client.get(`/ontology/structure?type=${type}`);
          return res.data.levels || [];
      } catch (e) { return []; }
    },
    defineStructure: async (levels: string[]) => {
      const res = await client.post('/ontology/structure', { type: 'PRODUCT', levels });
      return res.data;
    },
    defineLocationStructure: async (levels: string[]) => {
      const res = await client.post('/ontology/structure', { type: 'LOCATION', levels });
      return res.data;
    },
    getPolicies: async () => (await client.get('/governance/policies')).data,
  },

  // =========================================================
  // 6. WORKSPACE HELPERS
  // =========================================================
  
  async initPlanningWorkspace(scope: PlanningScope): Promise<string> {
      return `WS-${scope?.anchorLevel?.toUpperCase() || 'GLOBAL'}-${Date.now().toString().slice(-4)}`;
  }
};
