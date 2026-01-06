import { SKUData, SystemAlert, ModelMetric, Plane, DataContract, AuditLogEntry, SOP, FeasibilityMetric, LearningEvent, CartridgeManifest } from './types.ts';
import { GridRow } from './components/PlanningGrid.tsx';
import { AssortmentRow } from './components/AssortmentMatrix.tsx';

// --- 1. RETAIL CARTRIDGE (The Classic) ---
export const RETAIL_CARTRIDGE_V1: CartridgeManifest = {
  id: 'cart-retail-001',
  name: 'Retail Merchandising v1',
  version: '1.2.0',
  coreVersion: '>=4.2',
  description: 'Autonomous pricing, replenishment, and markdown optimization for omnichannel retail.',
  icon: 'ShoppingBag',
  status: 'INSTALLED',
  colorTheme: 'indigo',
  
  // NEW: Domain Vocabulary
  vocabulary: {
    product: 'Product',
    location: 'Store',
    transaction: 'Sales'
  },
  
  // NEW: Data Schema for Retail
  dataSchema: [
    { 
      id: 'CATALOG', label: 'Product Master', type: 'MASTER', icon: 'Box',
      description: 'SKU attributes, costs, and pricing.',
      mandatoryFields: ['Category', 'SKU'], optionalFields: ['Cost', 'Price', 'Brand'] 
    },
    { 
      id: 'NETWORK', label: 'Store Network', type: 'MASTER', icon: 'MapPin',
      description: 'Physical store locations and DCs.',
      mandatoryFields: ['Region', 'Store'], optionalFields: ['City', 'SqFt'] 
    },
    { 
      id: 'SALES', label: 'Sales History', type: 'TRANSACTIONAL', icon: 'Activity',
      description: 'Historical POS transaction logs.',
      mandatoryFields: ['Date', 'SKU', 'Store', 'Qty'], optionalFields: ['Revenue', 'Discount'] 
    }
  ],

  capabilities: {
    agents: [
      'Retail_Analyst (Compliance Officer)', 
      'Retail_Strategist (Merchandiser)', 
      'Inventory_Manager'
    ],
    sops: 142,
    workspaces: [
      'Nucleus Forecast',
      'Price Optimizer',
      'Markdown Manager',
      'Allocation Hub',
      'Merch Financial Planner',
      'Assortment Planner'
    ]
  },
  sopConfig: [
    { id: 'pricing_max_change', name: 'Max Price Hike Cap', rule: 'Max ±15% per week', type: 'hard_guardrail' },
    { id: 'margin_floor', name: 'Gross Margin Floor', rule: 'Min 25% GM required', type: 'hard_guardrail' },
    { id: 'markdown_cadence', name: 'Clearance Cadence', rule: 'Only on Thursdays', type: 'soft_guardrail' }
  ]
};

// --- 2. LOGISTICS CARTRIDGE (The New Domain) ---
export const LOGISTICS_CARTRIDGE: CartridgeManifest = {
  id: 'cart-logistics-001',
  name: 'Logistics & 3PL Command',
  version: '0.9.0-BETA',
  coreVersion: '>=4.2',
  description: 'Fleet optimization, fuel management, and route profitability analysis.',
  icon: 'Truck',
  status: 'INSTALLED',
  colorTheme: 'amber',

  // NEW: Domain Vocabulary
  vocabulary: {
    product: 'Vehicle',
    location: 'Depot',
    transaction: 'Fuel Log'
  },

  // NEW: Data Schema for Logistics
  dataSchema: [
    { 
      id: 'FLEET', label: 'Fleet Registry', type: 'MASTER', icon: 'Truck',
      description: 'Vehicle specifications and capacity.',
      mandatoryFields: ['Class', 'Vehicle_ID'], optionalFields: ['Max_Load', 'License_Plate'] 
    },
    { 
      id: 'DEPOTS', label: 'Depot Network', type: 'MASTER', icon: 'MapPin',
      description: 'Warehouses and refueling stations.',
      mandatoryFields: ['Region', 'Depot_ID'], optionalFields: ['Capacity', 'Fuel_Type'] 
    },
    { 
      id: 'FUEL', label: 'Fuel Telematics', type: 'TRANSACTIONAL', icon: 'Droplet',
      description: 'Daily fuel consumption logs.',
      mandatoryFields: ['Date', 'Vehicle_ID', 'Depot_ID', 'Qty'], optionalFields: ['Odometer'] 
    }
  ],

  capabilities: {
    agents: ['Fleet_Commander', 'Route_Optimizer'],
    sops: 45,
    workspaces: ['Fleet Command', 'Fuel Analytics']
  }
};

// --- MOCK DATA ---
export const MOCK_ALERTS: SystemAlert[] = [
  { id: 'ALT-001', plane: Plane.ORCHESTRATION, message: 'Conflict detected in Pricing vs Margin goals.', severity: 'medium', timestamp: '10 min ago' },
  { id: 'ALT-002', plane: Plane.DATA, message: 'Inventory feed delayed by 45ms.', severity: 'low', timestamp: '2 hours ago' },
];

export const MOCK_SKUS: SKUData[] = [
  { id: 'SKU-101', name: 'Premium Cotton Tee', category: 'Apparel', currentPrice: 25.00, cost: 8.50, inventory: 1200, elasticity: -1.2, predictedDemand: 150 },
  { id: 'SKU-102', name: 'Slim Fit Chinos', category: 'Apparel', currentPrice: 65.00, cost: 22.00, inventory: 450, elasticity: -0.8, predictedDemand: 80 },
];

export const MOCK_MODELS: ModelMetric[] = [
  { id: 'MDL-XGB-01', name: 'XGBoost Demand Forecaster', type: 'XGBoost', version: 'v2.1.4', accuracy: 94.2, drift: 0.03, status: 'Production', isChampion: true },
  { id: 'MDL-PROPHET-02', name: 'Prophet Seasonality', type: 'Prophet', version: 'v1.0.8', accuracy: 89.5, drift: 0.12, status: 'Staging', isChampion: false },
];

export const MOCK_LEARNING_EVENTS: LearningEvent[] = [
  { id: 'EVT-001', date: '2023-11-01', type: 'Price Elasticity Correction', description: 'Observed higher demand at $45 price point.', impactMetric: '+2.3% Margin', automated: true },
];

export const MOCK_LEARNING_LOOP_CHART = [
  { name: 'Week 1', revenue: 4000, projected: 4200 },
  { name: 'Week 2', revenue: 3000, projected: 3100 },
  { name: 'Week 3', revenue: 2000, projected: 2300 },
  { name: 'Week 4', revenue: 2780, projected: 2500 },
  { name: 'Week 5', revenue: 1890, projected: 2100 },
];

export const MOCK_OTB_DATA: GridRow[] = [
  { id: 'CAT-01', name: 'Men\'s Apparel', metrics: { sales: 120000, receipt: 45000, inventory: 32000, margin: 45 } },
  { id: 'CAT-02', name: 'Women\'s Apparel', metrics: { sales: 180000, receipt: 65000, inventory: 45000, margin: 48 } },
];

export const MOCK_ASSORTMENT_DATA: AssortmentRow[] = [
  { id: 'AST-01', product: 'Core Tee', status: 'Keep', performance: 'High', margin: 65, weeks_cover: 8 },
  { id: 'AST-02', product: 'Seasonal Jacket', status: 'Drop', performance: 'Low', margin: 25, weeks_cover: 22 },
];

export const MOCK_FEASIBILITY: FeasibilityMetric[] = [
  { id: 'FEAS-01', label: 'Production Capacity', value: 8500, limit: 10000, unit: 'units' },
  { id: 'FEAS-02', label: 'Marketing Budget', value: 42000, limit: 50000, unit: 'USD' },
  { id: 'FEAS-03', label: 'Logistics Bandwidth', value: 92, limit: 100, unit: '%' },
];

export const MOCK_DATA_CONTRACTS: DataContract[] = [
  { id: 'DC-01', datasetName: 'Sales Transaction Feed', source: 'Snowflake (ERP)', frequency: 'Daily @ 02:00', lastIngested: 'Today 02:00 AM', status: 'Healthy', qualityScore: 99.8, schemaValid: true },
];

export const MOCK_AUDIT_LOGS: AuditLogEntry[] = [
  { id: 'LOG-9921', timestamp: '2023-11-01 14:32:10', actor: 'Auctobot', action: 'Execute Decision Package DP-V4-001', entityId: 'DP-V4-001', status: 'Success', hash: 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855' },
];
