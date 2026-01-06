import { SKUData, RevenuePoint, DatasetType, CrossValidationResult, ModelMetric, ColumnMapping, FeatureImportance, TrainingLog, HierarchyLevel } from './types.ts';

// --- STANDARD SCHEMA DEFINITIONS ---
// The Nucleus expects data in these standard formats.
// The Data Plane's job is to map raw CSVs to these keys.
export const STANDARD_SCHEMA = {
    'Product': ['sku_id', 'name', 'category', 'cost', 'price', 'brand', 'weight', 'supplier'],
    'Sales': ['sku_id', 'date', 'revenue', 'units', 'store_id'],
    'Inventory': ['sku_id', 'quantity_on_hand', 'warehouse_id', 'safety_stock'],
    'Store': ['store_id', 'region', 'size', 'city'],
    'Unknown': []
};

// Synonyms for Auto-Mapping (Fuzzy Matcher)
const FIELD_SYNONYMS: Record<string, string[]> = {
    'sku_id': ['sku', 'id', 'item', 'item_no', 'product_id', 'product', 'code', 'material'],
    'price': ['retail_price', 'current_price', 'unit_price', 'msrp', 'selling_price'],
    'cost': ['unit_cost', 'standard_cost', 'base_cost', 'cost_price'],
    'revenue': ['sales', 'amount', 'total_sales', 'gross_sales'],
    'units': ['qty', 'quantity', 'volume', 'pieces', 'sold'],
    'quantity_on_hand': ['inventory', 'stock', 'qoh', 'available', 'balance'],
    'name': ['description', 'desc', 'product_name', 'title'],
    'category': ['dept', 'department', 'family', 'group', 'class'],
    'brand': ['manufacturer', 'brand_name', 'make'],
    'store_id': ['store', 'loc', 'location', 'site'],
    'date': ['transaction_date', 'day', 'time']
};

/**
 * Parses raw CSV text into an array of objects.
 * Assumes the first row is the header.
 * Limits processing to first 5000 rows to prevent browser stack overflow.
 */
export const parseCSV = (text: string): any[] => {
  if (!text) return [];
  
  const lines = text.split('\n').filter(line => line.trim() !== '');
  if (lines.length < 2) return [];

  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, '').toLowerCase().replace(/[\s_]+/g, ''));
  
  // Safety limit: only process first 5000 rows to avoid stack overflow in UI
  const processableLines = lines.slice(1, 5001); 

  return processableLines.map(line => {
    // Handle quotes properly in a real app, simplified here
    const values = line.split(',').map(v => v.trim().replace(/^"|"$/g, ''));
    const row: any = {};
    headers.forEach((header, index) => {
      // Try to convert to number if possible
      const value = values[index];
      if (value !== undefined && value !== '') {
          const num = parseFloat(value);
          row[header] = isNaN(num) ? value : num;
      } else {
          row[header] = null;
      }
    });
    return row;
  });
};

/**
 * AI Mapper: Guesses the standard field based on the raw header.
 */
export const generateColumnMapping = (rawHeaders: string[], type: DatasetType): ColumnMapping[] => {
    const standardFields = STANDARD_SCHEMA[type === 'Unknown' ? 'Product' : type] || [];
    
    return rawHeaders.map(header => {
        let bestMatch = '';
        
        // 1. Direct match
        if (standardFields.includes(header)) {
            bestMatch = header;
        } else {
            // 2. Synonym match
            for (const stdField of standardFields) {
                const synonyms = FIELD_SYNONYMS[stdField] || [];
                if (synonyms.some(syn => header.includes(syn))) {
                    bestMatch = stdField;
                    break;
                }
            }
        }

        return {
            rawHeader: header,
            standardField: bestMatch // Empty string means "Ignore" or "No Match"
        };
    });
};

/**
 * Transforms raw rows into Standardized Data using the approved mapping.
 */
export const standardizeData = (rawData: any[], mapping: ColumnMapping[]): any[] => {
    return rawData.map(row => {
        const standardRow: any = {};
        mapping.forEach(map => {
            if (map.standardField) {
                standardRow[map.standardField] = row[map.rawHeader];
            }
        });
        // Keep raw data attached for reference if needed, or just return clean data
        return standardRow;
    });
};

/**
 * Simple heuristic to detect what kind of file was uploaded based on columns.
 */
export const detectDatasetType = (data: any[]): DatasetType => {
    if (!data || data.length === 0) return 'Unknown';
    const keys = Object.keys(data[0]);
    
    // Check for Inventory specific keywords first
    if (keys.some(k => k.includes('qoh') || k.includes('stock') || k.includes('hand'))) return 'Inventory';
    
    // Check for Sales
    if (keys.some(k => k.includes('date') || k.includes('transaction') || k.includes('sales') || k.includes('revenue'))) return 'Sales';
    
    // Check for Product
    if (keys.some(k => k.includes('cost') || k.includes('category') || k.includes('price'))) return 'Product';
    
    return 'Unknown';
};

/**
 * Calculates a data quality score based on missing or null values.
 */
export const calculateDataQuality = (data: any[]): number => {
  if (!Array.isArray(data) || data.length === 0) return 0;
  
  let totalCells = 0;
  let validCells = 0;

  // Limit check to first 1000 rows for performance
  const sample = data.slice(0, 1000);

  sample.forEach(row => {
    Object.values(row).forEach(val => {
      totalCells++;
      if (val !== null && val !== undefined && val !== '' && !Number.isNaN(val)) {
        validCells++;
      }
    });
  });

  return totalCells === 0 ? 0 : parseFloat(((validCells / totalCells) * 100).toFixed(1));
};

/**
 * JOINS Multiple Standardized Datasets to create the Golden Dataset (SKUData).
 * Effectively acts as the "Feature Store" builder for the Nucleus.
 */
export const joinAndAggregateData = (
    datasetStore: Record<string, any[]>
): { skus: SKUData[], cvStats: CrossValidationResult } => {
    
    const products = datasetStore['Product'] || [];
    const sales = datasetStore['Sales'] || [];
    const inventory = datasetStore['Inventory'] || [];
    
    // Identify all Unique SKUs across all datasets (Full Outer Join logic)
    const allSkuIds = new Set<string>();
    
    const addToSet = (ds: any[]) => {
        ds.forEach(row => {
            if (row['sku_id']) allSkuIds.add(row['sku_id'].toString());
        });
    }
    
    addToSet(products);
    addToSet(sales);
    addToSet(inventory);

    // 1. Index Sales by SKU ID (Aggregate)
    const salesMap = new Map<string, { totalRevenue: number, totalUnits: number, distinctStores: Set<string> }>();
    sales.forEach(row => {
        const skuId = row['sku_id'];
        const rev = row['revenue'] || 0;
        const units = row['units'] || 1;
        const store = row['store_id'];
        
        if (skuId) {
            const sid = skuId.toString();
            const current = salesMap.get(sid) || { totalRevenue: 0, totalUnits: 0, distinctStores: new Set() };
            current.totalRevenue += rev;
            current.totalUnits += units;
            if (store) current.distinctStores.add(store);
            salesMap.set(sid, current);
        }
    });

    // 2. Index Inventory by SKU ID
    const invMap = new Map<string, any>();
    inventory.forEach(row => {
        const skuId = row['sku_id'];
        if (skuId) invMap.set(skuId.toString(), row);
    });

    // 3. Index Product Master
    const prodMap = new Map<string, any>();
    products.forEach(row => {
        const skuId = row['sku_id'];
        if (skuId) prodMap.set(skuId.toString(), row);
    });

    const joinedSkus: SKUData[] = [];
    let matchedProducts = 0;
    let missingCosts = 0;

    allSkuIds.forEach(id => {
        const prod = prodMap.get(id) || {};
        const salesData = salesMap.get(id);
        const invData = invMap.get(id) || {};

        if (prodMap.has(id)) matchedProducts++;
        
        let cost = prod['cost'] || 0;
        let currentPrice = prod['price'];
        
        // If price is missing, try to derive from average unit price in sales
        if ((!currentPrice || currentPrice === 0) && salesData && salesData.totalUnits > 0) {
            currentPrice = salesData.totalRevenue / salesData.totalUnits;
        }
        
        // STRICT: If still 0, it stays 0. No random generation.
        currentPrice = currentPrice || 0;

        if (cost === 0 && currentPrice > 0) {
            // Only try to guess cost if we at least know the price, otherwise 0
            cost = currentPrice * 0.6; 
        }
        if (cost === 0) missingCosts++;

        // STRICT: Inventory is 0 if no file uploaded
        let qoh = invData['quantity_on_hand'] || 0;

        // --- DYNAMIC FEATURE COLLECTION ---
        // Collect all other standardized keys from Product and Inventory to pass to the Feature Store
        const features: Record<string, any> = {};
        
        // From Product
        Object.keys(prod).forEach(k => {
            if (!['sku_id', 'cost', 'price', 'name', 'category'].includes(k)) {
                features[k] = prod[k];
            }
        });
        // From Inventory
        Object.keys(invData).forEach(k => {
             if (!['sku_id', 'quantity_on_hand'].includes(k)) {
                features[k] = invData[k];
            }
        });
        // Derived Features
        if (salesData) {
            features['total_lifetime_sales'] = salesData.totalRevenue;
            features['total_lifetime_units'] = salesData.totalUnits;
            features['store_presence_count'] = salesData.distinctStores.size;
        }

        joinedSkus.push({
            id: id.toString(),
            name: prod['name'] || `Item ${id}`,
            category: prod['category'] || 'General',
            currentPrice: parseFloat(currentPrice.toFixed(2)),
            cost: parseFloat(cost.toFixed(2)),
            inventory: qoh,
            elasticity: -1.0, // Default neutrality
            predictedDemand: 0,
            features
        });
    });

    // Count orphans
    let orphanCount = 0;
    for (const key of salesMap.keys()) {
        if (!prodMap.has(key)) orphanCount++;
    }

    // Match rate relative to the Master Product list (if exists), otherwise 100% of discovered
    const matchDenominator = products.length > 0 ? products.length : allSkuIds.size;
    
    return {
        skus: joinedSkus,
        cvStats: {
            matchRate: matchDenominator > 0 ? (matchedProducts / matchDenominator) * 100 : 0,
            productCount: products.length,
            salesCount: sales.length,
            orphanedSales: orphanCount,
            missingCosts
        }
    };
};

/**
 * Simulates an AutoML run with Transparency Artifacts.
 * NOW USES REAL DATA FOR FORECASTING (Simple Moving Average / Run Rate).
 */
export const runAutoMLSimulation = (skus: SKUData[]): { updatedSkus: SKUData[], models: ModelMetric[] } => {
    
    // 1. Identify Input Features from the Joined Data
    const featureKeys = ['price', 'inventory', 'cost', 'category'];
    if (skus.length > 0 && skus[0].features) {
        Object.keys(skus[0].features).forEach(k => featureKeys.push(k));
    }

    // 2. Define Candidates
    const candidates = [
        { name: 'ARIMA Baseline', type: 'ARIMA' as const, version: 'v1.0' },
        { name: 'Facebook Prophet', type: 'Prophet' as const, version: 'v2.1' },
        { name: 'XGBoost Regressor', type: 'XGBoost' as const, version: 'v3.0' }
    ];

    const models: ModelMetric[] = candidates.map((cand, idx) => {
        // Mock accuracy - in a real browser app we can't easily train these models.
        // But we can make the accuracy reflect Data Quality.
        const baseAcc = 80;
        const accuracy = parseFloat((baseAcc + (idx * 5) + (Math.random() * 5)).toFixed(1));
        const isChampion = idx === 2; // Make XGBoost Champion
        
        // Generate Feature Importance
        const featureImportance: FeatureImportance[] = featureKeys.map(f => ({
            feature: f,
            // Give higher weight to Price and Sales
            importance: (f.includes('price') || f.includes('sales')) 
                ? 0.7 + (Math.random() * 0.3) 
                : Math.random() * 0.5
        })).sort((a, b) => b.importance - a.importance).slice(0, 6); // Top 6 features

        const trainingHistory: TrainingLog[] = [];
        let currentLoss = 2.0;
        let currentAcc = 0.5;
        for (let epoch = 1; epoch <= 20; epoch++) {
            currentLoss = Math.max(0.1, currentLoss * 0.85);
            currentAcc = Math.min(accuracy / 100, currentAcc + 0.03);
            trainingHistory.push({ epoch, loss: currentLoss, accuracy: currentAcc });
        }

        return {
            id: `MOD-${idx}`,
            name: cand.name,
            type: cand.type,
            version: cand.version,
            accuracy,
            drift: parseFloat((Math.random() * 2).toFixed(2)),
            status: isChampion ? 'Production' : 'Candidate',
            isChampion,
            inputFeatures: featureKeys,
            featureImportance,
            trainingHistory,
            hyperparameters: isChampion ? {
                'learning_rate': 0.01,
                'n_estimators': 1000,
                'max_depth': 6,
                'subsample': 0.8
            } : { 'p': 2, 'd': 1, 'q': 2 } 
        };
    });

    const updatedSkus = skus.map(sku => {
        // REAL FORECAST LOGIC:
        // Use 'total_lifetime_units' from sales data to calculate a simple run-rate.
        // Assumption: Sales data provided represents history. 
        // If no units, forecast is 0.
        
        let runRate = 0;
        if (sku.features && sku.features['total_lifetime_units']) {
             // Arbitrarily assume history is ~10 weeks for demo calculation purposes if date range unknown
             // In a real app we'd calc date range from sales file.
             runRate = Math.ceil(sku.features['total_lifetime_units'] / 10); 
        }

        // Elasticity Calculation (Mocked for demo as we need extensive history to calc real elasticity)
        const elasticity = -0.5 - (Math.random() * 2.0);

        // Generate predictions derived from Run Rate
        const modelPredictions: Record<string, number> = {};
        models.forEach((model, i) => {
            let multiplier = 1.0;
            // Differentiate models slightly
            if (model.type === 'ARIMA') multiplier = 0.90; // Conservative
            if (model.type === 'Prophet') multiplier = 1.10; // Optimistic
            
            modelPredictions[model.id] = Math.ceil(runRate * multiplier);
        });

        // Champion Prediction
        const championId = models.find(m => m.isChampion)?.id || models[0].id;

        return {
            ...sku,
            predictedDemand: modelPredictions[championId], 
            modelPredictions,
            elasticity: parseFloat(elasticity.toFixed(2))
        };
    });

    return { updatedSkus, models };
};

export const mapToRevenue = (raw: any[]): RevenuePoint[] => {
  if (!Array.isArray(raw)) return [];
  
  // Strict: If no data, return empty
  if (raw.length === 0) return [];
  
  // Aggregate by Date
  const dateMap = new Map<string, number>();
  
  raw.forEach(row => {
      // Try to find a date field
      const dateVal = row['date'] || row['transaction_date'] || row['day'];
      const revVal = parseFloat(row['revenue'] || row['sales'] || row['amount'] || 0);
      
      if (dateVal && !isNaN(revVal)) {
          const current = dateMap.get(dateVal) || 0;
          dateMap.set(dateVal, current + revVal);
      }
  });

  const points: RevenuePoint[] = [];
  dateMap.forEach((revenue, date) => {
      points.push({
          name: date.toString(),
          revenue: revenue,
          projected: revenue * 1.1 // Simple projection
      });
  });

  // Safe Sort
  return points.sort((a, b) => String(a.name).localeCompare(String(b.name)));
};

// --- HIERARCHY UTILITIES ---

export interface HierarchyNode {
  value: string;
  children?: HierarchyNode[];
}

/**
 * Generates a mock hierarchy tree based on the provided configuration levels.
 * In production, this would be replaced by an API call to fetch the real product graph.
 */
export const generateMockHierarchy = (levels: HierarchyLevel[]): HierarchyNode[] => {
  const generateLevel = (depth: number, parentLabel: string): HierarchyNode[] => {
    if (depth >= levels.length) return [];
    
    const levelConfig = levels[depth];
    // Generate 2-4 items per level
    const count = Math.floor(Math.random() * 3) + 2; 
    
    return Array.from({ length: count }).map((_, i) => {
      // Create realistic-ish names (e.g., "Men A", "Men B")
      const value = `${levelConfig.label} ${String.fromCharCode(65 + i)}`; 
      return {
        value: value,
        children: generateLevel(depth + 1, value)
      };
    });
  };
  
  return generateLevel(0, 'Root');
};

/**
 * Returns valid dropdown options for a specific level, based on previous parent selections.
 */
export const getOptionsForLevel = (
  root: HierarchyNode[], 
  selections: Record<string, string>, 
  levels: HierarchyLevel[], 
  targetLevelIndex: number
): string[] => {
  let nodes = root;
  
  // Traverse down the tree following the user's selection path
  for (let i = 0; i < targetLevelIndex; i++) {
    const levelId = levels[i].id;
    const selectedValue = selections[levelId];
    
    // If a parent isn't selected yet, we can't show children
    if (!selectedValue) return [];
    
    const node = nodes.find(n => n.value === selectedValue);
    if (!node || !node.children) return [];
    
    nodes = node.children;
  }
  
  return nodes.map(n => n.value);
};