import { useState, useCallback, useEffect, useRef } from 'react';
import { api } from '../services/api';
import { 
  PlanNode, 
  PlanningContext, 
  VersionType, 
  PeriodData, 
  FinancialCell 
} from '../types';

/**
 * PLANSMART MATH ENGINE (v10.19)
 * * Capabilities:
 * 1. Lock & Hold Solver (Price/Unit/Amount physics)
 * 2. Time-Series Ripple (Inventory rolling forward)
 * 3. Hierarchical Spreading (Parent -> Child distribution)
 */
export const usePlanningEngine = (config: PlanningContext) => {
  const [tree, setTree] = useState<PlanNode[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Undo/Redo Stack could go here in future
  const treeRef = useRef<PlanNode[]>([]);

  // --- 1. INITIALIZATION ---

  const fetchTree = useCallback(async () => {
    setLoading(true);
    try {
      // Calls the new PlanSmart Aggregator in Backend
      const hierarchy = JSON.stringify([config.aggregateLevel, config.anchorLevel]);
      const data = await api.orchestration.getPlanningTree(hierarchy);
      setTree(data);
      treeRef.current = data;
    } catch (err) {
      console.error("Planning Engine Crash:", err);
      setError("Failed to load Planning Tree");
    } finally {
      setLoading(false);
    }
  }, [config.aggregateLevel, config.anchorLevel]);

  useEffect(() => {
    fetchTree();
  }, [fetchTree]);

  // --- 2. THE SOLVER (Physics Engine) ---

  const solveRowPhysics = (
    data: PeriodData, 
    version: VersionType, 
    changedMetric: 'sales_units' | 'sales_amt' | 'aur' | 'receipts'
  ) => {
    const units = data.sales_units[version];
    const amt = data.sales_amt[version];
    const aur = data.aur[version];
    const rec = data.receipts[version];
    const inv = data.inventory[version];

    // RULE 1: SALES TRIANGLE (Amt = Units * AUR)
    if (changedMetric === 'sales_amt') {
      if (aur.locked && aur.value !== 0) {
        units.value = Math.round(amt.value / aur.value);
      } else if (units.locked && units.value !== 0) {
        aur.value = amt.value / units.value;
      } else {
        // Default: Hold Price, Move Units
        if (aur.value !== 0) units.value = Math.round(amt.value / aur.value);
        else if (units.value !== 0) aur.value = amt.value / units.value;
      }
    } 
    else if (changedMetric === 'sales_units') {
      if (amt.locked) {
        if (units.value !== 0) aur.value = amt.value / units.value;
      } else {
        // Default: Hold Price, Move Amount
        amt.value = Math.round(units.value * aur.value);
      }
    }
    else if (changedMetric === 'aur') {
       if (amt.locked && aur.value !== 0) {
         units.value = Math.round(amt.value / aur.value);
       } else {
         // Default: Hold Units, Move Amount
         amt.value = Math.round(units.value * aur.value);
       }
    }

    // RULE 2: INVENTORY ROLL is handled in 'rippleTime' 
    // because it requires looking at previous/next periods.
  };

  // --- 3. TIME MACHINE (Inventory Ripple) ---

  const rippleInventory = (periods: Record<string, PeriodData>, version: VersionType) => {
    // Sort periods chronologically
    const sortedKeys = Object.keys(periods).sort();
    
    let previousEOP = 0; // Or fetch BOP from DB for first period
    
    sortedKeys.forEach((key, index) => {
      const p = periods[key];
      // BOP = Previous EOP
      const bop = index === 0 ? (p.inventory[version].value + p.sales_units[version].value - p.receipts[version].value) : previousEOP; 
      
      // Physics: EOP = BOP + Receipts - Sales
      const sales = p.sales_units[version].value;
      const rec = p.receipts[version].value;
      const eop = Math.max(0, bop + rec - sales); // No negative stock
      
      p.inventory[version].value = eop;
      previousEOP = eop;
    });
  };

  // --- 4. HIERARCHY MANAGER (Recursion) ---

  const findAndUpdateNode = (
    nodes: PlanNode[], 
    targetId: string, 
    periodId: string, 
    version: VersionType,
    metric: keyof PeriodData, 
    value: number
  ): PlanNode[] => {
    return nodes.map(node => {
      // A. MATCH FOUND
      if (node.id === targetId) {
        const period = node.data[periodId];
        if (!period) return node;

        // 1. Update the Raw Value
        // @ts-ignore - Dynamic key access
        if (period[metric] && period[metric][version]) {
             // @ts-ignore
            period[metric][version].value = value;
        }

        // 2. Run Local Solver
        solveRowPhysics(period, version, metric as any);

        // 3. Ripple Time (If Inventory drivers changed)
        if (['sales_units', 'receipts'].includes(metric as string)) {
          rippleInventory(node.data, version);
        }

        // 4. Spread to Children (If not a leaf)
        if (node.children && node.children.length > 0) {
             // TODO: Implement proportional spreading logic here
             // Current: Simple implementation assumes editing leaves mostly
        }

        return { ...node };
      }

      // B. RECURSE CHILDREN
      if (node.children && node.children.length > 0) {
        const updatedChildren = findAndUpdateNode(node.children, targetId, periodId, version, metric, value);
        
        // C. AGGREGATE UP (Rollup)
        // If a child changed, we must re-sum the parent
        if (updatedChildren !== node.children) {
             const period = node.data[periodId];
             // Simple Summation Rollup for Units/Amt
             if (metric === 'sales_units' || metric === 'sales_amt' || metric === 'receipts') {
                 let sum = 0;
                 updatedChildren.forEach(child => {
                     // @ts-ignore
                     sum += child.data[periodId]?.[metric]?.[version]?.value || 0;
                 });
                 // @ts-ignore
                 if(period[metric][version]) period[metric][version].value = sum;
                 
                 // Re-solve parent physics (e.g. Recalc AUR based on new sums)
                 solveRowPhysics(period, version, metric as any);
             }
        }
        
        return { ...node, children: updatedChildren };
      }

      return node;
    });
  };

  // --- 5. PUBLIC API ---

  const updateCell = useCallback((
    nodeId: string, 
    periodId: string, 
    version: VersionType, 
    metric: 'sales_units' | 'sales_amt' | 'aur' | 'receipts', 
    value: number
  ) => {
    setTree(prevTree => {
      // Deep clone to avoid mutation bugs in complex trees
      const deepCopy = JSON.parse(JSON.stringify(prevTree));
      return findAndUpdateNode(deepCopy, nodeId, periodId, version, metric, value);
    });
  }, []);

  const toggleLock = useCallback((nodeId: string, periodId: string, version: VersionType, metric: string) => {
      setTree(prev => {
          const deepCopy = JSON.parse(JSON.stringify(prev));
          const traverse = (nodes: PlanNode[]) => {
              for (const node of nodes) {
                  if (node.id === nodeId) {
                      // @ts-ignore
                      const cell = node.data[periodId]?.[metric]?.[version];
                      if (cell) cell.locked = !cell.locked;
                      return true;
                  }
                  if (node.children) traverse(node.children);
              }
          };
          traverse(deepCopy);
          return deepCopy;
      });
  }, []);

  return {
    tree,
    loading,
    error,
    updateCell,
    toggleLock,
    refresh: fetchTree
  };
};
