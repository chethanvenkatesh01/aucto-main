import { useState, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { DecisionState, DecisionStatus, TriageLevel, BackendDecisionResponse, ProposedAction } from '../types';

export const useDecisionEngine = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const executeDecision = useCallback(async (
    payload: Record<string, any>, 
    constraints: { risk_tolerance: string; budget_cap: number }
  ): Promise<BackendDecisionResponse> => {
    setLoading(true);
    setError(null);

    // 1. Construct the V5 Opaque Envelope
    const requestBody = {
      universal_context: {
        request_id: uuidv4(),
        user_role: 'admin', 
        trace_id: uuidv4(),
      },
      constraints: {
        risk_tolerance: constraints.risk_tolerance,
        budget_cap: constraints.budget_cap,
      },
      domain_context: {
        domain: 'retail', // Routing to Retail Cartridge
        payload,
      },
    };

    try {
      // 2. Hit the Kernel
      // Ensure your Python backend is running on port 8000
      const response = await fetch('http://localhost:8000/decision/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Kernel Error: ${response.statusText}`);
      }

      const data: BackendDecisionResponse = await response.json();
      return data;

    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown OS Error';
      setError(message);
      console.error("ADOS Kernel Call Failed:", err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { executeDecision, loading, error };
};
