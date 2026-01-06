import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Typography, 
  Button, 
  Grid, 
  Paper, 
  Alert,
  Snackbar,
  CircularProgress,
  Divider,
  Chip
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import PsychologyIcon from '@mui/icons-material/Psychology';
import RefreshIcon from '@mui/icons-material/Refresh';
import ModelTrainingIcon from '@mui/icons-material/ModelTraining';

// --- Services ---
import { api } from '../services/api';

// --- Components ---
// Note: Ensure you placed the new widgets in components/intelligence/
import { ForecastAccuracyWidget } from '../components/intelligence/ForecastAccuracyWidget';
import { GlassBoxModal } from '../components/intelligence/GlassBoxModal';
// FIX: Changed to Default Import to match MetricCard.tsx
import MetricCard from '../components/MetricCard';

export const IntelligenceView: React.FC = () => {
  // State
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [lastRun, setLastRun] = useState<string | null>(null);
  
  // Glass Box State
  const [inspectSku, setInspectSku] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Notification
  const [toast, setToast] = useState<{open: boolean, msg: string, type: 'success' | 'error' | 'info'}>({
    open: false, msg: '', type: 'info'
  });

  // Initial Fetch
  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    setLoading(true);
    try {
      const data = await api.ml.getMetrics();
      setMetrics(data);
    } catch (e) {
      console.error("Failed to load metrics", e);
    }
    setLoading(false);
  };

  const handleRunPipeline = async () => {
    setTraining(true);
    try {
      setToast({ open: true, msg: 'Initializing Intelligence Pipeline... (Cleanse -> Backtest -> Forecast)', type: 'info' });
      
      const res = await api.ml.triggerDemandModeling();
      
      if (res.status === 'success') {
        setToast({ open: true, msg: `Pipeline Complete. Generated ${res.nodes_generated} vectors.`, type: 'success' });
        setLastRun(new Date().toLocaleTimeString());
        loadMetrics(); // Refresh metrics
      } else {
        setToast({ open: true, msg: 'Pipeline Failed or Skipped.', type: 'error' });
      }
    } catch (e) {
      setToast({ open: true, msg: 'Error connecting to ML Engine.', type: 'error' });
    }
    setTraining(false);
  };

  const openInspector = () => {
    setInspectSku('PUMA-PALERMO-GRN');
    setDrawerOpen(true);
  };

  return (
    <Box p={3} sx={{ backgroundColor: '#f4f6f8', minHeight: '100vh' }}>
      
      {/* HEADER */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={4}>
        <Box>
          <Typography variant="h4" gutterBottom fontWeight="bold" color="#1a2027">
            Intelligence Center
          </Typography>
          <Typography variant="subtitle1" color="textSecondary">
            Manage the "Digital Brain" of your demand chain.
          </Typography>
        </Box>
        
        <Box display="flex" gap={2}>
           <Button 
            variant="outlined" 
            startIcon={<PsychologyIcon />} 
            onClick={openInspector}
          >
            Open Inspector
          </Button>
          <Button 
            variant="contained" 
            color="primary" 
            size="large"
            startIcon={training ? <CircularProgress size={20} color="inherit" /> : <PlayArrowIcon />}
            onClick={handleRunPipeline}
            disabled={training}
            sx={{ px: 4 }}
          >
            {training ? 'Running Cycle...' : 'Run Intelligence Cycle'}
          </Button>
        </Box>
      </Box>

      {/* STATUS BAR */}
      {lastRun && (
        <Alert icon={<CheckCircleIcon fontSize="inherit" />} severity="success" sx={{ mb: 3 }}>
          <strong>System Synchronized:</strong> Last full intelligence cycle completed at {lastRun}. 
          Hypercube and Audit Logs are up to date.
        </Alert>
      )}

      {/* KPI CARDS */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} md={4}>
          <MetricCard 
            title="Model Status" 
            value={metrics?.status || 'Unknown'} 
            // FIX: 'change' displays the text, 'trend' controls color/icon
            change={metrics?.mode || 'Standard'} 
            trend="neutral"
            // FIX: Pass component reference, not <Element />
            icon={AutoGraphIcon} 
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <MetricCard 
            title="Forecast Horizon" 
            value="5 Years" 
            change="Weekly Buckets" 
            trend="up"
            icon={ModelTrainingIcon}
          />
        </Grid>
        <Grid item xs={12} md={4}>
          <MetricCard 
            title="Active Models" 
            value="2" 
            // FIX: Combine subtext into change since MetricCard doesn't have subtext
            change="Tournament Mode (XGBoost vs Linear)" 
            trend="up"
            icon={PsychologyIcon}
          />
        </Grid>
      </Grid>

      <Divider sx={{ mb: 4 }} />

      {/* MAIN CONTENT AREA */}
      <Grid container spacing={3}>
        
        {/* LEFT: ACCURACY WIDGET */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 0, height: '500px', overflow: 'hidden', borderRadius: 2 }} elevation={2}>
            <ForecastAccuracyWidget />
          </Paper>
        </Grid>

        {/* RIGHT: SYSTEM HEALTH / LOGS */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, height: '500px', borderRadius: 2 }} elevation={2}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6" fontWeight="bold">Pipeline Health</Typography>
                <RefreshIcon color="action" style={{cursor: 'pointer'}} onClick={loadMetrics} />
            </Box>
            
            <Box display="flex" flexDirection="column" gap={2}>
                <Box p={2} bgcolor="#e3f2fd" borderRadius={2}>
                    <Typography variant="subtitle2" color="primary" fontWeight="bold">Hypercube Vectorization</Typography>
                    <Typography variant="body2" color="textSecondary">
                        Generating 100-point price elasticity curves for all SKUs.
                    </Typography>
                    <Chip label="Active" color="primary" size="small" sx={{ mt: 1 }} />
                </Box>

                <Box p={2} bgcolor="#f3e5f5" borderRadius={2}>
                    <Typography variant="subtitle2" color="secondary" fontWeight="bold">Glass Box Audit</Typography>
                    <Typography variant="body2" color="textSecondary">
                        Logging data quality fixes and model rationale for explainability.
                    </Typography>
                    <Chip label="Logging Enabled" color="secondary" size="small" sx={{ mt: 1 }} />
                </Box>

                <Box p={2} bgcolor="#e8f5e9" borderRadius={2}>
                    <Typography variant="subtitle2" color="success.main" fontWeight="bold">Self-Correction</Typography>
                    <Typography variant="body2" color="textSecondary">
                        Backtesting against 1-month and 3-month lag snapshots.
                    </Typography>
                     <Chip label="Online" color="success" size="small" sx={{ mt: 1 }} />
                </Box>
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* DRAWERS */}
      <GlassBoxModal 
        open={drawerOpen} 
        onClose={() => setDrawerOpen(false)} 
        skuId={inspectSku} 
      />

      {/* NOTIFICATIONS */}
      <Snackbar 
        open={toast.open} 
        autoHideDuration={6000} 
        onClose={() => setToast({ ...toast, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert severity={toast.type} variant="filled">
          {toast.msg}
        </Alert>
      </Snackbar>

    </Box>
  );
};

// Add default export to be safe if importing lazily elsewhere
export default IntelligenceView;
