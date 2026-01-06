import React, { useEffect, useState } from 'react';
import { api } from '../../services/api';
import {
  Drawer,
  Box,
  Typography,
  IconButton,
  Divider,
  CircularProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Paper,
  Grid,
  LinearProgress,
  Alert
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import VerifiedUserIcon from '@mui/icons-material/VerifiedUser';
import WarningIcon from '@mui/icons-material/Warning';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import PsychologyIcon from '@mui/icons-material/Psychology';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';

interface GlassBoxProps {
  open: boolean;
  onClose: () => void;
  skuId: string | null;
}

interface AuditLog {
  run_id: string;
  generated_at: string;
  data_health: {
    score: number;
    log: string[];
  };
  model_transparency: {
    features_used: string[];
    tournament_scoreboard: Record<string, number> | string; // Handle legacy string case
  };
  drivers: Record<string, number>;
}

export const GlassBoxModal: React.FC<GlassBoxProps> = ({ open, onClose, skuId }) => {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<AuditLog | null>(null);

  useEffect(() => {
    if (open && skuId) {
      fetchExplanation(skuId);
    }
  }, [open, skuId]);

  const fetchExplanation = async (id: string) => {
    setLoading(true);
    const result = await api.ml.getForecastExplanation(id);
    // Safety check for empty result
    if (result && result.run_id) {
        setData(result);
    } else {
        setData(null);
    }
    setLoading(false);
  };

  // Safe parsing for scoreboard
  const getScoreboard = () => {
    if (!data?.model_transparency?.tournament_scoreboard) return {};
    const sb = data.model_transparency.tournament_scoreboard;
    // If it came back as a JSON string by mistake, parse it
    if (typeof sb === 'string') {
        try { return JSON.parse(sb); } catch { return {}; }
    }
    return sb;
  };
  
  const scoreboard = getScoreboard();
  const winnerName = scoreboard['Winner'] || 'Unknown';

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      PaperProps={{ sx: { width: '450px', p: 3, backgroundColor: '#f9fafb' } }}
    >
      {/* HEADER */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Box>
            <Typography variant="h6" fontWeight="bold" color="primary">
              Glass Box Inspector
            </Typography>
            <Typography variant="caption" color="textSecondary">
              Auditing Forecast for: {skuId}
            </Typography>
        </Box>
        <IconButton onClick={onClose}><CloseIcon /></IconButton>
      </Box>
      
      <Divider sx={{ mb: 3 }} />

      {loading && (
        <Box display="flex" justifyContent="center" mt={10}>
          <CircularProgress />
        </Box>
      )}

      {!loading && !data && (
        <Alert severity="info">
            No audit log found for this SKU. Run the Intelligence Pipeline to generate insights.
        </Alert>
      )}

      {!loading && data && (
        <Box display="flex" flexDirection="column" gap={3}>
          
          {/* 1. DATA HEALTH CARD */}
          <Paper elevation={0} sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 2 }}>
            <Box display="flex" alignItems="center" gap={1} mb={1}>
                <VerifiedUserIcon color={data.data_health.score > 80 ? "success" : "warning"} />
                <Typography variant="subtitle1" fontWeight="bold">Input Data Health</Typography>
                <Chip 
                    label={`${data.data_health.score}/100`} 
                    color={data.data_health.score > 80 ? "success" : "warning"} 
                    size="small" 
                    sx={{ ml: 'auto', fontWeight: 'bold' }}
                />
            </Box>
            <Typography variant="body2" color="textSecondary" mb={2}>
                Quality of historical data used for training.
            </Typography>
            
            {data.data_health.log.length === 0 ? (
                <Typography variant="caption" fontStyle="italic">No issues detected.</Typography>
            ) : (
                <List dense disablePadding sx={{ bgcolor: '#fff', borderRadius: 1 }}>
                    {data.data_health.log.map((entry, idx) => (
                        <ListItem key={idx} sx={{ py: 0.5 }}>
                            <ListItemIcon sx={{ minWidth: 30 }}>
                                <WarningIcon fontSize="small" color="action" />
                            </ListItemIcon>
                            <ListItemText 
                                primary={entry} 
                                primaryTypographyProps={{ variant: 'caption', color: 'textPrimary' }} 
                            />
                        </ListItem>
                    ))}
                </List>
            )}
          </Paper>

          {/* 2. TOURNAMENT RESULTS */}
          <Paper elevation={0} sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 2 }}>
            <Box display="flex" alignItems="center" gap={1} mb={2}>
                <AutoGraphIcon color="primary" />
                <Typography variant="subtitle1" fontWeight="bold">Model Tournament</Typography>
            </Box>
            
            <Box display="flex" flexDirection="column" gap={1}>
                {Object.entries(scoreboard).map(([model, score]) => {
                    if (model === 'Winner' || model === 'Reason') return null;
                    const isWinner = model === winnerName;
                    const errorPct = (Number(score) * 100).toFixed(1);
                    
                    return (
                        <Box 
                            key={model} 
                            display="flex" 
                            justifyContent="space-between" 
                            alignItems="center"
                            sx={{ 
                                p: 1, 
                                borderRadius: 1,
                                bgcolor: isWinner ? '#e3f2fd' : 'transparent',
                                border: isWinner ? '1px solid #2196f3' : 'none'
                            }}
                        >
                            <Box>
                                <Typography variant="body2" fontWeight={isWinner ? 'bold' : 'normal'}>
                                    {model}
                                </Typography>
                                {isWinner && <Chip label="WINNER" size="small" color="primary" sx={{ height: 16, fontSize: '0.6rem' }} />}
                            </Box>
                            <Typography variant="body2" color="textSecondary">
                                Error: {errorPct}%
                            </Typography>
                        </Box>
                    );
                })}
            </Box>
          </Paper>

          {/* 3. DRIVER ANALYSIS */}
          <Paper elevation={0} sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 2 }}>
            <Box display="flex" alignItems="center" gap={1} mb={2}>
                <PsychologyIcon color="secondary" />
                <Typography variant="subtitle1" fontWeight="bold">Why this Forecast?</Typography>
            </Box>
            
            <Grid container spacing={2}>
                {Object.entries(data.drivers).map(([driver, weight]) => (
                    <Grid item xs={12} key={driver}>
                        <Box display="flex" justifyContent="space-between" mb={0.5}>
                            <Typography variant="caption" fontWeight="bold">{driver}</Typography>
                            <Typography variant="caption">{(weight * 100).toFixed(0)}% Impact</Typography>
                        </Box>
                        <LinearProgress 
                            variant="determinate" 
                            value={weight * 100} 
                            sx={{ height: 8, borderRadius: 4 }} 
                            color={driver === 'Price' ? "error" : "primary"}
                        />
                    </Grid>
                ))}
            </Grid>
          </Paper>

          {/* METADATA FOOTER */}
          <Box mt="auto" pt={2}>
            <Typography variant="caption" display="block" color="textSecondary" align="center">
                Run ID: {data.run_id}
            </Typography>
            <Typography variant="caption" display="block" color="textSecondary" align="center">
                Generated: {new Date(data.generated_at).toLocaleString()}
            </Typography>
          </Box>
          
        </Box>
      )}
    </Drawer>
  );
};
