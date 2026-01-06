import React, { useEffect, useState } from 'react';
import { api } from '../../services/api';
import { 
  Card, 
  CardHeader, 
  CardContent, 
  Typography, 
  Select, 
  MenuItem, 
  LinearProgress,
  Box,
  Tooltip,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Chip
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';

interface AccuracyScore {
  lag: string;
  wmape: number;
  accuracy: number;
  bias: number;
  health: 'GOOD' | 'RISK' | 'POOR';
}

interface NodeAccuracy {
  id: string;
  name: string;
  metrics: Record<string, AccuracyScore>;
}

export const ForecastAccuracyWidget: React.FC = () => {
  const [selectedNode, setSelectedNode] = useState<string>('GLOBAL');
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [nodes, setNodes] = useState<{id: string, name: string}[]>([
    { id: 'GLOBAL', name: 'Global Enterprise' },
    { id: 'DIV-FOOTWEAR', name: 'Div: Footwear' },
    { id: 'DIV-APPAREL', name: 'Div: Apparel' }
  ]);

  useEffect(() => {
    fetchAccuracy(selectedNode);
  }, [selectedNode]);

  const fetchAccuracy = async (nodeId: string) => {
    setLoading(true);
    const result = await api.ml.getForecastAccuracy(nodeId);
    setData(result);
    setLoading(false);
  };

  const getHealthColor = (score: number) => {
    // WMAPE: Lower is better
    if (score <= 0.20) return '#e6f4ea'; // Green (Good)
    if (score <= 0.40) return '#fef7e0'; // Yellow (Warning)
    return '#fce8e6'; // Red (Risk)
  };

  const getHealthTextColor = (score: number) => {
    if (score <= 0.20) return '#137333';
    if (score <= 0.40) return '#b06000';
    return '#c5221f';
  };

  const renderCell = (metrics: Record<string, AccuracyScore>, lag: string) => {
    const score = metrics?.[lag];
    if (!score) return <TableCell>-</TableCell>;

    return (
      <TableCell 
        align="center" 
        sx={{ 
          bgcolor: getHealthColor(score.wmape),
          color: getHealthTextColor(score.wmape),
          fontWeight: 'bold',
          border: '1px solid #eee'
        }}
      >
        <Tooltip title={`Bias: ${score.bias > 0 ? '+' : ''}${(score.bias * 100).toFixed(1)}% (Over/Under)`}>
          <span>{score.accuracy}%</span>
        </Tooltip>
      </TableCell>
    );
  };

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardHeader 
        title={
          <Box display="flex" alignItems="center" gap={2}>
            <Typography variant="h6">Forecast Reliability Matrix</Typography>
            <Select 
              size="small" 
              value={selectedNode}
              onChange={(e) => setSelectedNode(e.target.value)}
              sx={{ minWidth: 200 }}
            >
              {nodes.map(n => <MenuItem key={n.id} value={n.id}>{n.name}</MenuItem>)}
            </Select>
          </Box>
        }
        subheader="WMAPE Accuracy across Time Horizons (Self-Correcting)"
        action={
          <Tooltip title="We compare past forecasts (1mo, 3mo lags) against actuals to grade model performance. Green = Trust, Red = Intervene.">
            <InfoIcon color="action" />
          </Tooltip>
        }
      />
      {loading && <LinearProgress />}
      
      <CardContent sx={{ flexGrow: 1, overflow: 'auto', p: 0 }}>
        {!loading && data && (
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell>Hierarchy Node</TableCell>
                <TableCell align="center">Last Month<br/><span style={{fontSize: '0.75rem', color: '#666'}}>(Lag 4W)</span></TableCell>
                <TableCell align="center">Last Quarter<br/><span style={{fontSize: '0.75rem', color: '#666'}}>(Lag 12W)</span></TableCell>
                <TableCell align="center">Bias Trend</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {/* Parent Row */}
              <TableRow>
                <TableCell sx={{ fontWeight: 'bold' }}>
                  {selectedNode === 'GLOBAL' ? 'Global Enterprise' : selectedNode}
                  <Chip size="small" label="Selected" sx={{ ml: 1, height: 20 }} />
                </TableCell>
                {renderCell(data.scores, '1_MONTH')}
                {renderCell(data.scores, '3_MONTH')}
                <TableCell align="center">
                  {data.scores?.['1_MONTH']?.bias > 0 ? '🔼 Over' : '🔽 Under'}
                </TableCell>
              </TableRow>

              {/* Children Rows */}
              {data.children_scores?.map((child: NodeAccuracy) => (
                <TableRow key={child.id} hover>
                  <TableCell sx={{ pl: 4 }}>{child.name}</TableCell>
                  {renderCell(child.metrics, '1_MONTH')}
                  {renderCell(child.metrics, '3_MONTH')}
                  <TableCell align="center" sx={{ color: '#666' }}>
                     {child.metrics?.['1_MONTH']?.bias ? (child.metrics['1_MONTH'].bias * 100).toFixed(1) + '%' : '-'}
                  </TableCell>
                </TableRow>
              ))}
              
              {(!data.children_scores || data.children_scores.length === 0) && (
                <TableRow>
                  <TableCell colSpan={4} align="center" sx={{ py: 3, color: '#999' }}>
                    No children nodes found for this level.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
};