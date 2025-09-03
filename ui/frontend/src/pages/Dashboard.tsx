import React from 'react';
import { useQuery } from 'react-query';
import { 
  CpuChipIcon, 
  CogIcon, 
  BrainIcon, 
  AcademicCapIcon,
  ServerIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon
} from '@heroicons/react/24/outline';

// Components
import StatusCard from '../components/Dashboard/StatusCard';
import SystemHealthChart from '../components/Dashboard/SystemHealthChart';
import MotorStatusGrid from '../components/Dashboard/MotorStatusGrid';
import SensorStatusGrid from '../components/Dashboard/SensorStatusGrid';
import RecentActivity from '../components/Dashboard/RecentActivity';
import QuickActions from '../components/Dashboard/QuickActions';

// API and types
import { getSystemStatus, getHardwareStatus, getBrainStatus } from '../api/melvin';
import { SystemStatus, HardwareStatus, BrainStatus } from '../types/melvin';

const Dashboard: React.FC = () => {
  // Fetch system data
  const { data: systemStatus, isLoading: systemLoading } = useQuery<SystemStatus>(
    'systemStatus',
    getSystemStatus,
    { refetchInterval: 5000 }
  );

  const { data: hardwareStatus, isLoading: hardwareLoading } = useQuery<HardwareStatus>(
    'hardwareStatus',
    getHardwareStatus,
    { refetchInterval: 3000 }
  );

  const { data: brainStatus, isLoading: brainLoading } = useQuery<BrainStatus>(
    'brainStatus',
    getBrainStatus,
    { refetchInterval: 10000 }
  );

  const isLoading = systemLoading || hardwareLoading || brainLoading;

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'RUNNING':
        return 'success';
      case 'PAUSED':
        return 'warning';
      case 'ERROR':
        return 'danger';
      case 'INITIALIZING':
        return 'robot';
      default:
        return 'robot';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'RUNNING':
        return <CheckCircleIcon className="h-6 w-6 text-success-500" />;
      case 'PAUSED':
        return <ClockIcon className="h-6 w-6 text-warning-500" />;
      case 'ERROR':
        return <ExclamationTriangleIcon className="h-6 w-6 text-danger-500" />;
      case 'INITIALIZING':
        return <CogIcon className="h-6 w-6 text-robot-500 animate-spin" />;
      default:
        return <CogIcon className="h-6 w-6 text-robot-500" />;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-melvin-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-robot-900">Dashboard</h1>
          <p className="text-robot-600 mt-1">
            Melvin System Overview - {systemStatus?.timestamp || 'Loading...'}
          </p>
        </div>
        <QuickActions />
      </div>

      {/* Status Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatusCard
          title="System Status"
          value={systemStatus?.status || 'UNKNOWN'}
          icon={getStatusIcon(systemStatus?.status || 'UNKNOWN')}
          color={getStatusColor(systemStatus?.status || 'UNKNOWN')}
          change={systemStatus?.uptime || '0s'}
          changeLabel="Uptime"
        />
        
        <StatusCard
          title="Hardware"
          value={hardwareStatus?.motors_ok && hardwareStatus?.sensors_ok ? 'OK' : 'ISSUES'}
          icon={<CpuChipIcon className="h-6 w-6 text-melvin-500" />}
          color={hardwareStatus?.motors_ok && hardwareStatus?.sensors_ok ? 'success' : 'warning'}
          change={`${hardwareStatus?.active_motors || 0} motors`}
          changeLabel="Active"
        />
        
        <StatusCard
          title="Brain Graph"
          value={brainStatus?.total_nodes ? `${brainStatus.total_nodes} nodes` : '0 nodes'}
          icon={<BrainIcon className="h-6 w-6 text-melvin-500" />}
          color="melvin"
          change={`${brainStatus?.active_connections || 0} connections`}
          changeLabel="Active"
        />
        
        <StatusCard
          title="Learning Engine"
          value={brainStatus?.learning_rules ? `${brainStatus.learning_rules} rules` : '0 rules'}
          icon={<AcademicCapIcon className="h-6 w-6 text-melvin-500" />}
          color="melvin"
          change={`${brainStatus?.confidence || 0}%`}
          changeLabel="Confidence"
        />
      </div>

      {/* Charts and Monitoring */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-robot-200 p-6">
          <h3 className="text-lg font-semibold text-robot-900 mb-4">System Health</h3>
          <SystemHealthChart />
        </div>
        
        <div className="bg-white rounded-lg shadow-sm border border-robot-200 p-6">
          <h3 className="text-lg font-semibold text-robot-900 mb-4">Recent Activity</h3>
          <RecentActivity />
        </div>
      </div>

      {/* Hardware Status Grids */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-sm border border-robot-200 p-6">
          <h3 className="text-lg font-semibold text-robot-900 mb-4">Motor Status</h3>
          <MotorStatusGrid />
        </div>
        
        <div className="bg-white rounded-lg shadow-sm border border-robot-200 p-6">
          <h3 className="text-lg font-semibold text-robot-900 mb-4">Sensor Status</h3>
          <SensorStatusGrid />
        </div>
      </div>

      {/* System Information */}
      <div className="bg-white rounded-lg shadow-sm border border-robot-200 p-6">
        <h3 className="text-lg font-semibold text-robot-900 mb-4">System Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h4 className="font-medium text-robot-700 mb-2">Version</h4>
            <p className="text-robot-600 font-mono text-sm">
              {systemStatus?.version || '1.0.0'}
            </p>
          </div>
          <div>
            <h4 className="font-medium text-robot-700 mb-2">Build Date</h4>
            <p className="text-robot-600 font-mono text-sm">
              {systemStatus?.build_date || 'Unknown'}
            </p>
          </div>
          <div>
            <h4 className="font-medium text-robot-700 mb-2">Platform</h4>
            <p className="text-robot-600 font-mono text-sm">
              {systemStatus?.platform || 'Jetson Orin AGX'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
