import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { Toaster } from 'react-hot-toast';

// Layout components
import Layout from './components/Layout/Layout';
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';

// Page components
import Dashboard from './pages/Dashboard';
import Hardware from './pages/Hardware';
import Brain from './pages/Brain';
import Learning from './pages/Learning';
import Logs from './pages/Logs';
import Settings from './pages/Settings';

// Context providers
import { MelvinProvider } from './contexts/MelvinContext';
import { WebSocketProvider } from './contexts/WebSocketContext';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 3,
      staleTime: 30000, // 30 seconds
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <MelvinProvider>
        <WebSocketProvider>
          <Router>
            <div className="min-h-screen bg-robot-50">
              <Layout>
                <Sidebar />
                <div className="flex-1 flex flex-col">
                  <Header />
                  <main className="flex-1 overflow-y-auto bg-robot-50 p-6">
                    <Routes>
                      <Route path="/" element={<Dashboard />} />
                      <Route path="/hardware" element={<Hardware />} />
                      <Route path="/brain" element={<Brain />} />
                      <Route path="/learning" element={<Learning />} />
                      <Route path="/logs" element={<Logs />} />
                      <Route path="/settings" element={<Settings />} />
                    </Routes>
                  </main>
                </div>
              </Layout>
              
              {/* Toast notifications */}
              <Toaster
                position="top-right"
                toastOptions={{
                  duration: 4000,
                  style: {
                    background: '#1e293b',
                    color: '#f8fafc',
                    border: '1px solid #475569',
                  },
                  success: {
                    iconTheme: {
                      primary: '#22c55e',
                      secondary: '#f8fafc',
                    },
                  },
                  error: {
                    iconTheme: {
                      primary: '#ef4444',
                      secondary: '#f8fafc',
                    },
                  },
                }}
              />
            </div>
          </Router>
        </WebSocketProvider>
      </MelvinProvider>
    </QueryClientProvider>
  );
}

export default App;
