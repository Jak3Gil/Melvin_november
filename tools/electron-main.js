const { app, BrowserWindow, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow = null;
let dashboardProcess = null;
const DASHBOARD_PORT = 8080;
const DASHBOARD_HOST = '127.0.0.1';

function createWindow() {
    // Create the browser window
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 800,
        minHeight: 600,
        backgroundColor: '#0a0a0a',
        titleBarStyle: 'default',
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            enableRemoteModule: false
        },
        icon: path.join(__dirname, 'icon.png') // Optional icon
    });

    // Start Python dashboard server
    startDashboardServer();

    // Wait a moment for server to start, then load dashboard
    setTimeout(() => {
        const url = `http://${DASHBOARD_HOST}:${DASHBOARD_PORT}`;
        mainWindow.loadURL(url);
        
        // Open DevTools in development (optional)
        if (process.env.NODE_ENV === 'development') {
            mainWindow.webContents.openDevTools();
        }
    }, 2000);

    // Handle window closed
    mainWindow.on('closed', () => {
        mainWindow = null;
        stopDashboardServer();
    });

    // Handle external links
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });
}

function startDashboardServer() {
    const dashboardScript = path.join(__dirname, 'melvin_dashboard.py');
    const brainPath = process.env.MELVIN_BRAIN || '/tmp/melvin_brain.m';
    
    console.log('Starting Melvin Dashboard server...');
    console.log(`Brain: ${brainPath}`);
    console.log(`Port: ${DASHBOARD_PORT}`);
    
    dashboardProcess = spawn('python3', [
        dashboardScript,
        '--brain', brainPath,
        '--port', String(DASHBOARD_PORT),
        '--host', DASHBOARD_HOST
    ], {
        cwd: __dirname,
        stdio: 'inherit'
    });

    dashboardProcess.on('error', (err) => {
        console.error('Failed to start dashboard server:', err);
        if (mainWindow) {
            mainWindow.loadURL(`data:text/html,<h1>Error</h1><p>Failed to start dashboard server: ${err.message}</p>`);
        }
    });

    dashboardProcess.on('exit', (code) => {
        console.log(`Dashboard server exited with code ${code}`);
        if (code !== 0 && code !== null) {
            console.error('Dashboard server crashed!');
        }
    });
}

function stopDashboardServer() {
    if (dashboardProcess) {
        console.log('Stopping dashboard server...');
        dashboardProcess.kill();
        dashboardProcess = null;
    }
}

// App event handlers
app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    stopDashboardServer();
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('before-quit', () => {
    stopDashboardServer();
});

// Handle app termination
process.on('SIGINT', () => {
    stopDashboardServer();
    app.quit();
});

process.on('SIGTERM', () => {
    stopDashboardServer();
    app.quit();
});

