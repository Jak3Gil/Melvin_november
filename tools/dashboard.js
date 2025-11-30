// Melvin Dashboard JavaScript

class MelvinDashboard {
    constructor() {
        this.apiBase = '';
        this.controlApiBase = 'http://127.0.0.1:8081';
        this.updateInterval = null;
        this.controlUpdateInterval = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.nodeMeshes = [];
        this.edgeLines = [];
        this.graphData = null;
        
        this.init();
    }
    
    init() {
        this.setupFileDrop();
        this.setupControls();
        this.setupServiceControls();
        this.init3D();
        this.startPolling();
        this.startControlPolling();
    }
    
    setupServiceControls() {
        document.getElementById('btn-start').addEventListener('click', () => {
            this.controlService('start');
        });
        document.getElementById('btn-stop').addEventListener('click', () => {
            this.controlService('stop');
        });
        document.getElementById('btn-pause').addEventListener('click', () => {
            this.controlService('pause');
        });
        document.getElementById('btn-resume').addEventListener('click', () => {
            this.controlService('resume');
        });
    }
    
    async controlService(action) {
        try {
            const response = await fetch(`${this.controlApiBase}/api/control/${action}`);
            const data = await response.json();
            
            if (data.success) {
                this.addOutput(`✓ ${action.charAt(0).toUpperCase() + action.slice(1)}: ${data.message || 'Success'}`, 'success');
                this.updateServiceStatus(); // Refresh status
            } else {
                this.addOutput(`✗ ${action} failed: ${data.error}`, 'error');
            }
        } catch (error) {
            this.addOutput(`✗ Control API error: ${error.message}`, 'error');
        }
    }
    
    async updateServiceStatus() {
        try {
            const response = await fetch(`${this.controlApiBase}/api/status`);
            const data = await response.json();
            
            const statusText = document.getElementById('service-status-text');
            const startBtn = document.getElementById('btn-start');
            const stopBtn = document.getElementById('btn-stop');
            const pauseBtn = document.getElementById('btn-pause');
            const resumeBtn = document.getElementById('btn-resume');
            
            if (data.running) {
                statusText.textContent = 'RUNNING';
                statusText.className = 'status-value running';
                startBtn.disabled = true;
                stopBtn.disabled = false;
                pauseBtn.disabled = false;
                resumeBtn.disabled = true;
            } else {
                statusText.textContent = 'STOPPED';
                statusText.className = 'status-value stopped';
                startBtn.disabled = false;
                stopBtn.disabled = true;
                pauseBtn.disabled = true;
                resumeBtn.disabled = true;
            }
        } catch (error) {
            const statusText = document.getElementById('service-status-text');
            statusText.textContent = 'API ERROR';
            statusText.className = 'status-value';
        }
    }
    
    startControlPolling() {
        this.updateServiceStatus();
        this.controlUpdateInterval = setInterval(() => this.updateServiceStatus(), 2000);
    }
    
    setupFileDrop() {
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const fileList = document.getElementById('file-list');
        
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            this.handleFiles(e.dataTransfer.files);
        });
        
        fileInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });
    }
    
    async handleFiles(files) {
        for (let file of files) {
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                // For now, just show the file
                this.addFileToList(file.name, file.size);
                this.addOutput(`📁 Loaded: ${file.name} (${this.formatBytes(file.size)})`, 'info');
                
                // TODO: Actually feed to graph via API
                // const response = await fetch('/api/feed', {
                //     method: 'POST',
                //     body: formData
                // });
            } catch (error) {
                this.addOutput(`❌ Error loading ${file.name}: ${error.message}`, 'error');
            }
        }
    }
    
    addFileToList(name, size) {
        const fileList = document.getElementById('file-list');
        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `
            <span class="file-name">${name}</span>
            <span class="file-size">${this.formatBytes(size)}</span>
        `;
        fileList.appendChild(item);
    }
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }
    
    setupControls() {
        document.getElementById('btn-reset-view').addEventListener('click', () => {
            this.resetCamera();
        });
        
        document.getElementById('btn-toggle-labels').addEventListener('click', () => {
            // Toggle node labels
        });
        
        document.getElementById('node-size').addEventListener('input', (e) => {
            this.updateNodeSize(parseFloat(e.target.value));
        });
    }
    
    init3D() {
        const container = document.getElementById('graph-container');
        const canvas = document.getElementById('graph-canvas');
        
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75,
            container.clientWidth / container.clientHeight,
            0.1,
            10000
        );
        this.camera.position.set(0, 0, 50);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        this.renderer.setSize(container.clientWidth, container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        
        // Controls (simplified - OrbitControls needs to be loaded separately)
        // For now, use manual camera controls
        this.setupManualControls();
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
        this.scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 10);
        this.scene.add(directionalLight);
        
        // Initial graph visualization
        this.updateGraph3D({
            nodes: 0,
            edges: 0,
            node_positions: [],
            edges_list: []
        });
        
        // Handle resize
        window.addEventListener('resize', () => {
            this.camera.aspect = container.clientWidth / container.clientHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(container.clientWidth, container.clientHeight);
        });
        
        // Animation loop
        this.animate();
    }
    
    updateGraph3D(data) {
        // Clear existing
        this.nodeMeshes.forEach(mesh => this.scene.remove(mesh));
        this.edgeLines.forEach(line => this.scene.remove(line));
        this.nodeMeshes = [];
        this.edgeLines = [];
        
        if (!data.nodes || data.nodes === 0) {
            this.renderer.render(this.scene, this.camera);
            return;
        }
        
        // Create nodes (simplified - would need actual node positions from graph)
        const nodeCount = Math.min(data.nodes, 1000); // Limit for performance
        const radius = Math.cbrt(nodeCount) * 2;
        
        for (let i = 0; i < nodeCount; i++) {
            // Simple spherical distribution
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = radius * (0.5 + Math.random() * 0.5);
            
            const x = r * Math.sin(phi) * Math.cos(theta);
            const y = r * Math.sin(phi) * Math.sin(theta);
            const z = r * Math.cos(phi);
            
            const geometry = new THREE.SphereGeometry(0.3, 8, 8);
            const material = new THREE.MeshPhongMaterial({
                color: 0x22c55e,
                emissive: 0x0a4a0a,
                emissiveIntensity: 0.2
            });
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(x, y, z);
            this.scene.add(sphere);
            this.nodeMeshes.push(sphere);
        }
        
        // Create edges (simplified - would need actual edge data)
        const edgeCount = Math.min(data.edges, 5000); // Limit for performance
        
        for (let i = 0; i < Math.min(edgeCount, this.nodeMeshes.length * 2); i++) {
            if (this.nodeMeshes.length < 2) break;
            
            const src = Math.floor(Math.random() * this.nodeMeshes.length);
            const dst = Math.floor(Math.random() * this.nodeMeshes.length);
            if (src === dst) continue;
            
            const geometry = new THREE.BufferGeometry().setFromPoints([
                this.nodeMeshes[src].position,
                this.nodeMeshes[dst].position
            ]);
            const material = new THREE.LineBasicMaterial({
                color: 0x444444,
                opacity: 0.3,
                transparent: true
            });
            const line = new THREE.Line(geometry, material);
            this.scene.add(line);
            this.edgeLines.push(line);
        }
        
        this.renderer.render(this.scene, this.camera);
    }
    
    updateNodeSize(size) {
        this.nodeMeshes.forEach(mesh => {
            mesh.scale.set(size, size, size);
        });
        this.renderer.render(this.scene, this.camera);
    }
    
    setupManualControls() {
        let isDragging = false;
        let previousMousePosition = { x: 0, y: 0 };
        const canvas = this.renderer.domElement;
        
        canvas.addEventListener('mousedown', (e) => {
            isDragging = true;
            previousMousePosition = { x: e.clientX, y: e.clientY };
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            
            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;
            
            // Rotate camera around scene
            const spherical = new THREE.Spherical();
            spherical.setFromVector3(this.camera.position);
            spherical.theta -= deltaX * 0.01;
            spherical.phi += deltaY * 0.01;
            spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));
            
            this.camera.position.setFromSpherical(spherical);
            this.camera.lookAt(0, 0, 0);
            
            previousMousePosition = { x: e.clientX, y: e.clientY };
        });
        
        canvas.addEventListener('mouseup', () => {
            isDragging = false;
        });
        
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const delta = e.deltaY * 0.01;
            this.camera.position.multiplyScalar(1 + delta);
        });
    }
    
    resetCamera() {
        this.camera.position.set(0, 0, 50);
        this.camera.lookAt(0, 0, 0);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        this.renderer.render(this.scene, this.camera);
    }
    
    async fetchState() {
        try {
            const response = await fetch('/api/state');
            const data = await response.json();
            this.updateUI(data);
            this.updateGraph3D(data);
            document.getElementById('connection-status').textContent = 'Connected';
            document.getElementById('connection-status').className = 'status-badge connected';
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
        } catch (error) {
            document.getElementById('connection-status').textContent = 'Disconnected';
            document.getElementById('connection-status').className = 'status-badge disconnected';
            console.error('Error fetching state:', error);
        }
    }
    
    updateUI(data) {
        document.getElementById('stat-nodes').textContent = data.nodes.toLocaleString();
        document.getElementById('stat-edges').textContent = data.edges.toLocaleString();
        document.getElementById('stat-chaos').textContent = data.chaos.toFixed(6);
        document.getElementById('stat-activation').textContent = data.activation.toFixed(6);
        document.getElementById('stat-edge-strength').textContent = data.edge_strength.toFixed(6);
        document.getElementById('stat-active-nodes').textContent = (data.active_nodes || 0).toLocaleString();
    }
    
    addOutput(message, type = 'info') {
        const log = document.getElementById('output-log');
        const entry = document.createElement('div');
        entry.className = `output-entry ${type}`;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        log.insertBefore(entry, log.firstChild);
        
        // Keep only last 100 entries
        while (log.children.length > 100) {
            log.removeChild(log.lastChild);
        }
    }
    
    startPolling() {
        this.fetchState();
        this.updateInterval = setInterval(() => this.fetchState(), 500);
    }
    
    stopPolling() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }
}

// Initialize dashboard when page loads
let dashboard;
window.addEventListener('DOMContentLoaded', () => {
    dashboard = new MelvinDashboard();
});

