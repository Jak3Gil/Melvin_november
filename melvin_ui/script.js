// Melvin Growth Explorer JavaScript
class MelvinExplorer {
    constructor() {
        this.data = {
            cycles: [],
            concepts: [],
            connections: [],
            growthStats: {}
        };
        this.currentTab = 'timeline';
        this.graphSimulation = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadData();
    }
    
    setupEventListeners() {
        // Tab switching
        document.getElementById('timelineTab').addEventListener('click', () => this.switchTab('timeline'));
        document.getElementById('graphTab').addEventListener('click', () => this.switchTab('graph'));
        document.getElementById('analyticsTab').addEventListener('click', () => this.switchTab('analytics'));
        
        // Buttons
        document.getElementById('refreshBtn').addEventListener('click', () => this.loadData());
        document.getElementById('exportBtn').addEventListener('click', () => this.exportReport());
        document.getElementById('resetGraph').addEventListener('click', () => this.resetGraph());
        document.getElementById('exportGraph').addEventListener('click', () => this.exportGraph());
        
        // Filters
        document.getElementById('cycleFilter').addEventListener('change', (e) => this.filterCycles(e.target.value));
    }
    
    async loadData() {
        this.showLoading(true);
        
        try {
            // Simulate loading data from files (in real implementation, this would be API calls)
            await this.simulateDataLoading();
            this.updateUI();
        } catch (error) {
            console.error('Error loading data:', error);
            this.showError('Failed to load data');
        } finally {
            this.showLoading(false);
        }
    }
    
    async simulateDataLoading() {
        // Simulate loading evolution log
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Generate sample data for demonstration
        this.data.cycles = this.generateSampleCycles(1000);
        this.data.concepts = this.generateSampleConcepts(50);
        this.data.connections = this.generateSampleConnections(100);
        this.data.growthStats = this.generateSampleGrowthStats();
    }
    
    generateSampleCycles(count) {
        const cycles = [];
        const inputTypes = ['raw', 'conceptual', 'hybrid'];
        const drivers = ['dopamine', 'serotonin', 'endorphins'];
        
        for (let i = 1; i <= count; i++) {
            const inputType = inputTypes[Math.floor(Math.random() * inputTypes.length)];
            const driver = drivers[Math.floor(Math.random() * drivers.length)];
            
            cycles.push({
                cycle_id: i,
                input_type: inputType,
                input_content: `Sample ${inputType} input ${i}`,
                hypotheses: [`Hypothesis ${i}A`, `Hypothesis ${i}B`],
                hypothesis_confidences: [0.3 + Math.random() * 0.4, 0.4 + Math.random() * 0.4],
                validation_confirmed: Math.floor(Math.random() * 3),
                validation_refuted: Math.floor(Math.random() * 2),
                validation_uncertain: Math.floor(Math.random() * 2),
                dominant_driver: driver,
                strengthened_concepts: [`concept${i}_a`, `concept${i}_b`],
                weakened_concepts: [`old_concept${i}`],
                meta_learning_notes: [`Applied reinforcement learning for cycle ${i}`],
                overall_confidence: 0.2 + Math.random() * 0.6,
                timestamp: Date.now() - (count - i) * 1000,
                concepts_learned: Math.floor(Math.random() * 3),
                connections_created: Math.floor(Math.random() * 2),
                cache_hit_rate: Math.random() * 0.5,
                ollama_calls: Math.floor(Math.random() * 2)
            });
        }
        
        return cycles;
    }
    
    generateSampleConcepts(count) {
        const concepts = [];
        
        for (let i = 1; i <= count; i++) {
            concepts.push({
                concept: `concept_${i}`,
                definition: `Definition for concept ${i}`,
                activation: 0.1 + Math.random() * 0.9,
                importance: 0.1 + Math.random() * 0.9,
                access_count: Math.floor(Math.random() * 100),
                usage_frequency: Math.random() * 2.0,
                validation_successes: Math.floor(Math.random() * 50),
                validation_failures: Math.floor(Math.random() * 20),
                first_seen: Date.now() - Math.random() * 86400000,
                last_accessed: Date.now() - Math.random() * 3600000
            });
        }
        
        return concepts;
    }
    
    generateSampleConnections(count) {
        const connections = [];
        
        for (let i = 1; i <= count; i++) {
            const from = Math.floor(Math.random() * 50) + 1;
            const to = Math.floor(Math.random() * 50) + 1;
            
            connections.push({
                from_concept: `concept_${from}`,
                to_concept: `concept_${to}`,
                weight: 0.1 + Math.random() * 0.9,
                connection_type: Math.floor(Math.random() * 4),
                access_count: Math.floor(Math.random() * 20),
                usage_frequency: Math.random() * 1.0,
                first_created: Date.now() - Math.random() * 86400000,
                last_accessed: Date.now() - Math.random() * 3600000
            });
        }
        
        return connections;
    }
    
    generateSampleGrowthStats() {
        return {
            total_cycles: 1000,
            avg_confidence_start: 0.3,
            avg_confidence_end: 0.7,
            total_pruning_events: 150,
            total_merging_events: 75,
            final_cache_hit_rate: 0.65,
            total_ollama_calls: 200,
            driver_dominance_count: {
                dopamine: 350,
                serotonin: 400,
                endorphins: 250
            }
        };
    }
    
    updateUI() {
        this.updateSummaryCards();
        this.updateDriverDashboard();
        this.updateTimeline();
        
        if (this.currentTab === 'graph') {
            this.updateGraph();
        } else if (this.currentTab === 'analytics') {
            this.updateAnalytics();
        }
    }
    
    updateSummaryCards() {
        const stats = this.data.growthStats;
        
        document.getElementById('totalCycles').textContent = stats.total_cycles.toLocaleString();
        document.getElementById('avgConfidence').textContent = stats.avg_confidence_end.toFixed(3);
        document.getElementById('totalConcepts').textContent = this.data.concepts.length.toLocaleString();
        document.getElementById('cacheHitRate').textContent = (stats.final_cache_hit_rate * 100).toFixed(1) + '%';
    }
    
    updateDriverDashboard() {
        const stats = this.data.growthStats;
        const total = stats.total_cycles;
        
        // Update dopamine
        const dopaminePct = (stats.driver_dominance_count.dopamine / total) * 100;
        document.getElementById('dopamineBar').style.width = dopaminePct + '%';
        document.getElementById('dopamineValue').textContent = dopaminePct.toFixed(1) + '%';
        
        // Update serotonin
        const serotoninPct = (stats.driver_dominance_count.serotonin / total) * 100;
        document.getElementById('serotoninBar').style.width = serotoninPct + '%';
        document.getElementById('serotoninValue').textContent = serotoninPct.toFixed(1) + '%';
        
        // Update endorphins
        const endorphinsPct = (stats.driver_dominance_count.endorphins / total) * 100;
        document.getElementById('endorphinsBar').style.width = endorphinsPct + '%';
        document.getElementById('endorphinsValue').textContent = endorphinsPct.toFixed(1) + '%';
    }
    
    updateTimeline() {
        const container = document.getElementById('cycleTimeline');
        const cycles = this.data.cycles.slice(-50); // Show last 50 by default
        
        container.innerHTML = '';
        
        cycles.forEach(cycle => {
            const cycleCard = this.createCycleCard(cycle);
            container.appendChild(cycleCard);
        });
    }
    
    createCycleCard(cycle) {
        const card = document.createElement('div');
        card.className = 'cycle-card bg-white border border-gray-200 rounded-lg p-4';
        
        const confidencePct = (cycle.overall_confidence * 100).toFixed(1);
        const driverEmoji = {
            dopamine: '🧬',
            serotonin: '🔗',
            endorphins: '😊'
        };
        
        card.innerHTML = `
            <div class="flex justify-between items-start mb-3">
                <div class="flex items-center">
                    <span class="text-lg mr-2">${driverEmoji[cycle.dominant_driver]}</span>
                    <span class="font-semibold text-gray-900">Cycle ${cycle.cycle_id}</span>
                    <span class="ml-2 px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800">${cycle.input_type}</span>
                </div>
                <span class="text-sm text-gray-500">${new Date(cycle.timestamp).toLocaleTimeString()}</span>
            </div>
            
            <div class="mb-3">
                <p class="text-sm text-gray-700 mb-2">${cycle.input_content}</p>
                <div class="flex items-center">
                    <span class="text-xs text-gray-500 mr-2">Confidence:</span>
                    <div class="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                        <div class="confidence-bar bg-blue-500 h-2 rounded-full" style="width: ${confidencePct}%"></div>
                    </div>
                    <span class="text-xs font-medium text-gray-700">${confidencePct}%</span>
                </div>
            </div>
            
            <div class="grid grid-cols-2 gap-4 text-xs">
                <div>
                    <span class="text-gray-500">Learned:</span>
                    <span class="font-medium">${cycle.concepts_learned} concepts</span>
                </div>
                <div>
                    <span class="text-gray-500">Created:</span>
                    <span class="font-medium">${cycle.connections_created} connections</span>
                </div>
                <div>
                    <span class="text-gray-500">Validated:</span>
                    <span class="font-medium text-green-600">${cycle.validation_confirmed} ✓</span>
                </div>
                <div>
                    <span class="text-gray-500">Refuted:</span>
                    <span class="font-medium text-red-600">${cycle.validation_refuted} ✗</span>
                </div>
            </div>
        `;
        
        return card;
    }
    
    updateGraph() {
        const svg = d3.select('#graphSvg');
        svg.selectAll('*').remove();
        
        const width = svg.node().clientWidth;
        const height = svg.node().clientHeight;
        
        // Create force simulation
        this.graphSimulation = d3.forceSimulation(this.data.concepts.slice(0, 20)) // Limit to 20 nodes for performance
            .force('link', d3.forceLink(this.data.connections).id(d => d.concept))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collision', d3.forceCollide().radius(30));
        
        // Create links
        const link = svg.append('g')
            .selectAll('line')
            .data(this.data.connections.slice(0, 30)) // Limit connections
            .enter().append('line')
            .attr('class', 'connection-line')
            .attr('stroke', '#999')
            .attr('stroke-width', d => Math.sqrt(d.weight) * 3);
        
        // Create nodes
        const node = svg.append('g')
            .selectAll('circle')
            .data(this.data.concepts.slice(0, 20))
            .enter().append('circle')
            .attr('class', 'concept-node')
            .attr('r', d => Math.sqrt(d.importance) * 8 + 5)
            .attr('fill', d => this.getConfidenceColor(d))
            .attr('stroke', '#fff')
            .attr('stroke-width', 2)
            .call(d3.drag()
                .on('start', this.dragstarted.bind(this))
                .on('drag', this.dragged.bind(this))
                .on('end', this.dragended.bind(this)));
        
        // Add labels
        const label = svg.append('g')
            .selectAll('text')
            .data(this.data.concepts.slice(0, 20))
            .enter().append('text')
            .text(d => d.concept.replace('concept_', ''))
            .attr('font-size', '10px')
            .attr('text-anchor', 'middle')
            .attr('dy', '.35em');
        
        // Update positions
        this.graphSimulation.on('tick', () => {
            link
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y);
            
            node
                .attr('cx', d => d.x)
                .attr('cy', d => d.y);
            
            label
                .attr('x', d => d.x)
                .attr('y', d => d.y);
        });
    }
    
    getConfidenceColor(concept) {
        const confidence = concept.validation_successes / (concept.validation_successes + concept.validation_failures + 1);
        if (confidence > 0.7) return '#10b981'; // Green
        if (confidence > 0.4) return '#f59e0b'; // Yellow
        return '#ef4444'; // Red
    }
    
    updateAnalytics() {
        // This would integrate with Chart.js or similar for real charts
        console.log('Analytics charts would be rendered here');
    }
    
    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('border-blue-500', 'text-blue-600');
            btn.classList.add('border-transparent', 'text-gray-500');
        });
        
        document.getElementById(tabName + 'Tab').classList.remove('border-transparent', 'text-gray-500');
        document.getElementById(tabName + 'Tab').classList.add('border-blue-500', 'text-blue-600');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.add('hidden');
        });
        document.getElementById(tabName + 'Content').classList.remove('hidden');
        
        this.currentTab = tabName;
        
        // Update content when switching to graph or analytics
        if (tabName === 'graph') {
            setTimeout(() => this.updateGraph(), 100);
        } else if (tabName === 'analytics') {
            this.updateAnalytics();
        }
    }
    
    filterCycles(count) {
        const cycles = count === 'all' ? this.data.cycles : this.data.cycles.slice(-parseInt(count));
        const container = document.getElementById('cycleTimeline');
        
        container.innerHTML = '';
        cycles.forEach(cycle => {
            const cycleCard = this.createCycleCard(cycle);
            container.appendChild(cycleCard);
        });
    }
    
    exportReport() {
        const report = {
            summary: this.data.growthStats,
            cycles: this.data.cycles.slice(-100), // Export last 100 cycles
            concepts: this.data.concepts,
            exportTime: new Date().toISOString()
        };
        
        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `melvin_growth_report_${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
    
    exportGraph() {
        const svg = document.getElementById('graphSvg');
        const svgData = new XMLSerializer().serializeToString(svg);
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();
        
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            
            canvas.toBlob((blob) => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `melvin_concept_graph_${new Date().toISOString().split('T')[0]}.png`;
                a.click();
                URL.revokeObjectURL(url);
            });
        };
        
        img.src = 'data:image/svg+xml;base64,' + btoa(svgData);
    }
    
    resetGraph() {
        if (this.graphSimulation) {
            this.graphSimulation.alpha(0.3).restart();
        }
    }
    
    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        overlay.classList.toggle('hidden', !show);
    }
    
    showError(message) {
        // Simple error display
        alert('Error: ' + message);
    }
    
    // D3 drag functions
    dragstarted(event, d) {
        if (!event.active) this.graphSimulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    dragended(event, d) {
        if (!event.active) this.graphSimulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

// Initialize the explorer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new MelvinExplorer();
});
