/**
 * Evidence Chart - Real-time evidence score visualization using Plotly.js
 * 
 * Uses LiveView push_event for incremental data updates instead of DOM diffing.
 * This follows the pattern from https://elixirschool.com/blog/live-view-with-channels
 * where a dedicated channel pushes only new data points to avoid large diffs.
 */

// Color palette for objects (matches tbp.plot's tab10-inspired colors)
const OBJECT_COLORS = {
    'mug': '#1f77b4',
    'bowl': '#ff7f0e',
    'potted_meat_can': '#2ca02c',
    'spoon': '#d62728',
    'strawberry': '#9467bd',
    'mustard_bottle': '#8c564b',
    'dice': '#e377c2',
    'golf_ball': '#7f7f7f',
    'c_lego_duplo': '#bcbd22',
    'banana': '#17becf',
};

// Default color for unknown objects
const DEFAULT_COLOR = '#888888';

/**
 * Get color for an object name.
 * @param {string} objectName - Name of the object
 * @param {number} index - Index for fallback color generation
 * @returns {string} Hex color string
 */
function getObjectColor(objectName, index = 0) {
    if (OBJECT_COLORS[objectName]) {
        return OBJECT_COLORS[objectName];
    }
    // Generate a color from the index if not in palette
    const hue = (index * 137.508) % 360;  // Golden angle approximation
    return `hsl(${hue}, 70%, 50%)`;
}

/**
 * Evidence Chart class for managing the Plotly visualization.
 * Accumulates data client-side from incremental push_event updates.
 */
class EvidenceChart {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        
        if (!this.container) {
            console.warn(`EvidenceChart: Container '${containerId}' not found`);
            return;
        }
        
        this.options = {
            liveFollow: options.liveFollow !== false,
            xRangeWindow: options.xRangeWindow || 100,
        };
        
        // Client-side data accumulation (no more server-side full history)
        this.evidenceHistory = [];
        this.episodeMarkers = [];
        this.objectNames = new Set();
        
        this.initialized = false;
        this._initChart();
    }
    
    /**
     * Initialize the Plotly chart with empty data.
     */
    _initChart() {
        const layout = {
            title: {
                text: 'Evidence Over Time',
                font: { size: 14 }
            },
            xaxis: {
                title: 'Step',
                showgrid: true,
                gridcolor: '#e5e7eb',
            },
            yaxis: {
                title: 'Evidence Score',
                showgrid: true,
                gridcolor: '#e5e7eb',
            },
            showlegend: true,
            legend: {
                orientation: 'h',
                y: -0.2,
            },
            margin: { t: 40, r: 20, b: 60, l: 60 },
            paper_bgcolor: 'white',
            plot_bgcolor: '#fafafa',
            shapes: [],
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        };
        
        Plotly.newPlot(this.container, [], layout, config);
        this.initialized = true;
        console.log('EvidenceChart: Initialized with incremental push_event pattern');
    }
    
    /**
     * Handle incremental data from push_event.
     * Appends new data to local history and updates the chart.
     * @param {Object} data - Incremental data from server
     * @param {Array} data.new_points - New evidence points to append
     * @param {Array} data.new_markers - New episode markers to append
     * @param {Array} data.new_object_names - Current set of object names
     * @param {number} data.total_points - Total points on server (for verification)
     */
    appendData(data) {
        if (!data || !this.initialized) return;
        
        // Append new points
        if (data.new_points && data.new_points.length > 0) {
            this.evidenceHistory.push(...data.new_points);
            
            // Track object names
            data.new_points.forEach(point => {
                if (point.evidences) {
                    Object.keys(point.evidences).forEach(name => {
                        this.objectNames.add(name);
                    });
                }
            });
        }
        
        // Append new markers
        if (data.new_markers && data.new_markers.length > 0) {
            this.episodeMarkers.push(...data.new_markers);
        }
        
        // Update object names from server (authoritative)
        if (data.new_object_names) {
            data.new_object_names.forEach(name => this.objectNames.add(name));
        }
        
        // Update chart
        this._render();
        
        // Log progress
        console.log(`EvidenceChart: +${data.new_points?.length || 0} points, total: ${this.evidenceHistory.length}`);
    }
    
    /**
     * Reset chart data (e.g., on new experiment).
     */
    reset() {
        this.evidenceHistory = [];
        this.episodeMarkers = [];
        this.objectNames = new Set();
        this._render();
    }
    
    /**
     * Render the chart with current accumulated data.
     */
    _render() {
        if (!this.initialized || this.evidenceHistory.length === 0) return;
        
        const objectNamesArray = Array.from(this.objectNames).sort();
        const traces = this._buildTraces(this.evidenceHistory, objectNamesArray);
        const shapes = this._buildEpisodeShapes(this.episodeMarkers, this.evidenceHistory);
        const xRange = this._calculateXRange(this.evidenceHistory);
        
        Plotly.react(this.container, traces, {
            ...this.container.layout,
            shapes: shapes,
            xaxis: {
                ...this.container.layout.xaxis,
                range: xRange,
            },
        });
    }
    
    /**
     * Build Plotly traces from evidence history.
     */
    _buildTraces(history, objectNames) {
        const traces = [];
        
        objectNames.forEach((objName, index) => {
            const x = [];
            const y = [];
            
            history.forEach(point => {
                if (point.evidences && point.evidences[objName] !== undefined) {
                    x.push(point.step);
                    y.push(point.evidences[objName]);
                }
            });
            
            if (x.length > 0) {
                traces.push({
                    x: x,
                    y: y,
                    name: objName,
                    type: 'scatter',
                    mode: 'lines',
                    line: {
                        color: getObjectColor(objName, index),
                        width: 2,
                    },
                });
            }
        });
        
        return traces;
    }
    
    /**
     * Build background shapes for episode transitions.
     */
    _buildEpisodeShapes(markers, history) {
        if (markers.length === 0) return [];
        
        const shapes = [];
        const maxStep = history.length > 0 
            ? Math.max(...history.map(p => p.step)) 
            : 100;
        
        for (let i = 0; i < markers.length; i++) {
            const marker = markers[i];
            const nextMarker = markers[i + 1];
            
            const x0 = marker.start_step;
            const x1 = nextMarker ? nextMarker.start_step : maxStep;
            const color = getObjectColor(marker.target_object, i);
            
            shapes.push({
                type: 'rect',
                xref: 'x',
                yref: 'paper',
                x0: x0,
                x1: x1,
                y0: 0,
                y1: 1,
                fillcolor: color,
                opacity: 0.1,
                line: { width: 0 },
                layer: 'below',
            });
        }
        
        return shapes;
    }
    
    /**
     * Calculate x-axis range for auto-scrolling.
     */
    _calculateXRange(history) {
        if (!this.options.liveFollow || history.length === 0) {
            return null;
        }
        
        const maxStep = Math.max(...history.map(p => p.step));
        const minStep = Math.max(0, maxStep - this.options.xRangeWindow);
        
        return [minStep, maxStep + 5];
    }
    
    /**
     * Toggle live follow mode.
     */
    setLiveFollow(enabled) {
        this.options.liveFollow = enabled;
        this._render();
    }
    
    /**
     * Destroy the chart and clean up resources.
     */
    destroy() {
        if (this.container) {
            Plotly.purge(this.container);
        }
    }
}

// LiveView Hooks - pyview uses window.Hooks (like Phoenix LiveView)
// See: https://github.com/ogrodnek/pyview/blob/main/pyview/static/assets/app.js
window.Hooks = window.Hooks || {};

/**
 * EvidenceChart Hook - receives data via JSON tag (DOM patching)
 * 
 * Hybrid approach: reads from JSON tag on updates since pyview's
 * push_event doesn't work with handle_info context.
 * Keeps handleEvent listener for future push_event support.
 */
window.Hooks.EvidenceChart = {
    mounted() {
        console.log('EvidenceChart hook mounted');
        
        // Initialize the chart
        this.chart = new EvidenceChart('evidence-chart-container', {
            liveFollow: true,
            xRangeWindow: 100
        });
        
        // Load initial data from JSON tag
        this._loadFromJsonTag();
        
        // Also listen for push_event (for future pyview support)
        this.handleEvent("chart_data", (data) => {
            console.log('EvidenceChart: Received push_event with', 
                data.new_points?.length || 0, 'new points');
            if (this.chart) {
                this.chart.appendData(data);
            }
        });
        
        // Set up MutationObserver to watch for JSON tag changes
        this._setupObserver();
    },
    
    _loadFromJsonTag() {
        const dataEl = document.getElementById('chart-data');
        if (!dataEl || !this.chart) return;
        
        try {
            const data = JSON.parse(dataEl.textContent || '{}');
            if (data.evidence_history && data.evidence_history.length > 0) {
                // Reset and load full data (JSON tag contains all history)
                this.chart.reset();
                this.chart.appendData({
                    new_points: data.evidence_history,
                    new_markers: data.episode_markers || [],
                    new_object_names: data.object_names || [],
                    total_points: data.evidence_history.length
                });
            }
        } catch (e) {
            console.warn('EvidenceChart: Failed to parse JSON tag:', e);
        }
    },
    
    _setupObserver() {
        const dataEl = document.getElementById('chart-data');
        if (!dataEl) return;
        
        this.observer = new MutationObserver(() => {
            this._loadFromJsonTag();
        });
        
        this.observer.observe(dataEl, {
            childList: true,
            characterData: true,
            subtree: true
        });
    },
    
    updated() {
        // DOM was updated by LiveView - reload data from JSON tag
        this._loadFromJsonTag();
    },
    
    destroyed() {
        console.log('EvidenceChart hook destroyed');
        if (this.observer) {
            this.observer.disconnect();
        }
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }
    },
    
    disconnected() {
        console.log('EvidenceChart hook disconnected');
    },
    
    reconnected() {
        console.log('EvidenceChart hook reconnected');
        // On reconnect, reload from JSON tag
        this._loadFromJsonTag();
    }
};

/**
 * ConnectionStatus Hook - manages connection status badge
 */
window.Hooks.ConnectionStatus = {
    mounted() {
        console.log('ConnectionStatus hook mounted');
        // Store original state for restoration
        this.originalClass = this.el.className;
        this.originalText = this.el.textContent;
    },
    
    disconnected() {
        console.log('ConnectionStatus hook disconnected');
        this.el.className = 'status-badge status-disconnected';
        this.el.textContent = 'DISCONNECTED';
        this.el.style.background = '#6b7280';
        this.el.style.color = '#ffffff';
    },
    
    reconnected() {
        console.log('ConnectionStatus hook reconnected');
        this.el.className = this.originalClass;
        this.el.textContent = this.originalText;
        this.el.style.background = '';
        this.el.style.color = '';
    }
};

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { EvidenceChart, getObjectColor, OBJECT_COLORS };
}
