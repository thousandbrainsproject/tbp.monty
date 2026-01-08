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
        // console.log('EvidenceChart: Initialized with incremental push_event pattern');
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
        
        // Log progress (commented out to reduce console noise)
        // console.log(`EvidenceChart: +${data.new_points?.length || 0} points, total: ${this.evidenceHistory.length}`);
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
 * EvidenceChart Hook - receives data via phx-update='stream'
 * 
 * Uses the proper LiveView stream pattern:
 * - Chart has phx-update="ignore" so it's not re-rendered
 * - Data container has phx-update="stream" for efficient updates
 * - Hook reads all data on updated() and uses Plotly.react() for efficient diffing
 */
window.Hooks.EvidenceChart = {
    mounted() {
        // console.log('EvidenceChart hook mounted');
        this.chartEl = document.getElementById('evidence-chart-container');
        this.dataContainer = document.getElementById('evidence-stream');
        
        if (!this.chartEl) {
            console.warn('EvidenceChart: Chart container not found');
            return;
        }
        
        // Initialize empty chart
        this.initializeChart();
    },
    
    initializeChart() {
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
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
        };
        
        Plotly.newPlot(this.chartEl, [], layout, config);
    },
    
    updated() {
        // On each update, read all data points from stream and update chart
        const points = this.getAllDataPoints();
        
        if (points.length === 0) return;
        
        // Build traces from all points
        const traces = this.buildTraces(points);
        
        // Use Plotly.react for efficient updates (only changes what's different)
        Plotly.react(this.chartEl, traces, {
            ...this.chartEl.layout,
            xaxis: {
                ...this.chartEl.layout.xaxis,
                range: this.calculateXRange(points),
            },
        });
    },
    
    getAllDataPoints() {
        if (!this.dataContainer) return [];
        
        const points = [];
        const streamItems = this.dataContainer.querySelectorAll('[data-step]');
        
        streamItems.forEach(el => {
            try {
                const step = parseInt(el.dataset.step);
                const evidences = JSON.parse(el.dataset.evidences || '{}');
                const targetObject = el.dataset.targetObject || '';
                const episode = el.dataset.episode ? parseInt(el.dataset.episode) : null;
                
                points.push({
                    step,
                    evidences,
                    target_object: targetObject,
                    episode
                });
            } catch (e) {
                console.warn('Failed to parse stream item:', e);
            }
        });
        
        // Sort by step to ensure correct order
        return points.sort((a, b) => a.step - b.step);
    },
    
    buildTraces(points) {
        // Get all unique object names
        const objectNames = new Set();
        points.forEach(point => {
            Object.keys(point.evidences).forEach(name => objectNames.add(name));
        });
        
        const traces = [];
        let index = 0;
        
        objectNames.forEach(objName => {
            const x = [];
            const y = [];
            
            points.forEach(point => {
                if (point.evidences[objName] !== undefined) {
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
                index++;
            }
        });
        
        return traces;
    },
    
    calculateXRange(points) {
        if (points.length === 0) return null;
        
        const maxStep = Math.max(...points.map(p => p.step));
        const minStep = Math.max(0, maxStep - 100);  // Show last 100 steps
        
        return [minStep, maxStep + 5];
    },
    
    destroyed() {
        console.log('EvidenceChart hook destroyed');
        if (this.chartEl) {
            Plotly.purge(this.chartEl);
        }
    },
    
    disconnected() {
        console.log('EvidenceChart hook disconnected');
    },
    
    reconnected() {
        console.log('EvidenceChart hook reconnected');
        // On reconnect, re-read data and update
        this.updated();
    }
};

/**
 * ConnectionStatus Hook - manages connection status badge
 * 
 * Only shows "disconnected" if the experiment hasn't reached a terminal state.
 * Terminal states (completed, error, aborted) should be preserved even if
 * the WebSocket temporarily disconnects.
 */
window.Hooks.ConnectionStatus = {
    mounted() {
        // console.log('ConnectionStatus hook mounted');
        // Store original state for restoration
        this.originalClass = this.el.className;
        this.originalText = this.el.textContent;
        
        // Disable live socket debug logging
        disableDebugOnce();
    },
    
    updated() {
        // On LiveView update, refresh stored state and ensure correct display
        // This ensures terminal states from server are preserved
        const currentText = this.el.textContent.trim().toUpperCase();
        const terminalStates = ['COMPLETED', 'ERROR', 'ABORTED', 'ABORTING'];
        const isTerminalState = terminalStates.some(state => 
            currentText.includes(state) || this.el.className.includes(state.toLowerCase())
        );
        
        if (isTerminalState) {
            // Update stored state to reflect terminal status
            this.originalClass = this.el.className;
            this.originalText = this.el.textContent;
        }
    },
    
    disconnected() {
        console.log('ConnectionStatus hook disconnected');
        // When WebSocket disconnects, always show DISCONNECTED status
        // This indicates the server is no longer reachable, regardless of
        // experiment state. The experiment status (ABORTED, COMPLETED, etc.)
        // is separate from connection status.
        this.el.className = 'status-badge status-disconnected';
        this.el.textContent = 'DISCONNECTED';
        this.el.style.background = '#6b7280';
        this.el.style.color = '#ffffff';
    },
    
    reconnected() {
        // console.log('ConnectionStatus hook reconnected');
        // Disable live socket debug logging on reconnect
        disableDebugOnce();
        
        // Only restore if not in a terminal state
        const currentText = this.el.textContent.trim().toUpperCase();
        const terminalStates = ['COMPLETED', 'ERROR', 'ABORTED'];
        const isTerminalState = terminalStates.some(state => 
            currentText.includes(state) || this.el.className.includes(state.toLowerCase())
        );
        
        if (isTerminalState) {
            // Preserve terminal state even on reconnect
            console.log('Experiment in terminal state, preserving status:', currentText);
            return;
        }
        
        // Restore original state for running experiments
        this.el.className = this.originalClass;
        this.el.textContent = this.originalText;
        this.el.style.background = '';
        this.el.style.color = '';
    }
};

/**
 * MaxEvidence Hook - updates the current max evidence display
 */
window.Hooks.MaxEvidence = {
    mounted() {
        // Listen for chart_data events which include max evidence
        this.handleEvent("chart_data", (data) => {
            if (data.current_max_evidence) {
                this.updateMaxEvidence(data.current_max_evidence);
            }
        });
    },
    
    updated() {
        // Also check for max evidence in DOM updates
        // This handles initial render and full page updates
    },
    
    updateMaxEvidence(maxEvidence) {
        const contentEl = document.getElementById('max-evidence-content');
        if (!contentEl) return;
        
        if (maxEvidence && maxEvidence.object && maxEvidence.value !== undefined) {
            contentEl.innerHTML = `
                <span class="info-label">${maxEvidence.object}:</span>
                <span class="info-value">${maxEvidence.value.toFixed(3)}</span>
            `;
        } else {
            contentEl.innerHTML = `
                <span class="info-label">N/A</span>
                <span class="info-value"></span>
            `;
        }
    }
};

/**
 * Disable LiveView socket debug logging.
 * Should be called after connection is established.
 */
function disableDebugOnce() {
    try {
        if (window.liveSocket) {
            window.liveSocket.disableDebug();
        }
    } catch (e) {
        // Silently fail - not critical
    }
}

// Try to disable debug on page load (in case liveSocket is already available)
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', disableDebugOnce);
} else {
    // DOM already loaded, try immediately
    disableDebugOnce();
}

// Also try after a short delay to catch late initialization
setTimeout(disableDebugOnce, 1000);

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { EvidenceChart, getObjectColor, OBJECT_COLORS };
}
