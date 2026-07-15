// ═══════════════════════════════════════════════════════════════
// AccuLLM — Frontend Application Logic
// ═══════════════════════════════════════════════════════════════

const API = '';
let graphNetwork = null;
let pipelineRunning = false;

// ════════════════ INIT ════════════════
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    loadHistory();

    document.getElementById('searchInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !pipelineRunning) runPipeline();
    });
});

// ════════════════ STATS ════════════════
async function loadStats() {
    try {
        const res = await fetch(`${API}/api/stats`);
        const data = await res.json();
        updateStatsUI(data);
    } catch (e) {
        console.error('Failed to load stats:', e);
    }
}

function updateStatsUI(stats) {
    setText('statEntries', stats.total_entries || 0);
    setText('statQueries', stats.past_query_count || 0);
    setText('statVocab', stats.vocab_size || 0);
    setText('dbTotal', stats.total_entries || 0);
    setText('dbIndex', stats.index_size || 0);
    setText('dbVocab', stats.vocab_size || 0);
    setText('dbDim', stats.embed_dim || 128);
}

// ════════════════ HISTORY ════════════════
async function loadHistory() {
    try {
        const res = await fetch(`${API}/api/history`);
        const data = await res.json();
        renderHistory(data);
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

function renderHistory(items) {
    const el = document.getElementById('historyList');
    if (!items || items.length === 0) {
        el.innerHTML = '<div class="history-empty">No queries yet. Run the pipeline!</div>';
        return;
    }

    el.innerHTML = items.map(item => {
        const time = item.timestamp ? new Date(item.timestamp).toLocaleString() : '';
        return `
            <div class="history-item" onclick="fillQuery('${escapeHtml(item.query)}')">
                <div class="history-query">${escapeHtml(item.query)}</div>
                <div class="history-time">${time}</div>
            </div>
        `;
    }).join('');
}

function fillQuery(q) {
    document.getElementById('searchInput').value = q;
    document.getElementById('searchInput').focus();
}

// ════════════════ PIPELINE ════════════════
async function runPipeline() {
    const input = document.getElementById('searchInput');
    const question = input.value.trim();
    if (!question) {
        showToast('Please enter a question first.', 'error');
        input.focus();
        return;
    }

    if (pipelineRunning) return;
    pipelineRunning = true;

    const btn = document.getElementById('searchBtn');
    const btnText = document.getElementById('searchBtnText');
    const spinner = document.getElementById('searchSpinner');
    const loadingBar = document.getElementById('loadingBar');

    btn.disabled = true;
    btnText.textContent = 'Running...';
    spinner.style.display = 'block';
    loadingBar.className = 'loading-bar active';

    document.getElementById('welcomeCard').style.display = 'none';
    document.getElementById('resultsContainer').classList.add('visible');

    resetPipelineSteps();
    animatePipelineSteps();

    try {
        const res = await fetch(`${API}/api/pipeline`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || 'Pipeline failed');
        }

        const data = await res.json();
        renderResults(data);
        completePipelineSteps(data.steps);

        loadStats();
        loadHistory();

        showToast('Pipeline completed successfully!', 'success');
    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
        console.error(err);
    } finally {
        pipelineRunning = false;
        btn.disabled = false;
        btnText.textContent = 'Run Pipeline';
        spinner.style.display = 'none';
        loadingBar.className = 'loading-bar done';
        setTimeout(() => { loadingBar.className = 'loading-bar'; }, 500);
    }
}

function resetPipelineSteps() {
    document.querySelectorAll('.pipeline-step').forEach(step => {
        step.className = 'pipeline-step';
        const detail = step.querySelector('.step-detail');
        if (detail) detail.remove();
    });
}

function animatePipelineSteps() {
    const steps = document.querySelectorAll('.pipeline-step');
    let i = 0;
    const interval = setInterval(() => {
        if (i >= steps.length || !pipelineRunning) {
            clearInterval(interval);
            return;
        }
        if (i > 0) steps[i - 1].classList.replace('active', 'done');
        steps[i].classList.add('active');
        i++;
    }, 2000);
}

function completePipelineSteps(steps) {
    const stepEls = document.querySelectorAll('.pipeline-step');
    stepEls.forEach((el, idx) => {
        el.className = 'pipeline-step done';

        if (steps && steps[idx] && steps[idx].detail) {
            if (!el.querySelector('.step-detail')) {
                const detailEl = document.createElement('span');
                detailEl.className = 'step-detail';
                detailEl.textContent = steps[idx].detail;
                el.appendChild(detailEl);
            }
        }
    });
}

// ════════════════ RENDER RESULTS ════════════════
function renderResults(data) {
    // Teacher answer
    document.getElementById('teacherAnswer').textContent = data.teacher_answer || 'No answer';

    // Student answer
    document.getElementById('studentAnswer').textContent = data.student_answer || 'No answer';

    // Similar past queries
    if (data.similar_past && data.similar_past.length > 0) {
        document.getElementById('similarCard').style.display = 'block';
        document.getElementById('similarList').innerHTML = data.similar_past.map(s => `
            <div class="similar-item">
                <span class="similar-query">${escapeHtml(s.query || '')}</span>
                <span class="similar-score">${(s.similarity_score || 0).toFixed(3)}</span>
            </div>
        `).join('');
    } else {
        document.getElementById('similarCard').style.display = 'none';
    }

    // Model info
    if (data.model_info && data.model_info.total_params) {
        document.getElementById('modelCard').style.display = 'block';
        const info = data.model_info;
        document.getElementById('modelInfo').innerHTML = `
            <div class="model-tag"><span class="model-tag-label">Params</span><span class="model-tag-value">${info.total_params.toLocaleString()}</span></div>
            <div class="model-tag"><span class="model-tag-label">Vocab</span><span class="model-tag-value">${info.vocab_size}</span></div>
            <div class="model-tag"><span class="model-tag-label">Embed</span><span class="model-tag-value">${info.embed_dim}d</span></div>
            <div class="model-tag"><span class="model-tag-label">Heads</span><span class="model-tag-value">${info.num_heads}</span></div>
            <div class="model-tag"><span class="model-tag-label">Layers</span><span class="model-tag-value">${info.num_layers}</span></div>
            <div class="model-tag"><span class="model-tag-label">Type</span><span class="model-tag-value">GPT-style Decoder</span></div>
        `;
    }

    // Metrics
    if (data.metrics) {
        document.getElementById('metricsCard').style.display = 'block';
        const m = data.metrics;
        const ext = m.extractor || {};
        document.getElementById('metricsRow').innerHTML = `
            <div class="metric-chip"><span class="metric-label">Precision</span><span class="metric-value">${(ext.precision || 0).toFixed(3)}</span></div>
            <div class="metric-chip"><span class="metric-label">Recall</span><span class="metric-value">${(ext.recall || 0).toFixed(3)}</span></div>
            <div class="metric-chip"><span class="metric-label">F1</span><span class="metric-value">${(ext.f1 || 0).toFixed(3)}</span></div>
            <div class="metric-chip"><span class="metric-label">Student Similarity</span><span class="metric-value">${(m.student_similarity || 0).toFixed(4)}</span></div>
        `;
    }

    // Graph
    if (data.graph_data) {
        renderGraph(data.graph_data);
    }

    // VectorDB stats
    if (data.vectordb_stats) {
        updateStatsUI(data.vectordb_stats);
    }
}

// ════════════════ KNOWLEDGE GRAPH ════════════════
function renderGraph(graphData) {
    const container = document.getElementById('graphContainer');
    container.innerHTML = '';

    const nodeColors = {
        'Person': '#6c5ce7',
        'Organization': '#00d2a0',
        'Location': '#fdcb6e',
        'Event': '#ff6b6b',
        'Concept': '#74b9ff',
        'Product': '#a29bfe',
        'Technology': '#fd79a8',
        'Unknown': '#636e72',
    };

    const nodes = new vis.DataSet(
        graphData.nodes.map(n => ({
            id: n.id,
            label: n.label,
            color: {
                background: nodeColors[n.type] || nodeColors['Unknown'],
                border: 'rgba(255,255,255,0.2)',
                highlight: {
                    background: nodeColors[n.type] || nodeColors['Unknown'],
                    border: '#fff',
                }
            },
            font: { color: '#fff', size: 13, face: 'Inter', strokeWidth: 2, strokeColor: '#0a0a0f' },
            shape: 'dot',
            size: 20,
            borderWidth: 2,
            shadow: { enabled: true, color: 'rgba(0,0,0,0.3)', size: 10 },
            title: `${n.label} (${n.type})`,
        }))
    );

    const edges = new vis.DataSet(
        graphData.edges.map((e, i) => ({
            id: i,
            from: e.from,
            to: e.to,
            label: e.label,
            color: { color: 'rgba(162,155,254,0.4)', highlight: '#a29bfe' },
            font: { color: '#8888a0', size: 10, face: 'Inter', strokeWidth: 0, align: 'middle' },
            arrows: { to: { enabled: true, scaleFactor: 0.6 } },
            width: 1.5 + (e.confidence || 0.5),
            smooth: { type: 'curvedCW', roundness: 0.15 },
            title: `${e.label} (conf: ${(e.confidence || 0).toFixed(2)})`,
        }))
    );

    const options = {
        physics: {
            enabled: true,
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {
                gravitationalConstant: -40,
                centralGravity: 0.005,
                springLength: 150,
                springConstant: 0.04,
                damping: 0.4,
            },
            stabilization: { iterations: 200 },
        },
        interaction: {
            hover: true,
            tooltipDelay: 200,
            zoomView: true,
            dragView: true,
        },
        layout: { improvedLayout: true },
    };

    graphNetwork = new vis.Network(container, { nodes, edges }, options);
}

// ════════════════ CHAT ════════════════
async function sendChat() {
    const input = document.getElementById('chatInput');
    const question = input.value.trim();
    if (!question) return;

    input.value = '';
    addChatMessage(question, 'user');

    const btn = document.getElementById('chatSendBtn');
    btn.disabled = true;

    try {
        const res = await fetch(`${API}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        const data = await res.json();

        if (data.error) {
            addChatMessage(`Error: ${data.error}`, 'assistant');
        } else {
            addChatMessage(data.answer || 'No response', 'assistant');
        }

        if (data.stats) updateStatsUI(data.stats);
        loadHistory();
    } catch (err) {
        addChatMessage(`Error: ${err.message}`, 'assistant');
    } finally {
        btn.disabled = false;
        input.focus();
    }
}

function addChatMessage(text, role) {
    const container = document.getElementById('chatMessages');
    const msg = document.createElement('div');
    msg.className = `chat-msg ${role}`;
    msg.textContent = text;
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
}

// ════════════════ TABS ════════════════
function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

    event.target.classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');

    if (tabName === 'graph' && graphNetwork) {
        setTimeout(() => graphNetwork.fit(), 100);
    }
}

// ════════════════ TOAST ════════════════
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = { success: '✓', error: '✕', info: 'ℹ' };
    toast.innerHTML = `<span>${icons[type] || '•'}</span><span>${escapeHtml(message)}</span>`;

    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(40px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ════════════════ UTILS ════════════════
function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
