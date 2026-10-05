// ═══════════════════════════════════════════════════════════════
// AccuLLM — Frontend Application Logic (Complete)
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

    // Model info / architecture specs
    if (data.model_info && data.model_info.total_params) {
        document.getElementById('modelCard').style.display = 'block';
        const info = data.model_info;
        document.getElementById('modelInfo').innerHTML = `
            <div class="model-tag"><span class="model-tag-label">Params</span><span class="model-tag-value">${info.total_params.toLocaleString()}</span></div>
            <div class="model-tag"><span class="model-tag-label">Vocab</span><span class="model-tag-value">${info.vocab_size}</span></div>
            <div class="model-tag"><span class="model-tag-label">Embed</span><span class="model-tag-value">${info.embed_dim}d</span></div>
            <div class="model-tag"><span class="model-tag-label">Heads</span><span class="model-tag-value">${info.num_heads}</span></div>
            <div class="model-tag"><span class="model-tag-label">Layers</span><span class="model-tag-value">${info.num_layers}</span></div>
            <div class="model-tag"><span class="model-tag-label">FF Dim</span><span class="model-tag-value">${info.ff_dim}</span></div>
            <div class="model-tag"><span class="model-tag-label">Type</span><span class="model-tag-value">GPT-style Decoder</span></div>
        `;
    }

    // Structured Supervision Loss breakdown
    if (data.model_info && data.model_info.final_loss !== undefined) {
        document.getElementById('lossCard').style.display = 'block';
        const info = data.model_info;
        document.getElementById('lossBreakdown').innerHTML = `
            <div class="loss-item">
                <div class="loss-item-label">Total Distillation Loss</div>
                <div class="loss-item-val">${(info.final_loss || 0).toFixed(4)}</div>
            </div>
            <div class="loss-item">
                <div class="loss-item-label">Language Modeling Loss (L_LM)</div>
                <div class="loss-item-val" style="color: var(--info)">${(info.lm_loss || 0).toFixed(4)}</div>
            </div>
            <div class="loss-item">
                <div class="loss-item-label">KG Structure Alignment Loss (L_Struct)</div>
                <div class="loss-item-val" style="color: var(--success)">${(info.structure_loss || 0).toFixed(4)}</div>
            </div>
        `;

        // Loss history table
        if (info.loss_history && info.loss_history.length > 0) {
            const rows = info.loss_history.map(r => `
                <tr>
                    <td>${r.epoch}</td>
                    <td>${r.total_loss}</td>
                    <td>${r.lm_loss}</td>
                    <td>${r.structure_loss}</td>
                </tr>
            `).join('');
            document.getElementById('lossHistory').innerHTML = `
                <table class="loss-table">
                    <thead>
                        <tr>
                            <th>Epoch</th>
                            <th>Total Loss</th>
                            <th>LM Loss</th>
                            <th>Structure Loss</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            `;
        }
    }

    // Evaluation Metrics
    if (data.metrics) {
        document.getElementById('metricsCard').style.display = 'block';
        const m = data.metrics;
        const ext = m.extractor || {};
        const dist = m.distillation || {};
        const comp = m.compression || {};

        // Overall score badge
        const score = m.overall_score || 0;
        const overallBadge = document.getElementById('overallBadge');
        if (overallBadge) overallBadge.textContent = `Score: ${score.toFixed(3)}`;

        // Extractor metrics
        document.getElementById('extractorMetrics').innerHTML = `
            <div class="metric-stat-row">
                <span class="metric-stat-name">Precision</span>
                <span class="metric-stat-val accent">${(ext.precision || 0).toFixed(4)}</span>
            </div>
            <div class="metric-stat-row">
                <span class="metric-stat-name">Recall</span>
                <span class="metric-stat-val accent">${(ext.recall || 0).toFixed(4)}</span>
            </div>
            <div class="metric-stat-row">
                <span class="metric-stat-name">F1 Score</span>
                <span class="metric-stat-val highlight">${(ext.f1 || 0).toFixed(4)}</span>
            </div>
            <div class="metric-stat-row">
                <span class="metric-stat-name">Avg Confidence</span>
                <span class="metric-stat-val">${(ext.avg_confidence || 0).toFixed(4)}</span>
            </div>
            <div class="metric-stat-row">
                <span class="metric-stat-name">Relations Extracted</span>
                <span class="metric-stat-val">${ext.total_extracted || 0}</span>
            </div>
        `;

        // Distillation metrics
        document.getElementById('distillationMetrics').innerHTML = `
            <div class="metric-stat-row">
                <span class="metric-stat-name">Jaccard Similarity</span>
                <span class="metric-stat-val accent">${(dist.jaccard_similarity || 0).toFixed(4)}</span>
            </div>
            <div class="metric-stat-row">
                <span class="metric-stat-name">Token Overlap F1</span>
                <span class="metric-stat-val accent">${(dist.f1_overlap || 0).toFixed(4)}</span>
            </div>
            <div class="metric-stat-row">
                <span class="metric-stat-name">ROUGE-L Approx</span>
                <span class="metric-stat-val accent">${(dist.rouge_l_approx || 0).toFixed(4)}</span>
            </div>
            <div class="metric-stat-row">
                <span class="metric-stat-name">KG Fact Grounding</span>
                <span class="metric-stat-val highlight">${(dist.kg_fact_grounding || 0).toFixed(4)}</span>
            </div>
            <div class="metric-stat-row">
                <span class="metric-stat-name">Token Precision</span>
                <span class="metric-stat-val">${(dist.token_precision || 0).toFixed(4)}</span>
            </div>
            <div class="metric-stat-row">
                <span class="metric-stat-name">Token Recall</span>
                <span class="metric-stat-val">${(dist.token_recall || 0).toFixed(4)}</span>
            </div>
        `;

        // Compression metrics
        document.getElementById('compressionMetrics').innerHTML = `
            <div class="metric-stat-row">
                <span class="metric-stat-name">Compression Ratio</span>
                <span class="metric-stat-val highlight">${comp.compression_ratio || '—'}</span>
            </div>
            <div class="metric-stat-row">
                <span class="metric-stat-name">Parameter Reduction</span>
                <span class="metric-stat-val highlight">${comp.parameter_reduction_pct || '—'}</span>
            </div>
            <div class="metric-stat-row">
                <span class="metric-stat-name">Teacher Params</span>
                <span class="metric-stat-val">${(comp.teacher_params || 0).toLocaleString()}</span>
            </div>
            <div class="metric-stat-row">
                <span class="metric-stat-name">Student Params</span>
                <span class="metric-stat-val">${(comp.student_params || 0).toLocaleString()}</span>
            </div>
            <div class="metric-stat-row">
                <span class="metric-stat-name">Overall Score</span>
                <span class="metric-stat-val highlight">${(m.overall_score || 0).toFixed(4)}</span>
            </div>
        `;
    }

    // Knowledge Graph
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
        'Country': '#e17055',
        'Institution': '#55efc4',
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

    // Render legend
    const legendEl = document.getElementById('graphLegend');
    if (legendEl) {
        const presentTypes = [...new Set(graphData.nodes.map(n => n.type))];
        legendEl.innerHTML = presentTypes.map(type => `
            <span class="legend-item">
                <span class="legend-dot" style="background:${nodeColors[type] || nodeColors['Unknown']}"></span>
                ${type}
            </span>
        `).join('');
    }
}

// ════════════════ ATTENTION VISUALIZATION ════════════════
async function inspectAttention() {
    const input = document.getElementById('attnInput');
    const text = (input.value || '').trim();
    const wrapper = document.getElementById('attnHeatmap');

    if (!text) {
        showToast('Please enter a phrase to inspect.', 'error');
        return;
    }

    wrapper.innerHTML = '<div class="attn-empty">Loading attention weights...</div>';

    try {
        const res = await fetch(`${API}/api/attention`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });

        const data = await res.json();

        if (data.error) {
            wrapper.innerHTML = `<div class="attn-empty">${escapeHtml(data.error)}</div>`;
            return;
        }

        if (!data.tokens || data.tokens.length === 0) {
            wrapper.innerHTML = '<div class="attn-empty">No tokens found. Try a different phrase.</div>';
            return;
        }

        renderAttentionHeatmap(data.tokens, data.attention_matrix, wrapper);
    } catch (err) {
        wrapper.innerHTML = `<div class="attn-empty">Error: ${escapeHtml(err.message)}</div>`;
    }
}

function renderAttentionHeatmap(tokens, matrix, wrapper) {
    // Build header row
    const headerCells = tokens.map(t => `<th title="${escapeHtml(t)}">${escapeHtml(t.substring(0, 6))}</th>`).join('');

    // Build data rows with color-coded cells
    const rows = matrix.map((row, i) => {
        const cells = row.map((val, j) => {
            const intensity = Math.min(1, Math.max(0, val));
            const alpha = 0.1 + intensity * 0.8;
            const textColor = intensity > 0.5 ? '#fff' : '#8888a0';
            return `<td>
                <div class="attn-cell" title="${tokens[i]}→${tokens[j]}: ${val.toFixed(3)}"
                     style="background:rgba(108,92,231,${alpha.toFixed(2)});color:${textColor};padding:4px;border-radius:3px;">
                    ${val.toFixed(2)}
                </div>
            </td>`;
        }).join('');
        return `<tr><th>${escapeHtml(tokens[i].substring(0, 6))}</th>${cells}</tr>`;
    }).join('');

    wrapper.innerHTML = `
        <div>
            <p style="font-size:12px;color:var(--text-dim);margin-bottom:12px;">
                Multi-Head Self-Attention weights (averaged across heads, last layer). 
                Darker = stronger attention.
            </p>
            <table class="attn-matrix">
                <thead><tr><th></th>${headerCells}</tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;
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
function switchTab(tabName, event) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));

    if (event && event.target) {
        event.target.classList.add('active');
    }
    const tabEl = document.getElementById(`tab-${tabName}`);
    if (tabEl) tabEl.classList.add('active');

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
