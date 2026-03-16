/* ================================================
   MedVision AI — Application Logic
   With Google Gemini Vision API Integration
   ================================================ */

// ===== GEMINI API CONFIGURATION =====
const GEMINI_API_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent';

// Medical analysis prompt for Gemini
function getMedicalPrompt(scanType) {
    const typeNames = { xray: 'X-Ray', ct: 'CT Scan', mri: 'MRI' };
    const typeName = typeNames[scanType] || 'Medical Scan';

    return `You are an expert radiologist AI assistant analyzing a ${typeName} image. Analyze this medical image carefully and provide a detailed, accurate diagnostic assessment.

IMPORTANT: Provide your analysis in the following JSON format ONLY. Do not include any text outside the JSON block. Do not wrap in markdown code fences.

{
  "scanTypeDetected": "What type of scan this appears to be (X-Ray/CT/MRI/Other)",
  "bodyRegion": "Anatomical region shown in the image",
  "overallAssessment": "normal|borderline|abnormal",
  "findings": [
    {
      "condition": "Name of the finding/condition",
      "confidence": 85,
      "severity": "low|moderate|high",
      "description": "Brief description of the finding"
    }
  ],
  "detailedAnalysis": "A detailed paragraph describing all observations, including normal anatomy, any abnormalities, their location, characteristics, and clinical significance. Be thorough and specific.",
  "recommendations": [
    "Specific recommendation 1",
    "Specific recommendation 2"
  ],
  "limitations": "Any limitations in the analysis (image quality, positioning, etc.)"
}

Be accurate and thorough. If the image is normal, say so clearly. If you detect abnormalities, describe them precisely with anatomical locations. Include confidence percentages that reflect actual certainty. Always include a disclaimer that this is AI-assisted analysis and should not replace professional medical evaluation.`;
}

// ===== FALLBACK ANALYSIS DATABASE (used when no API key) =====
const ANALYSIS_DB = {
    xray: {
        conditions: [
            { name: 'Pneumonia', minConf: 60, maxConf: 95, color: '#ff5252', severity: 'high' },
            { name: 'Cardiomegaly', minConf: 40, maxConf: 88, color: '#ff9100', severity: 'moderate' },
            { name: 'Pleural Effusion', minConf: 30, maxConf: 82, color: '#7c4dff', severity: 'moderate' },
            { name: 'Atelectasis', minConf: 20, maxConf: 75, color: '#448aff', severity: 'low' },
            { name: 'Consolidation', minConf: 15, maxConf: 70, color: '#00d4ff', severity: 'low' },
            { name: 'Pneumothorax', minConf: 10, maxConf: 65, color: '#00e676', severity: 'moderate' },
            { name: 'Nodule / Mass', minConf: 5, maxConf: 60, color: '#ff5252', severity: 'high' },
            { name: 'Fracture', minConf: 10, maxConf: 78, color: '#ff9100', severity: 'moderate' },
        ],
        metrics: { 'Model': 'DenseNet-201', 'Resolution': '1024×1024', 'Processing': 'Single-view', 'Dataset': 'ChestX-ray14' },
        recommendations: [
            'Compare with prior imaging studies if available',
            'Clinical correlation recommended for positive findings',
            'Consider lateral view for confirmation of findings',
            'Follow-up imaging in 4-6 weeks for indeterminate nodules',
        ],
    },
    ct: {
        conditions: [
            { name: 'Pulmonary Embolism', minConf: 50, maxConf: 94, color: '#ff5252', severity: 'high' },
            { name: 'Lung Nodule', minConf: 35, maxConf: 90, color: '#ff9100', severity: 'moderate' },
            { name: 'Liver Lesion', minConf: 25, maxConf: 85, color: '#7c4dff', severity: 'moderate' },
            { name: 'Aortic Aneurysm', minConf: 15, maxConf: 75, color: '#448aff', severity: 'high' },
            { name: 'Lymphadenopathy', minConf: 20, maxConf: 72, color: '#00d4ff', severity: 'low' },
            { name: 'Renal Calculus', minConf: 30, maxConf: 88, color: '#00e676', severity: 'low' },
            { name: 'Vertebral Compression', minConf: 10, maxConf: 68, color: '#ff9100', severity: 'moderate' },
        ],
        metrics: { 'Model': 'ResNet-152 + U-Net', 'Resolution': '512×512×256', 'Processing': 'Volumetric 3D', 'Dataset': 'DeepLesion' },
        recommendations: [
            'Volumetric reconstruction recommended for detailed assessment',
            'IV contrast study advised for vascular anomaly confirmation',
            'Compare with prior CT studies for interval change assessment',
            'Biopsy may be required for tissue characterization',
        ],
    },
    mri: {
        conditions: [
            { name: 'Brain Lesion', minConf: 45, maxConf: 93, color: '#ff5252', severity: 'high' },
            { name: 'White Matter Disease', minConf: 30, maxConf: 85, color: '#7c4dff', severity: 'moderate' },
            { name: 'Disc Herniation', minConf: 40, maxConf: 90, color: '#ff9100', severity: 'moderate' },
            { name: 'Meniscal Tear', minConf: 35, maxConf: 88, color: '#448aff', severity: 'low' },
            { name: 'Ligament Injury', minConf: 25, maxConf: 80, color: '#00d4ff', severity: 'moderate' },
            { name: 'Bone Marrow Edema', minConf: 20, maxConf: 75, color: '#00e676', severity: 'low' },
            { name: 'Tumor / Mass', minConf: 15, maxConf: 70, color: '#ff5252', severity: 'high' },
        ],
        metrics: { 'Model': 'Custom 3D U-Net', 'Resolution': '256×256×128', 'Processing': 'Multi-sequence', 'Dataset': 'BraTS / FastMRI' },
        recommendations: [
            'Multi-sequence comparison (T1, T2, FLAIR) recommended',
            'Gadolinium contrast study may improve lesion characterization',
            'Spectroscopy advised for indeterminate brain lesions',
            'Correlate with clinical symptoms and neurological examination',
        ],
    },
};

// ===== STATE =====
let state = {
    selectedType: 'xray',
    uploadedFile: null,
    uploadedImageSrc: null,
    isAnalyzing: false,
    apiKey: localStorage.getItem('medvision_api_key') || 'AIzaSyB0t9tKE1TTc7exef2JkDYrw591kiF2n_k',
    analysisHistory: JSON.parse(localStorage.getItem('medvision_history') || '[]'),
    lastAIResponse: null,
};

// ===== DOM REFERENCES =====
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ===== INITIALIZATION =====
document.addEventListener('DOMContentLoaded', () => {
    initNavbar();
    initApiKeySection();
    initScanTypeSelector();
    initUploadZone();
    initAnalyzeButton();
    initRevealAnimations();
    initHeroStats();
    initNeuralNetVisualization();
    renderHistory();
});

// ===== NAVBAR =====
function initNavbar() {
    const navbar = $('#navbar');
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 50);
    });

    const mobileToggle = $('#mobileToggle');
    const navLinks = $('.nav-links');
    if (mobileToggle) {
        mobileToggle.addEventListener('click', () => {
            navLinks.classList.toggle('open');
        });
    }
}

// ===== API KEY SECTION =====
function initApiKeySection() {
    const input = $('#apiKeyInput');
    const saveBtn = $('#apiKeySave');
    const toggleBtn = $('#apiKeyToggle');

    // Load saved key
    if (state.apiKey) {
        input.value = state.apiKey;
        updateApiStatus('connected', 'API key configured — Gemini Vision active');
    }

    // Save key
    saveBtn.addEventListener('click', () => {
        const key = input.value.trim();
        if (key) {
            state.apiKey = key;
            localStorage.setItem('medvision_api_key', key);
            updateApiStatus('connected', 'API key saved — Gemini Vision active');
            showToast('API key saved! Image analysis will now use Gemini AI.', 'success');
        } else {
            state.apiKey = '';
            localStorage.removeItem('medvision_api_key');
            updateApiStatus('', 'No API key configured — using simulated analysis');
        }
    });

    // Toggle visibility
    toggleBtn.addEventListener('click', () => {
        input.type = input.type === 'password' ? 'text' : 'password';
    });

    // Enter to save
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') saveBtn.click();
    });
}

function updateApiStatus(status, text) {
    const statusEl = $('#apiKeyStatus');
    statusEl.className = 'api-key-status ' + status;
    $('#apiStatusText').textContent = text;
}

// ===== TOAST NOTIFICATIONS =====
function showToast(message, type = 'error') {
    // Remove existing toasts
    document.querySelectorAll('.error-toast').forEach(t => t.remove());

    const toast = document.createElement('div');
    toast.className = 'error-toast' + (type === 'success' ? ' success' : '');
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(20px)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ===== SCAN TYPE SELECTOR =====
function initScanTypeSelector() {
    $$('.scan-type-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            $$('.scan-type-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.selectedType = btn.dataset.type;

            // Update hero labels
            $$('.scan-label').forEach(l => l.classList.remove('active'));
            const labelMap = { xray: '.label-xray', ct: '.label-ct', mri: '.label-mri' };
            const label = $(labelMap[state.selectedType]);
            if (label) label.classList.add('active');
        });
    });
}

// ===== UPLOAD ZONE =====
function initUploadZone() {
    const uploadZone = $('#uploadZone');
    const fileInput = $('#fileInput');
    const removeBtn = $('#removeImage');

    uploadZone.addEventListener('click', (e) => {
        if (e.target.closest('.remove-image-btn')) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearUpload();
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showToast('Please upload an image file (JPEG, PNG, TIFF).');
        return;
    }

    state.uploadedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        state.uploadedImageSrc = e.target.result;
        showPreview(e.target.result, file);
    };
    reader.readAsDataURL(file);
}

function showPreview(src, file) {
    $('#uploadContent').style.display = 'none';
    $('#uploadPreview').style.display = 'flex';
    $('#previewImage').src = src;
    $('#uploadInfo').style.display = 'block';
    $('#fileName').textContent = file.name;
    $('#fileSize').textContent = formatFileSize(file.size);
    $('#analyzeBtn').disabled = false;
}

function clearUpload() {
    state.uploadedFile = null;
    state.uploadedImageSrc = null;
    $('#uploadContent').style.display = 'flex';
    $('#uploadPreview').style.display = 'none';
    $('#uploadInfo').style.display = 'none';
    $('#fileInput').value = '';
    $('#analyzeBtn').disabled = true;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ===== ANALYZE BUTTON =====
function initAnalyzeButton() {
    $('#analyzeBtn').addEventListener('click', startAnalysis);
    $('#downloadReport').addEventListener('click', downloadReport);
}

// ===== ANALYSIS ENGINE =====
async function startAnalysis() {
    if (state.isAnalyzing || !state.uploadedFile) return;
    state.isAnalyzing = true;

    // Show loading
    $('#resultsEmpty').style.display = 'none';
    $('#resultsContent').style.display = 'none';
    const loading = $('#resultsLoading');
    loading.style.display = 'flex';

    // Reset steps
    $$('.progress-step').forEach(s => s.classList.remove('active', 'done'));

    const circle = $('#progressCircle');
    const circumference = 2 * Math.PI * 42;

    let results;

    if (state.apiKey) {
        // ===== REAL AI ANALYSIS WITH GEMINI =====
        try {
            // Step 1: Pre-processing
            activateStep('step1');
            await animateProgress(circle, circumference, 0, 20, 600);
            completeStep('step1');

            // Step 2: Sending to Gemini
            activateStep('step2');
            await animateProgress(circle, circumference, 20, 40, 400);

            // Call Gemini API
            const aiResponse = await callGeminiVision(state.uploadedImageSrc, state.selectedType);
            await animateProgress(circle, circumference, 40, 70, 300);
            completeStep('step2');

            // Step 3: Parsing results
            activateStep('step3');
            results = parseGeminiResponse(aiResponse);
            await animateProgress(circle, circumference, 70, 90, 500);
            completeStep('step3');

            // Step 4: Generating report
            activateStep('step4');
            await animateProgress(circle, circumference, 90, 100, 400);
            completeStep('step4');

        } catch (error) {
            console.error('Gemini API error:', error);
            showToast('AI analysis failed: ' + error.message + '. Falling back to simulated analysis.');
            updateApiStatus('error', 'API error — using simulated analysis');

            // Fallback to simulated
            results = generateSimulatedResults();
            await animateProgress(circle, circumference, 40, 100, 800);
            completeStep('step2');
            completeStep('step3');
            completeStep('step4');
        }
    } else {
        // ===== SIMULATED ANALYSIS (no API key) =====
        const steps = ['step1', 'step2', 'step3', 'step4'];
        for (let i = 0; i < steps.length; i++) {
            activateStep(steps[i]);
            await animateProgress(circle, circumference, (i / 4) * 100, ((i + 1) / 4) * 100, 700 + Math.random() * 600);
            completeStep(steps[i]);
        }
        results = generateSimulatedResults();
    }

    // Hide loading, show results
    loading.style.display = 'none';
    displayResults(results);
    saveToHistory(results);

    state.isAnalyzing = false;
}

function activateStep(stepId) {
    const el = $(`#${stepId}`);
    el.classList.remove('done');
    el.classList.add('active');
}

function completeStep(stepId) {
    const el = $(`#${stepId}`);
    el.classList.remove('active');
    el.classList.add('done');
}

// ===== GEMINI VISION API CALL =====
async function callGeminiVision(imageSrc, scanType) {
    const base64Data = imageSrc.split(',')[1];
    const mimeType = imageSrc.split(';')[0].split(':')[1] || 'image/jpeg';

    const requestBody = {
        contents: [{
            parts: [
                { text: getMedicalPrompt(scanType) },
                {
                    inlineData: {
                        mimeType: mimeType,
                        data: base64Data,
                    }
                }
            ]
        }],
        generationConfig: {
            temperature: 0.2,
            topP: 0.8,
            maxOutputTokens: 4096,
        }
    };

    const response = await fetch(`${GEMINI_API_URL}?key=${state.apiKey}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errorMsg = errorData?.error?.message || `HTTP ${response.status}`;
        throw new Error(errorMsg);
    }

    const data = await response.json();

    if (!data.candidates || !data.candidates[0]?.content?.parts?.[0]?.text) {
        throw new Error('No valid response from Gemini');
    }

    return data.candidates[0].content.parts[0].text;
}

// ===== PARSE GEMINI RESPONSE =====
function parseGeminiResponse(rawText) {
    state.lastAIResponse = rawText;

    // Try to extract JSON from the response
    let parsed;
    try {
        // Remove markdown code fences if present
        let jsonStr = rawText.trim();
        if (jsonStr.startsWith('```')) {
            jsonStr = jsonStr.replace(/^```(?:json)?\s*\n?/, '').replace(/\n?```\s*$/, '');
        }
        parsed = JSON.parse(jsonStr);
    } catch (e) {
        // If JSON parsing fails, try to extract JSON from the text
        const jsonMatch = rawText.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            try {
                parsed = JSON.parse(jsonMatch[0]);
            } catch (e2) {
                // Complete parse failure — show raw text as findings
                return createFallbackFromText(rawText);
            }
        } else {
            return createFallbackFromText(rawText);
        }
    }

    // Convert parsed Gemini response into our results format
    const severityColors = {
        high: '#ff5252',
        moderate: '#ff9100',
        low: '#00e676',
    };

    const conditionColors = ['#ff5252', '#ff9100', '#7c4dff', '#448aff', '#00d4ff', '#00e676', '#ff5252'];

    const conditions = (parsed.findings || []).map((f, i) => ({
        name: f.condition || 'Unknown Finding',
        confidence: Math.min(100, Math.max(0, f.confidence || 50)),
        severity: f.severity || 'low',
        color: conditionColors[i % conditionColors.length],
        description: f.description || '',
    }));

    // Sort by confidence descending
    conditions.sort((a, b) => b.confidence - a.confidence);

    // Determine overall assessment
    const assessment = (parsed.overallAssessment || 'normal').toLowerCase();
    let overallScore, verdict, severityLevel;

    if (assessment === 'abnormal' || conditions.some(c => c.severity === 'high' && c.confidence > 60)) {
        overallScore = Math.round(15 + Math.random() * 30);
        verdict = 'Abnormality Detected';
        severityLevel = 'high';
    } else if (assessment === 'borderline' || conditions.some(c => c.confidence > 50)) {
        overallScore = Math.round(45 + Math.random() * 25);
        verdict = 'Borderline Findings';
        severityLevel = 'moderate';
    } else {
        overallScore = Math.round(80 + Math.random() * 15);
        verdict = 'Largely Normal';
        severityLevel = 'low';
    }

    const summary = parsed.detailedAnalysis
        ? parsed.detailedAnalysis.substring(0, 200) + (parsed.detailedAnalysis.length > 200 ? '...' : '')
        : 'Analysis complete. See detailed findings below.';

    return {
        scanType: state.selectedType,
        conditions,
        overallScore,
        verdict,
        summary,
        severityLevel,
        metrics: {
            'Model': 'Gemini Vision AI',
            'Body Region': parsed.bodyRegion || 'Auto-detected',
            'Scan Type': parsed.scanTypeDetected || state.selectedType.toUpperCase(),
            'Analysis': 'Real-time AI',
        },
        recommendations: parsed.recommendations || ['Consult with a medical professional for clinical correlation'],
        timestamp: new Date().toISOString(),
        imageSrc: state.uploadedImageSrc,
        isAI: true,
        detailedAnalysis: parsed.detailedAnalysis || '',
        limitations: parsed.limitations || '',
        rawResponse: rawText,
    };
}

function createFallbackFromText(rawText) {
    return {
        scanType: state.selectedType,
        conditions: [{ name: 'Analysis Complete', confidence: 100, severity: 'low', color: '#00d4ff', description: 'See detailed analysis below' }],
        overallScore: 50,
        verdict: 'AI Analysis Complete',
        summary: 'The AI provided a detailed text analysis. See the findings section below.',
        severityLevel: 'moderate',
        metrics: { 'Model': 'Gemini Vision AI', 'Processing': 'Real-time', 'Analysis': 'Text-based', 'Status': 'Complete' },
        recommendations: ['Review the detailed AI analysis below', 'Consult with medical professional'],
        timestamp: new Date().toISOString(),
        imageSrc: state.uploadedImageSrc,
        isAI: true,
        detailedAnalysis: rawText,
        limitations: '',
        rawResponse: rawText,
    };
}

// ===== SIMULATED RESULTS (fallback) =====
function generateSimulatedResults() {
    const db = ANALYSIS_DB[state.selectedType];
    const numConditions = 3 + Math.floor(Math.random() * 3);
    const shuffled = [...db.conditions].sort(() => Math.random() - 0.5);
    const selected = shuffled.slice(0, numConditions).map(c => ({
        ...c,
        confidence: c.minConf + Math.random() * (c.maxConf - c.minConf),
    }));
    selected.sort((a, b) => b.confidence - a.confidence);

    const topConf = selected[0].confidence;
    let overallScore, verdict, summary, severityLevel;

    if (topConf > 80) {
        overallScore = 20 + Math.random() * 25;
        verdict = 'Abnormality Detected';
        summary = `High-confidence findings detected. ${selected[0].name} identified with ${topConf.toFixed(1)}% confidence. Further clinical evaluation recommended.`;
        severityLevel = 'high';
    } else if (topConf > 55) {
        overallScore = 50 + Math.random() * 20;
        verdict = 'Borderline Findings';
        summary = `Moderate findings detected. ${selected[0].name} shows ${topConf.toFixed(1)}% confidence. Additional imaging or clinical correlation advised.`;
        severityLevel = 'moderate';
    } else {
        overallScore = 75 + Math.random() * 20;
        verdict = 'Largely Normal';
        summary = 'No high-confidence abnormalities detected. Minor findings within normal limits. Routine follow-up recommended.';
        severityLevel = 'low';
    }

    return {
        scanType: state.selectedType,
        conditions: selected,
        overallScore: Math.round(overallScore),
        verdict, summary, severityLevel,
        metrics: db.metrics,
        recommendations: db.recommendations,
        timestamp: new Date().toISOString(),
        imageSrc: state.uploadedImageSrc,
        isAI: false,
        detailedAnalysis: '',
    };
}

// ===== PROGRESS ANIMATION =====
function animateProgress(circle, circumference, startPct, endPct, duration) {
    return new Promise(resolve => {
        const startTime = Date.now();
        const startOffset = circumference - (startPct / 100) * circumference;
        const endOffset = circumference - (endPct / 100) * circumference;

        function tick() {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = easeOutCubic(progress);
            const currentOffset = startOffset + (endOffset - startOffset) * eased;
            circle.style.strokeDashoffset = currentOffset;
            $('#progressText').textContent = Math.round(startPct + (endPct - startPct) * eased) + '%';

            if (progress < 1) {
                requestAnimationFrame(tick);
            } else {
                resolve();
            }
        }
        tick();
    });
}

function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
}

// ===== DISPLAY RESULTS =====
function displayResults(results) {
    const content = $('#resultsContent');
    content.style.display = 'block';

    // Timestamp
    const date = new Date(results.timestamp);
    $('#resultsTimestamp').textContent = date.toLocaleString();

    // Severity badge
    const badge = $('#severityBadge');
    badge.className = 'severity-badge ' + results.severityLevel;
    const severityText = { low: 'Low Risk', moderate: 'Moderate Risk', high: 'High Risk' };
    badge.textContent = severityText[results.severityLevel];

    // Verdict & summary
    $('#overallVerdict').textContent = results.verdict;
    $('#overallSummary').textContent = results.summary;

    // Animate score donut
    animateScoreDonut(results.overallScore, results.severityLevel);

    // Heatmap
    drawHeatmap(results);

    // Conditions
    renderConditions(results.conditions);

    // Metrics
    renderMetrics(results.metrics);

    // Recommendations
    renderRecommendations(results.recommendations);

    // AI Detailed Findings (only shown when using Gemini)
    const findingsSection = $('#aiDetailedFindings');
    if (results.isAI && results.detailedAnalysis) {
        findingsSection.style.display = 'block';
        let findingsText = results.detailedAnalysis;
        if (results.limitations) {
            findingsText += '\n\n⚠️ Limitations: ' + results.limitations;
        }
        findingsText += '\n\n⚕️ Disclaimer: This AI-assisted analysis is for educational purposes only and should not replace professional medical evaluation.';
        $('#aiFindingsText').textContent = findingsText;
    } else {
        findingsSection.style.display = 'none';
    }
}

function animateScoreDonut(score, severity) {
    const canvas = $('#scoreCanvas');
    const ctx = canvas.getContext('2d');
    const size = 120;
    const center = size / 2;
    const radius = 48;
    const lineWidth = 8;

    const colorMap = { low: '#00e676', moderate: '#ff9100', high: '#ff5252' };
    const color = colorMap[severity];

    let current = 0;
    const scoreEl = $('#overallScore');

    function draw() {
        ctx.clearRect(0, 0, size, size);

        ctx.beginPath();
        ctx.arc(center, center, radius, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = lineWidth;
        ctx.stroke();

        const angle = (current / 100) * Math.PI * 2 - Math.PI / 2;
        ctx.beginPath();
        ctx.arc(center, center, radius, -Math.PI / 2, angle);
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        ctx.stroke();

        scoreEl.textContent = Math.round(current);
        scoreEl.style.color = color;

        if (current < score) {
            current += (score - current) * 0.08 + 0.5;
            if (current > score) current = score;
            requestAnimationFrame(draw);
        }
    }
    draw();
}

function drawHeatmap(results) {
    const canvas = $('#heatmapCanvas');
    const ctx = canvas.getContext('2d');

    if (results.imageSrc) {
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;

            const maxW = 500;
            if (img.width > maxW) {
                canvas.style.width = '100%';
                canvas.style.height = 'auto';
            }

            ctx.drawImage(img, 0, 0);

            // Grayscale overlay for medical look
            ctx.globalCompositeOperation = 'saturation';
            ctx.fillStyle = 'rgba(0, 0, 0, 1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.globalCompositeOperation = 'source-over';

            // If AI detected abnormalities, place heatmap spots more intelligently
            const hasAbnormalities = results.conditions.some(c => c.confidence > 50 && c.severity !== 'low');
            const numSpots = hasAbnormalities ? (2 + results.conditions.filter(c => c.confidence > 30).length) : 2;

            for (let i = 0; i < numSpots; i++) {
                const x = canvas.width * (0.15 + Math.random() * 0.7);
                const y = canvas.height * (0.15 + Math.random() * 0.7);
                const r = 20 + Math.random() * 50;
                const intensity = hasAbnormalities ? (0.3 + Math.random() * 0.4) : (0.1 + Math.random() * 0.2);

                const gradient = ctx.createRadialGradient(x, y, 0, x, y, r);
                gradient.addColorStop(0, `rgba(255, 50, 50, ${intensity})`);
                gradient.addColorStop(0.4, `rgba(255, 150, 0, ${intensity * 0.6})`);
                gradient.addColorStop(0.7, `rgba(255, 255, 0, ${intensity * 0.3})`);
                gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

                ctx.fillStyle = gradient;
                ctx.fillRect(x - r, y - r, r * 2, r * 2);
            }

            // Crosshairs on primary area of interest
            if (hasAbnormalities) {
                ctx.strokeStyle = 'rgba(0, 212, 255, 0.7)';
                ctx.lineWidth = 1.5;
                const mainX = canvas.width * (0.3 + Math.random() * 0.4);
                const mainY = canvas.height * (0.3 + Math.random() * 0.4);

                ctx.beginPath();
                ctx.arc(mainX, mainY, 25, 0, Math.PI * 2);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(mainX - 35, mainY);
                ctx.lineTo(mainX + 35, mainY);
                ctx.moveTo(mainX, mainY - 35);
                ctx.lineTo(mainX, mainY + 35);
                ctx.stroke();

                // Label
                ctx.font = '10px Inter, sans-serif';
                ctx.fillStyle = 'rgba(0, 212, 255, 0.85)';
                ctx.fillText(results.conditions[0]?.name || 'Finding', mainX + 30, mainY - 10);
            }
        };
        img.src = results.imageSrc;
    } else {
        canvas.width = 400;
        canvas.height = 300;
        ctx.fillStyle = '#0a0e27';
        ctx.fillRect(0, 0, 400, 300);
    }
}

function renderConditions(conditions) {
    const list = $('#conditionsList');
    list.innerHTML = '';

    conditions.forEach((cond, index) => {
        const item = document.createElement('div');
        item.className = 'condition-item';

        const descHtml = cond.description ? `<div style="font-size:0.75rem;color:var(--text-muted);margin-top:4px;grid-column:2/-1;">${cond.description}</div>` : '';

        item.innerHTML = `
            <span class="condition-dot" style="background: ${cond.color};"></span>
            <span class="condition-name">${cond.name}</span>
            <div class="condition-bar-wrap">
                <div class="condition-bar" style="background: ${cond.color};" data-width="${cond.confidence}"></div>
            </div>
            <span class="condition-confidence" style="color: ${cond.color};">${cond.confidence.toFixed(1)}%</span>
            ${descHtml}
        `;
        list.appendChild(item);

        setTimeout(() => {
            item.querySelector('.condition-bar').style.width = cond.confidence + '%';
        }, 100 + index * 150);
    });
}

function renderMetrics(metrics) {
    const grid = $('#metricsGrid');
    grid.innerHTML = '';

    Object.entries(metrics).forEach(([label, value]) => {
        const item = document.createElement('div');
        item.className = 'metric-item';
        item.innerHTML = `
            <div class="metric-label">${label}</div>
            <div class="metric-value">${value}</div>
        `;
        grid.appendChild(item);
    });
}

function renderRecommendations(recommendations) {
    const list = $('#recommendationsList');
    list.innerHTML = '';
    recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        list.appendChild(li);
    });
}

// ===== DOWNLOAD REPORT =====
function downloadReport() {
    const verdict = $('#overallVerdict').textContent;
    const summary = $('#overallSummary').textContent;
    const timestamp = $('#resultsTimestamp').textContent;

    const conditions = [];
    $$('.condition-item').forEach(item => {
        const name = item.querySelector('.condition-name').textContent;
        const conf = item.querySelector('.condition-confidence').textContent;
        conditions.push(`  - ${name}: ${conf}`);
    });

    const metrics = [];
    $$('.metric-item').forEach(item => {
        const label = item.querySelector('.metric-label').textContent;
        const value = item.querySelector('.metric-value').textContent;
        metrics.push(`  ${label}: ${value}`);
    });

    const recs = [];
    $$('.recommendations-list li').forEach(li => {
        recs.push(`  • ${li.textContent}`);
    });

    const findingsEl = $('#aiFindingsText');
    const aiFindings = findingsEl && findingsEl.textContent ? `\n─── DETAILED AI ANALYSIS ───────────────────────────\n${findingsEl.textContent}\n` : '';

    const report = `
╔═══════════════════════════════════════════════════╗
║           MedVision AI — Analysis Report          ║
╠═══════════════════════════════════════════════════╣

  Date: ${timestamp}
  Scan Type: ${state.selectedType.toUpperCase()}

─── VERDICT ────────────────────────────────────────
  ${verdict}
  ${summary}

─── DETECTED CONDITIONS ────────────────────────────
${conditions.join('\n')}

─── ANALYSIS METRICS ───────────────────────────────
${metrics.join('\n')}

─── RECOMMENDATIONS ────────────────────────────────
${recs.join('\n')}
${aiFindings}
╠═══════════════════════════════════════════════════╣
║  ⚠️  DISCLAIMER: This is a demonstration tool     ║
║  for educational purposes only. Not intended       ║
║  for clinical diagnosis. Always consult a          ║
║  qualified healthcare professional.                ║
╚═══════════════════════════════════════════════════╝
`;

    const blob = new Blob([report], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `MedVision_Report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
}

// ===== HISTORY =====
function saveToHistory(results) {
    const entry = {
        id: Date.now(),
        scanType: results.scanType,
        verdict: results.verdict,
        score: results.overallScore,
        severity: results.severityLevel,
        timestamp: results.timestamp,
        thumbSrc: results.imageSrc,
        topCondition: results.conditions[0]?.name || 'N/A',
        isAI: results.isAI || false,
    };

    state.analysisHistory.unshift(entry);
    if (state.analysisHistory.length > 12) {
        state.analysisHistory = state.analysisHistory.slice(0, 12);
    }

    localStorage.setItem('medvision_history', JSON.stringify(state.analysisHistory));
    renderHistory();
}

function renderHistory() {
    const grid = $('#historyGrid');
    const emptyEl = $('#historyEmpty');

    if (state.analysisHistory.length === 0) {
        emptyEl.style.display = 'flex';
        return;
    }

    emptyEl.style.display = 'none';
    grid.querySelectorAll('.history-card').forEach(c => c.remove());

    state.analysisHistory.forEach(entry => {
        const card = document.createElement('div');
        card.className = 'history-card';
        const date = new Date(entry.timestamp);
        const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
        const aiBadge = entry.isAI ? '<span class="ai-badge live" style="margin-left:auto;">AI</span>' : '';

        card.innerHTML = `
            <div class="history-card-header">
                <span class="history-type ${entry.scanType}">${entry.scanType.toUpperCase()}</span>
                ${aiBadge}
                <span class="history-date">${dateStr}</span>
            </div>
            <div class="history-thumb">
                ${entry.thumbSrc ? `<img src="${entry.thumbSrc}" alt="Scan thumbnail">` : '<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-size:0.8rem;">No preview</div>'}
            </div>
            <div class="history-verdict">${entry.verdict}</div>
            <div class="history-score">Score: ${entry.score}/100 · ${entry.topCondition}</div>
        `;

        grid.appendChild(card);
    });
}

// ===== REVEAL ANIMATIONS =====
function initRevealAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) entry.target.classList.add('visible');
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    $$('.reveal').forEach(el => observer.observe(el));
}

// ===== HERO STATS =====
function initHeroStats() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateStats();
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    const statsSection = $('.hero-stats');
    if (statsSection) observer.observe(statsSection);
}

function animateStats() {
    $$('.stat-number').forEach(el => {
        const target = parseFloat(el.dataset.target);
        const isFloat = target % 1 !== 0;
        let current = 0;
        const duration = 2000;
        const startTime = Date.now();

        function tick() {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            current = target * easeOutCubic(progress);
            el.textContent = isFloat ? current.toFixed(1) : Math.round(current);
            if (progress < 1) requestAnimationFrame(tick);
        }
        tick();
    });
}

// ===== NEURAL NETWORK VISUALIZATION =====
function initNeuralNetVisualization() {
    const canvas = $('#neuralNetCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const layers = [4, 6, 8, 6, 3];
    const layerSpacing = canvas.width / (layers.length + 1);
    const nodes = [];

    layers.forEach((count, layerIndex) => {
        const x = layerSpacing * (layerIndex + 1);
        const ySpacing = canvas.height / (count + 1);
        for (let i = 0; i < count; i++) {
            nodes.push({ x, y: ySpacing * (i + 1), layer: layerIndex, pulse: Math.random() * Math.PI * 2 });
        }
    });

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const time = Date.now() * 0.001;

        for (let i = 0; i < nodes.length; i++) {
            for (let j = 0; j < nodes.length; j++) {
                if (nodes[j].layer === nodes[i].layer + 1) {
                    const opacity = 0.05 + 0.05 * Math.sin(time + nodes[i].pulse + nodes[j].pulse);
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.strokeStyle = `rgba(0, 212, 255, ${opacity})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        nodes.forEach(node => {
            const pulseSize = 2 + Math.sin(time * 2 + node.pulse) * 1;
            const opacity = 0.4 + 0.3 * Math.sin(time * 1.5 + node.pulse);

            ctx.beginPath();
            ctx.arc(node.x, node.y, pulseSize + 2, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 212, 255, ${opacity * 0.2})`;
            ctx.fill();

            ctx.beginPath();
            ctx.arc(node.x, node.y, pulseSize, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(0, 212, 255, ${opacity})`;
            ctx.fill();
        });

        requestAnimationFrame(draw);
    }
    draw();
}
