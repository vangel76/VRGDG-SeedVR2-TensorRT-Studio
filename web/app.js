const $ = (selector) => document.querySelector(selector);
const before = $('#before');
const after = $('#after');
const inspector = $('#inspector');
const stage = $('#stage');
const afterPane = document.querySelector('.after-pane');
const divider = $('#divider');
let fps = 30;
let syncFrame = 0;
let zoom = 1, panX = 0, panY = 0, originX = 50, originY = 50, drag = '', dragX = 0, dragY = 0, baseX = 0, baseY = 0;
let currentOutputPath = '', currentOutputReprocessable = false;
let selectedSourceFile = null;
const SETTINGS_STORAGE_KEY = 'seedvr-studio-render-settings-v2';
const LEGACY_SETTINGS_STORAGE_KEY = 'seedvr-studio-render-settings-v1';
const SAVED_SETTING_IDS = [
  'backend', 'preset', 'output-preset', 'source-scale', 'crop-policy', 'chunked-render', 'chunk-seconds',
  'model', 'batch-size', 'seed', 'color', 'attention', 'blocks', 'vae-tiling',
  'sharpen', 'sharpen-strength', 'microtexture', 'microtexture-strength',
  'skin-finishing', 'skin-evenness', 'skin-smoothing', 'skin-redness', 'skin-shine', 'blemish-mode', 'preserve-marks',
  'grain', 'grain-intensity', 'grain-saturation', 'seam-enabled',
  'face-model', 'face-strength', 'face-fidelity', 'face-detail', 'face-min-size', 'rtx-quality',
  'preview-start', 'preview-seconds', 'source-path'
];
const loadSavedSettings = () => {
  try {
    const current = JSON.parse(localStorage.getItem(SETTINGS_STORAGE_KEY) || 'null');
    if (current?.values) return current;
    const legacy = JSON.parse(localStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY) || 'null');
    if (!legacy?.values) return null;
    // The first settings implementation could capture Chromium's restored SDPA
    // value before our intended default was applied. Migrate everything else,
    // but repair attention once; future explicit saves remain authoritative.
    legacy.values.attention = 'sageattn_2';
    legacy.version = 2;
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(legacy));
    return legacy;
  } catch (_) { return null; }
};
const refreshFaceUi = () => { const off = $('#face-model').value === 'none'; $('#face-options').classList.toggle('inactive', off); $('#face-options').querySelectorAll('input').forEach((control) => { control.disabled = off; }); $('#face-fidelity').disabled = off || $('#face-model').value !== 'codeformer'; };
const refreshSettingsUi = () => {
  refreshFaceUi();
  $('#chunk-length-wrap').hidden = !$('#chunked-render').checked;
  $('#seam-warning').hidden = $('#seam-enabled').checked;
  $('#blocks-value').textContent = $('#blocks').value;
  const skinOff = !$('#skin-finishing').checked;
  $('#skin-options').classList.toggle('inactive', skinOff);
  $('#skin-options').querySelectorAll('input,select').forEach((control) => { control.disabled = skinOff; });
  ['sharpen-strength', 'microtexture-strength', 'skin-evenness', 'skin-smoothing', 'skin-redness', 'skin-shine', 'grain-intensity', 'grain-saturation'].forEach((id) => document.getElementById(id)?.dispatchEvent(new Event('input')));
};
const restoreSettings = () => {
  const saved = loadSavedSettings();
  const values = {attention:'sageattn_2', color:'none', ...(saved?.values || {})};
  Object.entries(values).forEach(([id, value]) => {
    const element = document.getElementById(id);
    if (!element) return;
    if (element.type === 'checkbox') { element.checked = Boolean(value); return; }
    if (element.tagName === 'SELECT' && !Array.from(element.options).some((option) => option.value === String(value))) return;
    element.value = String(value);
  });
  // Browser form restoration must never silently replace the unsaved default.
  if (!saved) { $('#backend').value = 'SeedVR2 + TensorRT'; $('#attention').value = 'sageattn_2'; $('#color').value = 'none'; }
  refreshSettingsUi();
  $('#settings-save-status').textContent = saved ? 'Saved settings loaded' : 'Using app defaults';
  if (typeof loadSourcePath === 'function') loadSourcePath();
  updateOutputSizeNote();
  renderPreviewBand();
  if (typeof refreshEngineUi === 'function') refreshEngineUi();
};
const saveCurrentSettings = () => {
  const values = {};
  SAVED_SETTING_IDS.forEach((id) => { const element = document.getElementById(id); if (element) values[id] = element.type === 'checkbox' ? element.checked : element.value; });
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify({version:1, savedAt:new Date().toISOString(), values}));
    $('#settings-save-status').textContent = 'Saved — these settings will load next time';
  } catch (_) { $('#settings-save-status').textContent = 'Could not save settings in this browser'; }
};
// Establish the real defaults immediately, before Chromium restores old form state.
$('#attention').value = 'sageattn_2';
$('#color').value = 'none';

const format = (value) => { const v = Math.max(0, Number(value) || 0); const m = Math.floor(v / 60); const s = Math.floor(v % 60); const ms = Math.floor((v % 1) * 1000); return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${String(ms).padStart(3,'0')}`; };
const loadedVideos = () => [before, after].filter((video) => video.readyState >= 1 && Number.isFinite(video.duration) && video.duration > 0);
const playbackVideo = () => loadedVideos()[0];
// The restored video may cover only a preview slot of the original timeline: [restoredOffset, restoredOffset + after.duration).
let restoredOffset = 0, restoredSlot = false;
const hasVideo = (video) => loadedVideos().includes(video);
const duration = () => { if (hasVideo(before)) return before.duration; return hasVideo(after) ? after.duration : 0; };
const afterTimeFor = (t) => hasVideo(after) ? Math.max(0, Math.min(after.duration, t - restoredOffset)) : 0;
const inRestoredSlot = (t) => !restoredSlot || !hasVideo(after) || (t >= restoredOffset - 0.02 && t < restoredOffset + after.duration + 0.02);
const restoredSlotLabel = () => hasVideo(after) ? `${format(restoredOffset)} – ${format(restoredOffset + after.duration)}` : '';
const applySlotVisibility = () => { const t = before === playbackVideo() ? before.currentTime : after.currentTime + restoredOffset; const outside = restoredSlot && hasVideo(after) && !inRestoredSlot(t); afterPane.classList.toggle('outside-slot', outside); const label = afterPane.querySelector('span'); if (label) label.textContent = outside ? `Restored preview covers ${restoredSlotLabel()}` : (restoredSlot ? `Restored · preview ${restoredSlotLabel()}` : 'Restored'); if (outside && !after.paused) after.pause(); };
const setRestoredSlot = (start, enabled) => { restoredOffset = enabled ? Math.max(0, Number(start) || 0) : 0; restoredSlot = Boolean(enabled); applySlotVisibility(); };
const update = () => { const d = duration(); const pv = playbackVideo(); const time = !pv ? 0 : (pv === before ? pv.currentTime : pv.currentTime + restoredOffset); applySlotVisibility(); $('#seek').max = d; $('#seek').value = time; $('#clock').textContent = `${format(time)} / ${format(d)}`; const total = Math.max(0, Math.round(d * fps) - 1); $('#frame').max = total; $('#frame').value = Math.min(total, Math.round(time * fps)); $('#frame-total').textContent = `/ ${total}`;  if (typeof renderPreviewBand === 'function') renderPreviewBand(); };
const previewStrip = $('#preview-strip'), previewBand = $('#preview-band'), previewLabel = $('#preview-band-label'), previewPlayhead = $('#preview-playhead');
const previewStartInput = $('#preview-start'), previewLengthInput = $('#preview-seconds');
const clampPreview = (start, length) => {
  const d = duration();
  const maxLength = Number(previewLengthInput.max) || 20;
  length = Math.max(Number(previewLengthInput.min) || 0.5, Math.min(maxLength, Number(length) || 0));
  if (d > 0) { start = Math.max(0, Math.min(Math.max(0, d - 0.1), Number(start) || 0)); length = Math.min(length, Math.max(0.5, d - start)); }
  else start = Math.max(0, Number(start) || 0);
  if (fps > 0) { start = Math.floor(start * fps + 1e-6) / fps; length = Math.max(1, Math.round(length * fps)) / fps; }
  return { start: Math.round(start * 1000) / 1000, length: Math.round(length * 1000) / 1000 };
};
const renderPreviewBand = () => {
  const d = duration();
  const { start, length } = clampPreview(previewStartInput.value, previewLengthInput.value);
  if (!(d > 0)) { previewBand.style.left = '0%'; previewBand.style.width = '0%'; previewLabel.textContent = ''; previewPlayhead.style.left = '0%'; return; }
  previewBand.style.left = `${(start / d) * 100}%`; previewBand.style.width = `${(Math.min(length, d - start) / d) * 100}%`;
  previewLabel.textContent = `${format(start)} – ${format(start + length)} · ${length.toFixed(1)}s`;
  previewPlayhead.style.left = `${((playbackVideo()?.currentTime || 0) / d) * 100}%`;
};
const setPreview = (start, length) => { const clamped = clampPreview(start, length); previewStartInput.value = String(Number(clamped.start.toFixed(3))); previewLengthInput.value = clamped.length; renderPreviewBand(); };
const stripTime = (clientX) => { const rect = previewStrip.getBoundingClientRect(); return Math.max(0, Math.min(1, (clientX - rect.left) / Math.max(1, rect.width))) * duration(); };
let previewDrag = null;
previewStrip.addEventListener('pointerdown', (event) => {
  if (!(duration() > 0)) return;
  const handle = event.target.closest('[data-handle]')?.dataset.handle;
  const onBand = Boolean(event.target.closest('#preview-band'));
  const current = clampPreview(previewStartInput.value, previewLengthInput.value);
  const t = stripTime(event.clientX);
  previewDrag = { mode: handle || (onBand ? 'move' : 'place'), grabOffset: t - current.start, start: current.start, length: current.length, end: current.start + current.length };
  if (previewDrag.mode === 'place') { setPreview(t - current.length / 2, current.length); previewDrag.mode = 'move'; previewDrag.grabOffset = current.length / 2; }
  previewBand.classList.add('dragging'); previewStrip.setPointerCapture(event.pointerId); event.preventDefault();
});
previewStrip.addEventListener('pointermove', (event) => {
  if (!previewDrag) return;
  const t = stripTime(event.clientX);
  if (previewDrag.mode === 'move') setPreview(t - previewDrag.grabOffset, previewDrag.length);
  else if (previewDrag.mode === 'start') { const start = Math.min(t, previewDrag.end - 0.5); setPreview(start, previewDrag.end - start); }
  else if (previewDrag.mode === 'end') setPreview(previewDrag.start, t - previewDrag.start);
});
['pointerup', 'pointercancel'].forEach((name) => previewStrip.addEventListener(name, (event) => { if (!previewDrag) return; previewDrag = null; previewBand.classList.remove('dragging'); try { previewStrip.releasePointerCapture(event.pointerId); } catch (_) {} }));
previewStrip.addEventListener('dblclick', (event) => { if (duration() > 0) seek(stripTime(event.clientX)); });
$('#preview-here').addEventListener('click', () => setPreview(playbackVideo()?.currentTime || 0, previewLengthInput.value));
[previewStartInput, previewLengthInput].forEach((input) => { input.addEventListener('input', renderPreviewBand); input.addEventListener('change', () => setPreview(previewStartInput.value, previewLengthInput.value)); });
const seek = (value) => { const t = Math.max(0, Math.min(duration(), Number(value) || 0)); if (hasVideo(before)) before.currentTime = t; if (hasVideo(after)) after.currentTime = hasVideo(before) ? afterTimeFor(t) : t; update(); };
const stopPlaybackSync = () => { if (syncFrame) cancelAnimationFrame(syncFrame); syncFrame = 0; after.playbackRate = 1; };
const syncTime = (source, target) => {
  if (!hasVideo(target)) return;
  const wanted = source === before ? afterTimeFor(source.currentTime) : Math.min(before.duration, source.currentTime + restoredOffset);
  if (Math.abs(target.currentTime - wanted) > 0.08) target.currentTime = wanted;
  if (source === before && target === after && !before.paused) { const inside = inRestoredSlot(before.currentTime); if (inside && after.paused && !after.ended) after.play().catch(() => {}); else if (!inside && !after.paused) after.pause(); }
};
const pause = () => { before.pause(); after.pause(); stopPlaybackSync(); $('#play').textContent = '▶'; };
const play = async () => { const video = playbackVideo(); if (!video) return; if (video.paused) { try { const t = video === before ? video.currentTime : video.currentTime + restoredOffset; seek(t >= duration() ? 0 : t); const targets = loadedVideos().filter((item) => item !== after || inRestoredSlot(before.currentTime)); await Promise.all(targets.map((item) => item.play())); $('#play').textContent = 'Ⅱ'; } catch (_) { pause(); } } else pause(); };
const setWipe = (value) => { const p = Math.max(0, Math.min(100, Number(value) || 50)); $('#wipe').value = p; afterPane.style.setProperty('clip-path', `inset(0 0 0 ${p}%)`, inspector.classList.contains('mode-compare') ? 'important' : ''); divider.style.left = `${p}%`; $('#wipe-value').textContent = `${Math.round(p)}%`; };
// Scale at which the video is drawn when zoom = 1 (object-fit: contain), relative to its real pixels.
const fitScale = () => {
  const video = (after.videoWidth && after.readyState >= 1) ? after : before;
  if (!video.videoWidth || !video.videoHeight) return 1;
  const pane = video.parentElement.getBoundingClientRect();
  if (!pane.width || !pane.height) return 1;
  return Math.min(pane.width / video.videoWidth, pane.height / video.videoHeight);
};
const realZoomPercent = () => Math.round(zoom * fitScale() * 100);
const maxZoom = () => Math.max(8, 1 / fitScale() * 4);
const minZoom = () => Math.min(1, 1 / fitScale());  // small videos may shrink below fit to reach 1:1
const applyTransform = () => { [before, after].forEach((video) => { video.style.transformOrigin = `${originX}% ${originY}%`; video.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`; }); $('#zoom-value').textContent = `${realZoomPercent()}%`; };
const resetZoom = () => { zoom = 1; panX = panY = 0; originX = originY = 50; applyTransform(); };
const zoomToActualPixels = () => { const scale = fitScale(); if (!scale) return; zoom = 1 / scale; panX = panY = 0; originX = originY = 50; applyTransform(); };
window.addEventListener('resize', applyTransform);
[before, after].forEach((video) => video.addEventListener('loadedmetadata', applyTransform));
const setMode = (mode) => { ['original','restored','compare','side'].forEach((name) => inspector.classList.toggle(`mode-${name}`, name === mode)); document.querySelectorAll('[data-mode]').forEach((button) => button.classList.toggle('active', button.dataset.mode === mode)); afterPane.style.setProperty('clip-path', mode === 'compare' ? `inset(0 0 0 ${$('#wipe').value}%)` : 'none', 'important'); resetZoom(); updateEmptyState(); };
const updateEmptyState = () => { const hasBefore = Boolean(before.currentSrc || before.src); const hasAfter = Boolean(after.currentSrc || after.src); const mode = ['original','restored','compare','side'].find((name) => inspector.classList.contains(`mode-${name}`)) || 'compare'; const ready = mode === 'original' ? hasBefore : mode === 'restored' ? hasAfter : hasBefore && hasAfter; $('#empty-state').hidden = ready; if (ready) return; if (mode === 'restored' && !hasAfter) { $('#empty-title').textContent = 'Waiting for restored video'; $('#empty-subtitle').textContent = 'Render a preview or full video and it will appear here automatically.'; } else if (!hasBefore) { $('#empty-title').textContent = 'Waiting for original video'; $('#empty-subtitle').textContent = 'Upload an original video to begin.'; } else { $('#empty-title').textContent = 'Waiting for restored video'; $('#empty-subtitle').textContent = 'Render a preview or full video and it will appear here automatically.'; } };
let sourceDimensions = null;
const updateSourceScaleNote = () => {
  const note = $('#source-scale-note'); const divisor = Number(formValue('source-scale')) || 1;
  if (!note) return;
  if (divisor <= 1 || !sourceDimensions) { note.textContent = 'For blurry sources. Output size still counts from the original size.'; return; }
  const even = (value) => Math.max(2, Math.floor(value / 2) * 2);
  note.textContent = `SeedVR2 gets ${even(sourceDimensions.width / divisor)}×${even(sourceDimensions.height / divisor)} · output size still from ${sourceDimensions.width}×${sourceDimensions.height}`;
};
const updateOutputSizeNote = () => {
  updateSourceScaleNote();
  const note = $('#output-size-note'); if (!note) return;
  const preset = (window.__seedvrOutputPresets || {})[$('#output-preset').value];
  if (!preset) { note.textContent = ''; return; }
  const even = (value) => Math.max(2, Math.round(value / 2) * 2);
  if (preset.scale != null) {
    if (!sourceDimensions) { note.textContent = `${preset.scale}× the source size · load a video to see the result`; return; }
    const w = even(sourceDimensions.width * preset.scale), h = even(sourceDimensions.height * preset.scale);
    note.textContent = `${sourceDimensions.width}×${sourceDimensions.height} → ${w}×${h}`;
    return;
  }
  if (sourceDimensions) {
    const landscape = sourceDimensions.width >= sourceDimensions.height;
    const shortSide = preset.resolution; let longSide = Math.round(shortSide * Math.max(sourceDimensions.width, sourceDimensions.height) / Math.min(sourceDimensions.width, sourceDimensions.height));
    if (preset.max_resolution && longSide > preset.max_resolution) { const k = preset.max_resolution / longSide; longSide = preset.max_resolution; note.textContent = `short side ${Math.round(shortSide * k)} · long side capped at ${longSide}`; return; }
    note.textContent = landscape ? `${even(longSide)}×${even(shortSide)}` : `${even(shortSide)}×${even(longSide)}`;
    return;
  }
  note.textContent = `short side ${preset.resolution}px · long side up to ${preset.max_resolution}px`;
};
before.addEventListener('loadedmetadata', () => { if (before.videoWidth) { sourceDimensions = {width: before.videoWidth, height: before.videoHeight}; updateOutputSizeNote(); } });
const loadVideo = (video, url) => { if (url) video.src = url; else video.removeAttribute('src'); video.load(); updateEmptyState(); update(); };
const setFile = (video, file, nameTarget) => { if (!file) return; if (video === before) beginNewProject(file); loadVideo(video, URL.createObjectURL(file)); $(nameTarget).textContent = file.name; $('#media-info').textContent = `${file.name} · local file`; };

const sourceInput = $('#source-file');
const sourceDropzone = document.querySelector('label[for="source-file"]');
const isVideoFile = (file) => Boolean(file && (file.type.startsWith('video/') || /\.(mp4|mov|mkv|avi|webm|m4v|mpeg|mpg)$/i.test(file.name)));
const acceptSourceFile = (file) => {
  if (!isVideoFile(file)) { $('#media-info').textContent = 'Please choose a supported video file.'; return; }
  selectedSourceFile = file;
  // A dropped file replaces a previously loaded path; keep the field only when it names a folder.
  const typedPath = sourcePathInput.value.trim();
  if (typedPath && (typedPath === lastLoadedSourcePath || isVideoFile({ name: typedPath, type: '' }))) { sourcePathInput.value = ''; setPathStatus('Paste a path, then press Load or Enter'); }
  lastLoadedSourcePath = '';
  setFile(before, file, '#source-name');
};
sourceInput.addEventListener('change', (event) => acceptSourceFile(event.target.files[0]));
const sourcePathInput = $('#source-path');
const sourcePathStatus = $('#source-path-status');
let lastLoadedSourcePath = '';
const setPathStatus = (text, kind = '') => { sourcePathStatus.textContent = text; sourcePathStatus.className = `file-name ${kind}`.trim(); };
const loadSourcePath = async (force = false) => {
  const path = sourcePathInput.value.trim();
  if (!path) { setPathStatus('Paste a path, then press Load or Enter'); return; }
  if (!force && path === lastLoadedSourcePath) return;
  setPathStatus(`Loading ${path}…`);
  try {
    const response = await fetch('/api/local-video/info?path=' + encodeURIComponent(path), {cache:'no-store'});
    if (!response.ok) {
      let detail = (await response.json().catch(() => ({}))).detail;
      const oldServer = detail === 'Not Found';
      if (oldServer) detail = 'Studio server is running old code · restart SeedVR Studio, then hard-refresh this page';
      if (response.status === 404 && !oldServer && !/\.[a-z0-9]{2,4}$/i.test(path)) { setPathStatus('Folder set · full renders will be saved there'); return; }
      setPathStatus(detail || (response.status === 404 ? `Not found (restart Studio if the server is old): ${path}` : `Could not read ${path} (${response.status})`), 'error');
      return;
    }
    const info = await response.json();
    lastLoadedSourcePath = path;
    selectedSourceFile = null; sourceInput.value = '';
    beginNewProject({ name: info.name });
    if (Number(info.fps) > 0) fps = Number(info.fps);
    loadVideo(before, '/api/local-video?path=' + encodeURIComponent(path));
    $('#source-name').textContent = info.name;
    setPathStatus(`Loaded ${info.name} · ${info.fps ? `${info.fps} fps · ` : ''}result goes to ${info.folder}`, 'ok');
    $('#media-info').textContent = `${info.name} · local path`;
  } catch (error) { setPathStatus(`Could not read ${path}: ${error.message}`, 'error'); }
};
$('#load-source-path').addEventListener('click', () => loadSourcePath(true));
$('#browse-source').addEventListener('click', async () => {
  const button = $('#browse-source'); button.disabled = true; setPathStatus('Choose a video in the file dialog…');
  try {
    const data = new FormData(); data.append('initial', sourcePathInput.value.trim());
    const response = await fetch('/api/pick-file', {method:'POST', body:data});
    if (!response.ok) { const detail = (await response.json().catch(() => ({}))).detail; setPathStatus(detail || (response.status === 404 ? 'Studio server is running old code · restart SeedVR Studio' : `File dialog failed (${response.status})`), 'error'); return; }
    const result = await response.json();
    if (result.cancelled || !result.path) { setPathStatus(sourcePathInput.value.trim() ? 'Dialog cancelled' : 'Paste a path, then press Load or Enter'); return; }
    sourcePathInput.value = result.path; await loadSourcePath(true);
  } catch (error) { setPathStatus(`File dialog failed: ${error.message}`, 'error'); }
  finally { button.disabled = false; }
});
sourcePathInput.addEventListener('change', () => loadSourcePath());
sourcePathInput.addEventListener('keydown', (event) => { if (event.key === 'Enter') { event.preventDefault(); loadSourcePath(true); } });
sourcePathInput.addEventListener('paste', () => setTimeout(() => loadSourcePath(), 0));
before.addEventListener('error', () => { if (before.src.includes('/api/local-video')) setPathStatus(`Browser could not decode ${lastLoadedSourcePath} (codec unsupported in this browser)`, 'error'); });
['dragenter','dragover'].forEach((name) => sourceDropzone.addEventListener(name, (event) => {
  event.preventDefault(); event.stopPropagation(); sourceDropzone.classList.add('drag-over');
}));
['dragleave','drop'].forEach((name) => sourceDropzone.addEventListener(name, (event) => {
  event.preventDefault(); event.stopPropagation(); sourceDropzone.classList.remove('drag-over');
}));
sourceDropzone.addEventListener('drop', (event) => acceptSourceFile(event.dataTransfer?.files?.[0]));
['dragover','drop'].forEach((name) => document.addEventListener(name, (event) => {
  if (Array.from(event.dataTransfer?.types || []).includes('Files')) event.preventDefault();
}));
document.querySelectorAll('[data-mode]').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));
$('#seek').addEventListener('input', (event) => seek(event.target.value));
$('#frame').addEventListener('change', (event) => seek((Number(event.target.value) || 0) / fps));
$('#prev').addEventListener('click', () => { pause(); seek((playbackVideo()?.currentTime || 0) - 1 / fps); });
$('#next').addEventListener('click', () => { pause(); seek((playbackVideo()?.currentTime || 0) + 1 / fps); });
$('#play').addEventListener('click', play);
$('#wipe').addEventListener('input', (event) => setWipe(event.target.value));
$('#zoom-reset').addEventListener('click', resetZoom);
$('#zoom-actual').addEventListener('click', zoomToActualPixels);
$('#fullscreen').addEventListener('click', () => (inspector.requestFullscreen || inspector.webkitRequestFullscreen)?.call(inspector));
$('#exit-studio').addEventListener('click', async () => { if (!window.confirm('Exit SeedVR Studio? Any active render will stop and GPU/RAM will be released.')) return; $('#render-status').textContent = 'Shutting down…'; try { await fetch('/api/shutdown', {method:'POST', keepalive:true}); } catch (_) {} window.close(); });
[before, after].forEach((video) => {
  video.addEventListener('timeupdate', () => { if (video === playbackVideo()) { syncTime(video, video === before ? after : before); update(); } });
  video.addEventListener('ended', () => { if (video === after && restoredSlot && hasVideo(before)) return; pause(); });
  ['loadedmetadata', 'durationchange', 'emptied'].forEach((event) => video.addEventListener(event, update));
});
stage.addEventListener('wheel', (event) => { event.preventDefault(); const rect = stage.getBoundingClientRect(); originX = Math.max(0, Math.min(100, ((event.clientX - rect.left) / rect.width) * 100)); originY = Math.max(0, Math.min(100, ((event.clientY - rect.top) / rect.height) * 100)); zoom = Math.max(minZoom(), Math.min(maxZoom(), zoom * (event.deltaY < 0 ? 1.2 : 1 / 1.2))); if (zoom <= 1) panX = panY = 0; applyTransform(); }, { passive:false });
stage.addEventListener('dblclick', resetZoom);
stage.addEventListener('pointerdown', (event) => { drag = event.target.closest('#divider') ? 'wipe' : (zoom > 1 ? 'pan' : ''); if (!drag) return; stage.setPointerCapture?.(event.pointerId); dragX = event.clientX; dragY = event.clientY; baseX = panX; baseY = panY; });
stage.addEventListener('pointermove', (event) => { if (drag === 'wipe') setWipe(((event.clientX - stage.getBoundingClientRect().left) / stage.getBoundingClientRect().width) * 100); if (drag === 'pan') { panX = baseX + event.clientX - dragX; panY = baseY + event.clientY - dragY; applyTransform(); } });
stage.addEventListener('pointerup', () => { drag = ''; });
inspector.addEventListener('keydown', (event) => { if (event.target.matches('input')) return; if (event.code === 'Space') { event.preventDefault(); play(); } if (event.key === 'ArrowLeft') $('#prev').click(); if (event.key === 'ArrowRight') $('#next').click(); });

let selectedWorkspaceOutput = null;
const outputPicker = $('#output-picker'); const outputPickerToggle = $('#output-picker-toggle'); const outputPickerMenu = $('#output-picker-menu'); const outputPickerLabel = $('#output-picker-label'); const outputPickerThumb = $('#output-picker-thumb');
const closeOutputPicker = () => { outputPickerMenu.hidden = true; outputPickerToggle.setAttribute('aria-expanded', 'false'); };
const selectWorkspaceOutput = (output) => { selectedWorkspaceOutput = output; outputPickerLabel.textContent = output.name; outputPickerThumb.src = '/api/output-thumbnail?path=' + encodeURIComponent(output.path); outputPickerToggle.classList.add('has-selection'); closeOutputPicker(); };
outputPickerToggle.addEventListener('click', () => { outputPickerMenu.hidden = !outputPickerMenu.hidden; outputPickerToggle.setAttribute('aria-expanded', String(!outputPickerMenu.hidden)); });
document.addEventListener('click', (event) => { if (!outputPicker.contains(event.target)) closeOutputPicker(); });
async function loadOutputs() {
  try {
    const response = await fetch('/api/outputs'); const outputs = await response.json(); outputPickerMenu.innerHTML = ''; selectedWorkspaceOutput = null; outputPickerLabel.textContent = outputs.length ? 'Select a workspace output…' : 'No workspace outputs yet'; outputPickerThumb.removeAttribute('src'); outputPickerToggle.classList.remove('has-selection');
    outputs.forEach((output) => { const item = document.createElement('button'); item.type = 'button'; item.className = 'output-picker-item'; item.setAttribute('role', 'option'); const thumb = document.createElement('img'); thumb.className = 'output-item-thumb'; thumb.alt = ''; thumb.setAttribute('aria-hidden', 'true'); thumb.src = '/api/output-thumbnail?path=' + encodeURIComponent(output.path); const copy = document.createElement('span'); copy.className = 'output-item-copy'; const name = document.createElement('strong'); name.textContent = output.name; const kind = document.createElement('small'); kind.textContent = output.reprocessable ? 'TensorRT result' : 'Workspace result'; copy.append(name, kind); item.append(thumb, copy); item.addEventListener('click', () => selectWorkspaceOutput(output)); outputPickerMenu.append(item); });
    $('#api-dot').classList.add('ok'); $('#api-status').textContent = 'API ready · ' + outputs.length + ' output' + (outputs.length === 1 ? '' : 's');
  } catch (_) { $('#api-status').textContent = 'API unavailable'; }
}$('#load-output').addEventListener('click', () => { const output = selectedWorkspaceOutput; if (!output) return; currentOutputPath = output.path || output.url; currentOutputReprocessable = Boolean(output.reprocessable); fps = Number(output.fps) || 30; setRestoredSlot(0, false); $('#reprocess-post').hidden = !currentOutputReprocessable; $('#open-output-folder').hidden = false; loadVideo(after, output.url); $('#media-info').textContent = currentOutputReprocessable ? 'Previous result loaded · post-only reprocess available' : 'Previous workspace result loaded'; });
$('#open-output-folder').addEventListener('click', async () => { if (!currentOutputPath) return; const data = new FormData(); data.append('output_path', currentOutputPath); try { const response = await fetch('/api/open-folder', {method:'POST', body:data}); if (!response.ok) throw new Error(await response.text()); } catch (error) { $('#media-info').textContent = `Could not open folder: ${error.message}`; } });
const chunkToggle = $('#chunked-render');
const chunkLengthWrap = $('#chunk-length-wrap');
chunkToggle.addEventListener('change', () => { chunkLengthWrap.hidden = !chunkToggle.checked; });
$('#chunk-help').addEventListener('click', () => $('#chunk-help-dialog').showModal());
const seamToggle = $('#seam-enabled');
seamToggle.addEventListener('change', () => { $('#seam-warning').hidden = seamToggle.checked; });
$('#skin-finishing').addEventListener('change', refreshSettingsUi);
$('#face-model').addEventListener('change', refreshFaceUi);
$('#save-settings').addEventListener('click', saveCurrentSettings);
window.addEventListener('pageshow', () => setTimeout(restoreSettings, 0));
setWipe(50); applyTransform(); updateEmptyState(); update(); loadOutputs();

let activeJob = null, lastFailedJob = null, elapsedTimer = null, elapsedStarted = 0, projectToken = 0;
const clearWorkspace = async () => { try { await fetch('/api/workspace/clear', {method:'POST'}); } catch (_) {} loadOutputs(); };
const beginNewProject = (file) => { clearWorkspace(); projectToken += 1; setRestoredSlot(0, false); const previousJob = activeJob; activeJob = null; lastFailedJob = null; currentOutputPath = ''; currentOutputReprocessable = false; stopElapsed(0); setRenderBusy(false); loadVideo(after, ''); $('#reprocess-post').hidden = true; $('#open-output-folder').hidden = true; $('#media-info').textContent = `${file.name} · new project`; $('#render-status').textContent = previousJob ? 'New project ready · previous render continues in background' : 'Ready'; };
const formatElapsed = (seconds) => { const total = Math.max(0, Math.floor(Number(seconds) || 0)); return `${String(Math.floor(total / 60)).padStart(2,'0')}:${String(total % 60).padStart(2,'0')}`; };
const updateEta = (progress) => { const p = Number(progress) || 0; const elapsed = (performance.now() - elapsedStarted) / 1000; if (p < 0.05 || elapsed < 5) { $('#eta-time').textContent = 'Estimating…'; return; } $('#eta-time').textContent = `~${formatElapsed(elapsed * (1 - Math.min(p, 0.99)) / p)}`; };
const startElapsed = () => { clearInterval(elapsedTimer); elapsedStarted = performance.now(); $('#elapsed-time').textContent = '00:00'; $('#eta-time').textContent = 'Estimating…'; elapsedTimer = setInterval(() => { const elapsed = (performance.now() - elapsedStarted) / 1000; $('#elapsed-time').textContent = formatElapsed(elapsed); updateEta(window.__seedvrProgress || 0); }, 250); };
const stopElapsed = (seconds) => { clearInterval(elapsedTimer); elapsedTimer = null; $('#elapsed-time').textContent = formatElapsed(seconds); $('#eta-time').textContent = '—'; };
const formValue = (id) => document.getElementById(id).value;
const formChecked = (id) => document.getElementById(id).checked ? 'true' : 'false';
const appendRtxSettings = (data) => { data.append('rtx_quality', formValue('rtx-quality')); };
const appendFaceSettings = (data) => { appendRtxSettings(data); data.append('face_model', formValue('face-model')); data.append('face_strength', formValue('face-strength')); data.append('face_fidelity', formValue('face-fidelity')); data.append('face_detail', formValue('face-detail')); data.append('face_min_size', formValue('face-min-size')); return data; };
const renderForm = (type, sourceFps = 0) => {
  const file = selectedSourceFile || $('#source-file').files[0];
  const sourcePath = formValue('source-path').trim();
  if (!file && !sourcePath) throw new Error('Choose an original video or enter its full path first.');
  const data = new FormData();
  if (file) data.append('file', file);
  data.append('source_path', sourcePath);
  return appendRenderSettings(data, type, sourceFps);
};
const appendRenderSettings = (data, type, sourceFps = 0) => {
  data.append('job_type', type); data.append('backend_name', formValue('backend'));
  data.append('preview_start', formValue('preview-start')); data.append('preview_seconds', formValue('preview-seconds')); data.append('output_preset', formValue('output-preset')); data.append('crop_policy', formValue('crop-policy'));
  data.append('chunked_render', type === 'full' && $('#chunked-render').checked ? 'true' : 'false'); data.append('chunk_seconds', formValue('chunk-seconds'));
  data.append('batch_size', formValue('batch-size')); data.append('seed', formValue('seed'));
  data.append('model_label', formValue('model')); data.append('color_correction', formValue('color')); data.append('attention_mode', formValue('attention')); data.append('blocks_to_swap', formValue('blocks'));
  data.append('vae_tiling', formChecked('vae-tiling')); data.append('stop_before_vae', 'false'); data.append('sharpen_enabled', formChecked('sharpen')); data.append('sharpen_strength', formValue('sharpen-strength')); data.append('microtexture_enabled', $('#skin-finishing').checked ? formChecked('microtexture') : 'false'); data.append('microtexture_strength', formValue('microtexture-strength')); data.append('grain_enabled', formChecked('grain')); data.append('grain_intensity', formValue('grain-intensity')); data.append('grain_saturation', formValue('grain-saturation'));
  data.append('skin_finishing_enabled', formChecked('skin-finishing')); data.append('skin_evenness', formValue('skin-evenness')); data.append('skin_smoothing', formValue('skin-smoothing')); data.append('skin_redness', formValue('skin-redness')); data.append('skin_shine', formValue('skin-shine')); data.append('blemish_mode', formValue('blemish-mode')); data.append('preserve_marks', formChecked('preserve-marks'));
  data.append('seam_mode', $('#seam-enabled').checked ? 'match' : 'off'); data.append('seam_frames', '2'); data.append('decoder_mode', 'optimized_fast');
  data.append('source_fps', String(sourceFps || 0)); data.append('source_scale', formValue('source-scale'));
  appendFaceSettings(data);
  return data;
};
const requestSourceFps = () => {
  while (true) {
    const value = window.prompt('Frame rate could not be detected. Please enter the source frame rate (for example 23.976, 29.97, 48, or 60):');
    if (value === null) return null;
    const parsed = Number(value);
    if (Number.isFinite(parsed) && parsed >= 1 && parsed <= 240) return parsed;
    window.alert('Enter a frame rate between 1 and 240. Decimals are allowed.');
  }
};

const setRenderBusy = (busy) => { ['render-preview','render-full','reprocess-post'].forEach((id) => { $( `#${id}` ).disabled = busy; }); $('#stop-render').hidden = !busy; if (busy) { $('#resume-render').hidden = true; lastFailedJob = null; } };
const pollJob = async (id, token = projectToken) => {
  const job = await (await fetch(`/api/jobs/${id}`)).json();
  if (token !== projectToken) return;
  window.__seedvrProgress = Number(job.progress) || 0;
  updateEta(window.__seedvrProgress);
  $('#render-status').textContent = `${job.message || job.status}${job.progress != null ? ` · ${Math.round(job.progress * 100)}%` : ''}`;
  if (job.status === 'complete') { stopElapsed(job.elapsed_seconds); setRenderBusy(false); activeJob = null; if (Number(job.fps) > 0) fps = Number(job.fps); const isSlot = job.preview_start != null; const keepOriginal = isSlot && hasVideo(before) && Math.abs(before.duration - Number(job.source_duration || 0)) < 0.5 && (!job.crop_policy || job.crop_policy === 'Preserve original aspect ratio'); if (job.original_url && !keepOriginal) loadVideo(before, job.original_url); if (job.restored_url) loadVideo(after, job.restored_url); setRestoredSlot(job.preview_start, keepOriginal); if (keepOriginal) { after.addEventListener('loadedmetadata', () => seek(restoredOffset), {once:true}); } currentOutputPath = job.output_relative || ''; currentOutputReprocessable = Boolean(job.reprocessable); $('#reprocess-post').hidden = !currentOutputReprocessable; $('#open-output-folder').hidden = !currentOutputPath; $('#media-info').textContent = job.job_type === 'reprocess' ? 'Post-only reprocess complete' : job.saved_path ? `Saved to ${job.saved_path}` : job.saved_error ? job.saved_error : `${job.job_type} complete`; loadOutputs(); return; }
  if (job.status === 'error' || job.status === 'cancelled') { stopElapsed(job.elapsed_seconds); setRenderBusy(false); activeJob = null; if (job.status === 'error' && job.failure_code === 'fps_required') { const enteredFps = requestSourceFps(); if (enteredFps !== null) startRender(job.job_type, enteredFps); else $('#render-status').textContent = 'Render cancelled · source frame rate is required'; return; } lastFailedJob = job.resumable ? job.id : null; $('#resume-render').hidden = !lastFailedJob; $('#render-status').textContent = `${job.failure_reason || job.error || job.message || job.status}${job.log_file ? ' · log saved' : ''}`; return; }
  setTimeout(() => pollJob(id, token).catch((error) => { if (token !== projectToken) return; setRenderBusy(false); $('#render-status').textContent = error.message; }), 800);
};
const startRender = async (type, sourceFps = 0) => { try { setRenderBusy(true); $('#render-status').textContent = `Uploading ${type}…`; const response = await fetch('/api/jobs', { method:'POST', body:renderForm(type, sourceFps) }); if (!response.ok) throw new Error(await response.text()); activeJob = (await response.json()).id; startElapsed(); pollJob(activeJob, projectToken); } catch (error) { stopElapsed(0); setRenderBusy(false); $('#render-status').textContent = error.message; } };
const startReprocess = async () => { try { if (!currentOutputPath || !currentOutputReprocessable) throw new Error('Load a TensorRT result with saved decoded batches first.'); const data = new FormData(); data.append('output_path', currentOutputPath); data.append('seed', formValue('seed')); data.append('sharpen_enabled', formChecked('sharpen')); data.append('sharpen_strength', formValue('sharpen-strength')); data.append('microtexture_enabled', $('#skin-finishing').checked ? formChecked('microtexture') : 'false'); data.append('microtexture_strength', formValue('microtexture-strength')); data.append('skin_finishing_enabled', formChecked('skin-finishing')); data.append('skin_evenness', formValue('skin-evenness')); data.append('skin_smoothing', formValue('skin-smoothing')); data.append('skin_redness', formValue('skin-redness')); data.append('skin_shine', formValue('skin-shine')); data.append('blemish_mode', formValue('blemish-mode')); data.append('preserve_marks', formChecked('preserve-marks')); data.append('grain_enabled', formChecked('grain')); data.append('grain_intensity', formValue('grain-intensity')); data.append('grain_saturation', formValue('grain-saturation')); data.append('seam_mode', $('#seam-enabled').checked ? 'match' : 'off'); data.append('seam_frames', '2'); data.append('color_correction', formValue('color')); appendFaceSettings(data); setRenderBusy(true); $('#render-status').textContent = 'Starting post-only reprocess…'; const response = await fetch('/api/reprocess', {method:'POST', body:data}); if (!response.ok) throw new Error(await response.text()); activeJob = (await response.json()).id; startElapsed(); pollJob(activeJob, projectToken); } catch (error) { stopElapsed(0); setRenderBusy(false); $('#render-status').textContent = error.message; } };
$('#render-preview').addEventListener('click', () => startRender('preview')); $('#render-full').addEventListener('click', () => startRender('full'));
$('#reprocess-post').addEventListener('click', startReprocess);
$('#resume-render').addEventListener('click', async () => { if (!lastFailedJob) return; try { $('#resume-render').hidden = true; setRenderBusy(true); const response = await fetch(`/api/jobs/${lastFailedJob}/resume`, {method:'POST'}); if (!response.ok) throw new Error(await response.text()); activeJob = lastFailedJob; startElapsed(); pollJob(activeJob, projectToken); } catch (error) { setRenderBusy(false); $('#render-status').textContent = error.message; } });
$('#stop-render').addEventListener('click', async () => { if (!activeJob) return; await fetch(`/api/jobs/${activeJob}/cancel`, {method:'POST'}); });
$('#blocks').addEventListener('input', (event) => { $('#blocks-value').textContent = event.target.value; });
const bindSliderValue = (inputId, outputId, digits) => { const input = $(`#${inputId}`); const output = $(`#${outputId}`); const refresh = () => { output.textContent = Number(input.value).toFixed(digits); }; input.addEventListener('input', refresh); refresh(); };
[['face-strength','face-strength-value',2],['face-fidelity','face-fidelity-value',2],['face-detail','face-detail-value',2],['face-min-size','face-min-size-value',0]].forEach(([input, output, digits]) => bindSliderValue(input, output, digits));
bindSliderValue('sharpen-strength', 'sharpen-strength-value', 2);
bindSliderValue('microtexture-strength', 'microtexture-strength-value', 2);
bindSliderValue('skin-evenness', 'skin-evenness-value', 2);
bindSliderValue('skin-smoothing', 'skin-smoothing-value', 2);
bindSliderValue('skin-redness', 'skin-redness-value', 2);
bindSliderValue('skin-shine', 'skin-shine-value', 2);
bindSliderValue('grain-intensity', 'grain-intensity-value', 3);
bindSliderValue('grain-saturation', 'grain-saturation-value', 2);
const fillSelect = (id, values) => { const select = $(`#${id}`); select.innerHTML = ''; values.forEach((value) => { const option = document.createElement('option'); option.value = value; option.textContent = value; select.append(option); }); };
const defaultBatchSizes = {'SeedVR2 + TensorRT': [5, 21], 'SeedVR2 (Legacy)': [1, 5, 9, 13, 17, 21, 33, 45], 'RTX Video Super Resolution': [21]};
const refreshEngineUi = () => { const rtx = $('#backend').value === 'RTX Video Super Resolution'; $('#rtx-quality-wrap').hidden = !rtx; $('#batch-note').textContent = rtx ? 'RTX VSR needs no SeedVR2 model, temporal batch or attention settings.' : ($('#backend').value === 'SeedVR2 + TensorRT' ? 'TensorRT supports temporal batches 5 and 21.' : 'Legacy path accepts all 4n+1 batch sizes.'); ['model','batch-size','attention','blocks','vae-tiling','preset'].forEach((id) => { const element = document.getElementById(id); if (element) element.disabled = rtx; }); };
const updateBatchOptions = (backend, preferred = null) => { const values = (window.__seedvrBatchSizes?.[backend] || defaultBatchSizes[backend] || defaultBatchSizes['SeedVR2 (Legacy)']).map(String); const select = $('#batch-size'); const current = preferred == null ? select.value : String(preferred); select.innerHTML = ''; values.forEach((value) => { const option = document.createElement('option'); option.value = value; option.textContent = value; select.append(option); }); select.value = values.includes(current) ? current : (values.includes('21') ? '21' : values[0]); $('#batch-note').textContent = backend === 'SeedVR2 + TensorRT' ? 'TensorRT supports temporal batches 5 and 21.' : 'SeedVR2 (Legacy) supports 4n+1 temporal batch sizes.'; };
$('#backend').addEventListener('change', () => { updateBatchOptions($('#backend').value); refreshSettingsUi(); refreshEngineUi(); });
const setSkinBackendAvailable = (available) => { $('#skin-backend-warning').hidden = available; $('#skin-finishing').disabled = !available; if (!available) $('#skin-finishing').checked = false; refreshSettingsUi(); };
// ---------- Batch mode ----------
const batchList = $('#batch-list'), batchSummary = $('#batch-summary'), batchStatus = $('#batch-status');
let batchSnapshot = null, batchTimer = null;
const batchSettingsJson = () => { const data = appendRenderSettings(new FormData(), 'full', 0); const object = {}; data.forEach((value, key) => { object[key] = value; }); return JSON.stringify(object); };
const setBatchStatus = (text, kind = '') => { batchStatus.textContent = text; batchStatus.className = `file-name ${kind}`.trim(); };
const batchCall = async (url, body) => { const response = await fetch(url, {method:'POST', body}); const payload = await response.json().catch(() => ({})); if (!response.ok) throw new Error(payload.detail || `${url} failed (${response.status})`); return payload; };
const renderBatch = (snapshot) => {
  batchSnapshot = snapshot;
  const items = snapshot.items || [];
  const counts = snapshot.counts || {};
  const running = snapshot.state !== 'idle';
  batchSummary.textContent = items.length ? `${items.length} in queue · ${counts.queued || 0} waiting · ${counts.done || 0} done${counts.failed ? ` · ${counts.failed} failed` : ''}${counts.skipped ? ` · ${counts.skipped} skipped` : ''}${snapshot.state === 'pausing' ? ' · pausing after current' : running ? ' · running' : ''}` : 'Queue empty';
  $('#batch-start').hidden = running; $('#batch-start').disabled = !(counts.queued > 0);
  $('#batch-pause').hidden = !running || snapshot.state === 'pausing'; $('#batch-stop').hidden = !running; $('#batch-skip').hidden = !running || !snapshot.current;
  $('#batch-clear').hidden = !items.some((item) => ['done','failed','skipped','cancelled'].includes(item.status));
  batchList.innerHTML = '';
  items.forEach((item) => {
    const row = document.createElement('div'); row.className = `batch-item ${item.status}`; row.title = item.path;
    const name = document.createElement('span'); name.className = 'batch-name'; name.textContent = item.name;
    const state = document.createElement('span'); state.className = 'batch-state'; state.textContent = item.status === 'running' ? `${Math.round((item.progress || 0) * 100)}%` : item.status;
    row.append(name, state);
    if (item.message && item.status !== 'queued') { const message = document.createElement('span'); message.className = 'batch-message'; message.textContent = item.message; row.append(message); }
    if (item.status === 'running') { const bar = document.createElement('div'); bar.className = 'batch-bar'; const fill = document.createElement('i'); fill.style.width = `${Math.round((item.progress || 0) * 100)}%`; bar.append(fill); row.append(bar); }
    const actions = document.createElement('div'); actions.className = 'batch-actions';
    const button = (label, handler, title = '') => { const element = document.createElement('button'); element.type = 'button'; element.textContent = label; element.title = title; element.addEventListener('click', (event) => { event.stopPropagation(); handler().catch((error) => setBatchStatus(error.message, 'error')); }); actions.append(element); };
    if (item.status === 'queued') { button('↑', () => batchCall(`/api/batch/items/${item.id}/move`, new URLSearchParams({direction:'-1'})).then(renderBatch), 'Move up'); button('↓', () => batchCall(`/api/batch/items/${item.id}/move`, new URLSearchParams({direction:'1'})).then(renderBatch), 'Move down'); }
    if (item.status !== 'running') button('Remove', () => batchCall(`/api/batch/items/${item.id}/remove`).then(renderBatch));
    if (['failed','skipped','cancelled','done'].includes(item.status)) button('Retry', () => batchCall(`/api/batch/items/${item.id}/retry`).then(renderBatch), item.status === 'skipped' ? 'Render anyway (writes name-seed-2.mp4)' : 'Queue again');
    if (actions.childElementCount) row.append(actions);
    if (item.status === 'done' && item.output) row.addEventListener('click', () => loadBatchResult(item));
    batchList.append(row);
  });
  if (running && !batchTimer) batchTimer = setTimeout(() => { batchTimer = null; refreshBatch(); }, 1000);
  else if (!running && batchTimer) { clearTimeout(batchTimer); batchTimer = null; }
};
const refreshBatch = async () => { try { const response = await fetch('/api/batch', {cache:'no-store'}); if (!response.ok) return; renderBatch(await response.json()); } catch (_) {} };
const loadBatchResult = async (item) => {
  sourcePathInput.value = item.path; await loadSourcePath(true);
  setRestoredSlot(0, false); loadVideo(after, '/api/local-video?path=' + encodeURIComponent(item.output));
  currentOutputPath = ''; currentOutputReprocessable = false; $('#reprocess-post').hidden = true; $('#open-output-folder').hidden = true;
  $('#media-info').textContent = `Batch result · ${item.output}`;
};
const addToBatch = async (paths) => {
  if (!paths.length) return;
  setBatchStatus(`Adding ${paths.length} path${paths.length === 1 ? '' : 's'}…`);
  const data = new FormData(); data.append('paths', JSON.stringify(paths)); data.append('settings', batchSettingsJson());
  const result = await batchCall('/api/batch/add', data);
  renderBatch(result);
  const notes = [];
  if (result.added) notes.push(`${result.added} queued`);
  if (result.skipped?.length) notes.push(`${result.skipped.length} skipped (already rendered)`);
  if (result.duplicates?.length) notes.push(`${result.duplicates.length} already in queue`);
  if (result.missing?.length) notes.push(`not found: ${result.missing.join(', ')}`);
  setBatchStatus(notes.join(' · ') || 'No videos found', result.missing?.length ? 'error' : 'ok');
};
$('#batch-add-folder').addEventListener('click', async () => { try { setBatchStatus('Choose a folder in the dialog…'); const data = new FormData(); data.append('initial', sourcePathInput.value.trim()); const result = await batchCall('/api/pick-folder', data); if (result.cancelled || !result.path) { setBatchStatus('Dialog cancelled'); return; } await addToBatch([result.path]); } catch (error) { setBatchStatus(error.message, 'error'); } });
$('#batch-add-files').addEventListener('click', async () => { try { setBatchStatus('Choose videos in the dialog…'); const data = new FormData(); data.append('initial', sourcePathInput.value.trim()); const result = await batchCall('/api/pick-files', data); if (result.cancelled || !result.paths?.length) { setBatchStatus('Dialog cancelled'); return; } await addToBatch(result.paths); } catch (error) { setBatchStatus(error.message, 'error'); } });
$('#batch-start').addEventListener('click', async () => { try { renderBatch(await batchCall('/api/batch/start')); setBatchStatus('Batch running'); } catch (error) { setBatchStatus(error.message, 'error'); } });
$('#batch-pause').addEventListener('click', async () => { try { renderBatch(await batchCall('/api/batch/pause')); setBatchStatus('Finishing the current video, then pausing'); } catch (error) { setBatchStatus(error.message, 'error'); } });
$('#batch-stop').addEventListener('click', async () => { try { renderBatch(await batchCall('/api/batch/stop')); setBatchStatus('Batch stopped'); } catch (error) { setBatchStatus(error.message, 'error'); } });
$('#batch-skip').addEventListener('click', async () => { try { renderBatch(await batchCall('/api/batch/skip-current')); setBatchStatus('Skipping current video'); } catch (error) { setBatchStatus(error.message, 'error'); } });
$('#batch-clear').addEventListener('click', async () => { try { renderBatch(await batchCall('/api/batch/clear-finished')); setBatchStatus(''); } catch (error) { setBatchStatus(error.message, 'error'); } });
refreshBatch();

const loadConfig = async () => { try { const config = await (await fetch('/api/config')).json(); window.__seedvrBatchSizes = config.batch_sizes || defaultBatchSizes; fillSelect('model', config.models); fillSelect('preset', Object.keys(config.presets)); fillSelect('output-preset', Object.keys(config.output_presets)); window.__seedvrOutputPresets = config.output_presets; $('#output-preset').addEventListener('change', updateOutputSizeNote); $('#source-scale').addEventListener('change', updateSourceScaleNote); fillSelect('crop-policy', config.crop_policies); const saved = loadSavedSettings(); const savedBackend = String(saved?.values?.backend || 'SeedVR2 + TensorRT'); const backendAliases = {'TensorRT VAE (experimental)':'SeedVR2 + TensorRT', 'SeedVR2 AI':'SeedVR2 (Legacy)'}; const requestedBackend = backendAliases[savedBackend] || savedBackend; $('#backend').value = Array.from($('#backend').options).some((option) => option.value === requestedBackend) ? requestedBackend : 'SeedVR2 + TensorRT'; updateBatchOptions($('#backend').value); $('#output-preset').value = '1K / 1080p'; $('#crop-policy').value = 'Preserve original aspect ratio'; if (config.seam_modes) { $('#seam-enabled').disabled = false; $('#seam-enabled').title = 'Color + noise match over 2 frames'; } else { $('#seam-enabled').title = 'Available after the JS backend is restarted'; } const preset = $('#preset'); preset.addEventListener('change', () => { const value = config.presets[preset.value]; if (!value) return; ['model','batch-size','attention','blocks'].forEach((id) => { const key = {'batch-size':'batch_size','model':'model','attention':'attention','blocks':'blocks'}[id]; if (value[key] != null) $(`#${id}`).value = value[key]; }); updateBatchOptions($('#backend').value, value.batch_size); $('#vae-tiling').checked = Boolean(value.vae_tiling); $('#blocks-value').textContent = $('#blocks').value; }); restoreSettings(); updateBatchOptions($('#backend').value, $('#batch-size').value); setSkinBackendAvailable(config.features?.skin_finishing === true); $('#browse-source').hidden = config.features?.file_dialog === false; const backend = config.backend; $('#api-status').textContent = backend.seedvr?.message || 'API ready'; } catch (_) { $('#backend').value = 'SeedVR2 + TensorRT'; updateBatchOptions($('#backend').value); restoreSettings(); if (!$('#backend').value) $('#backend').value = 'SeedVR2 + TensorRT'; updateBatchOptions($('#backend').value, $('#batch-size').value); setSkinBackendAvailable(false); $('#api-status').textContent = 'API ready · config unavailable'; } };loadConfig();



$('#engine-help').addEventListener('click', () => $('#engine-help-dialog').showModal());

const updateDialog = document.createElement('dialog');
updateDialog.id = 'update-dialog';
updateDialog.innerHTML = '<form method="dialog"><button class="dialog-close" aria-label="Close">&times;</button><h2>SeedVR Studio updates</h2><p id="update-message">Checking for updates...</p><p id="update-version"></p><div class="render-actions"><button class="secondary-button" value="close">Close</button><button class="primary-button" id="apply-update" type="button" hidden>Update and restart</button></div></form>';
document.body.append(updateDialog);
const updateButton = document.createElement('button');
updateButton.className = 'icon-button';
updateButton.id = 'check-update';
updateButton.title = 'Check for updates';
updateButton.setAttribute('aria-label', 'Check for updates');
updateButton.innerHTML = '&#8635;';
$('#exit-studio').before(updateButton);
const applyUpdateButton = $('#apply-update');
const updateMessage = $('#update-message');
const updateVersion = $('#update-version');

updateButton.addEventListener('click', async () => {
  updateButton.disabled = true;
  updateMessage.textContent = 'Checking for updates...';
  updateVersion.textContent = '';
  applyUpdateButton.hidden = true;
  updateDialog.showModal();
  try {
    const response = await fetch('/api/update/check', {cache:'no-store'});
    const status = await response.json();
    if (!response.ok) throw new Error(status.detail || 'Update check failed.');
    updateMessage.textContent = status.message;
    const versions = [];
    if (status.current_version) versions.push(`Installed: ${status.current_version}`);
    if (status.latest_version) versions.push(`Latest: ${status.latest_version}`);
    updateVersion.textContent = versions.join(' / ');
    applyUpdateButton.hidden = !(status.supported && status.update_available);
  } catch (error) {
    updateMessage.textContent = `Update check failed: ${error.message}`;
  } finally {
    updateButton.disabled = false;
  }
});

applyUpdateButton.addEventListener('click', async () => {
  if (!window.confirm('Update and restart SeedVR Studio now? Active renders must be stopped first.')) return;
  applyUpdateButton.disabled = true;
  updateMessage.textContent = 'Starting the safe updater...';
  try {
    const response = await fetch('/api/update/apply', {method:'POST'});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Update could not start.');
    updateMessage.textContent = payload.message;
    applyUpdateButton.hidden = true;
    setTimeout(() => window.close(), 900);
  } catch (error) {
    updateMessage.textContent = `Update failed: ${error.message}`;
    applyUpdateButton.disabled = false;
  }
});
