/* SynthBench Studio v3 — dependency-free interactive calibration bench. */
(() => {
  'use strict';

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const fmt = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';
  const pct = (value, digits = 1) => Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—';
  const mean = values => values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
  const median = values => {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : 0.5 * (sorted[middle - 1] + sorted[middle]);
  };
  const std = values => {
    const avg = mean(values);
    return Math.sqrt(mean(values.map(value => (value - avg) ** 2)));
  };
  const css = getComputedStyle(document.documentElement);
  const color = name => css.getPropertyValue(name).trim();
  const palette = {
    blue: color('--blue') || '#0071e3',
    teal: color('--teal') || '#2b9c94',
    coral: color('--coral') || '#e66a4e',
    violet: color('--violet') || '#7b6bb0',
    amber: color('--amber') || '#c77d1a',
    ink: color('--ink') || '#1d1d1f',
    muted: color('--muted') || '#6e6e73',
  };

  const app = { bundle: null, dsp: null, decoder: null, eval: null, plan: null };
  const motion = {
    active: new Set(),
    enteredAt: new Map(),
    drawers: new Map(),
    reduce: Boolean(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches),
  };

  function registerAmbientDrawer(key, draw) {
    motion.drawers.set(key, draw);
  }

  function sectionProgress(key, duration = 1450) {
    if (motion.reduce) return 1;
    const enteredAt = motion.enteredAt.get(key) || performance.now();
    const raw = clamp((performance.now() - enteredAt) / duration, 0, 1);
    return 1 - Math.pow(1 - raw, 3);
  }

  function sectionIsActive(key) {
    return motion.active.has(key);
  }

  function installScrollStory() {
    const targets = $$('[data-reveal]');
    if (motion.reduce || !('IntersectionObserver' in window)) {
      targets.forEach(target => {
        target.classList.add('is-inview');
        motion.active.add(target.dataset.reveal);
        motion.enteredAt.set(target.dataset.reveal, performance.now());
      });
      return;
    }
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        const key = entry.target.dataset.reveal;
        const visible = entry.isIntersecting;
        entry.target.classList.toggle('is-inview', visible);
        if (visible) {
          if (!motion.active.has(key)) motion.enteredAt.set(key, performance.now());
          motion.active.add(key);
        } else {
          motion.active.delete(key);
        }
      });
    }, { rootMargin: '-6% 0px -10% 0px', threshold: [0.04, 0.12, 0.28] });
    targets.forEach(target => observer.observe(target));
  }

  function startAmbientLoop() {
    if (motion.reduce) {
      motion.drawers.forEach(draw => draw(performance.now(), 1));
      return;
    }
    let last = 0;
    const step = now => {
      if (now - last >= 30) {
        last = now;
        motion.drawers.forEach((draw, key) => {
          if (sectionIsActive(key)) draw(now, sectionProgress(key));
        });
      }
      requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function setupCanvas(canvas) {
    if (!canvas) return null;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(280, canvas.clientWidth || canvas.parentElement?.clientWidth || 700);
    const baseHeight = Number(canvas.dataset.baseHeight || canvas.getAttribute('height') || 220);
    canvas.dataset.baseHeight = String(baseHeight);
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(baseHeight * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(baseHeight * dpr);
    }
    canvas.style.height = `${baseHeight}px`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, baseHeight);
    return { ctx, width, height: baseHeight };
  }

  function roundedRect(ctx, x, y, width, height, radius = 8) {
    const r = Math.min(radius, Math.abs(width) / 2, Math.abs(height) / 2);
    ctx.beginPath();
    ctx.roundRect(x, y, width, height, r);
  }

  function drawPlotFrame(ctx, width, height, left = 42, right = 18, top = 18, bottom = 26) {
    ctx.strokeStyle = 'rgba(29,29,31,.095)';
    ctx.lineWidth = 1;
    ctx.strokeRect(left + 0.5, top + 0.5, width - left - right, height - top - bottom);
  }

  function setActiveSegment(groupSelector, attr, value) {
    $$(`${groupSelector} .seg-btn`).forEach(button => {
      button.classList.toggle('is-active', button.dataset[attr] === value);
    });
  }

  function statCards(container, items) {
    if (!container) return;
    container.innerHTML = items.map(item => `
      <div class="stat-card">
        <div class="stat-card__k">${item.label}</div>
        <div class="stat-card__v ${item.accent ? `stat-card__v--${item.accent}` : ''}">${item.value}</div>
      </div>`).join('');
  }

  function nextPowerOfTwo(value) {
    let n = 1;
    while (n < value) n <<= 1;
    return n;
  }

  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i += 1) {
      let bit = n >> 1;
      while (j & bit) { j ^= bit; bit >>= 1; }
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let length = 2; length <= n; length <<= 1) {
      const angle = -2 * Math.PI / length;
      const wRe = Math.cos(angle);
      const wIm = Math.sin(angle);
      const half = length >> 1;
      for (let start = 0; start < n; start += length) {
        let currentRe = 1;
        let currentIm = 0;
        for (let offset = 0; offset < half; offset += 1) {
          const even = start + offset;
          const odd = even + half;
          const oddRe = re[odd] * currentRe - im[odd] * currentIm;
          const oddIm = re[odd] * currentIm + im[odd] * currentRe;
          re[odd] = re[even] - oddRe;
          im[odd] = im[even] - oddIm;
          re[even] += oddRe;
          im[even] += oddIm;
          const nextRe = currentRe * wRe - currentIm * wIm;
          currentIm = currentRe * wIm + currentIm * wRe;
          currentRe = nextRe;
        }
      }
    }
  }

  function ifft(re, im) {
    const n = re.length;
    const imag = im.map(value => -value);
    fft(re, imag);
    for (let i = 0; i < n; i += 1) { re[i] /= n; im[i] = -imag[i] / n; }
  }

  function hannWindow(length) {
    if (length <= 1) return new Float64Array([1]);
    return Float64Array.from({ length }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (length - 1)));
  }

  function bandpassFilter(signal, sampleRate, lowHz = 300, highHz = 3400) {
    const n = signal.length;
    if (n < 4 || sampleRate <= 0) return Float64Array.from(signal);
    const size = nextPowerOfTwo(n);
    const re = new Float64Array(size);
    const im = new Float64Array(size);
    re.set(signal);
    fft(re, im);
    const nyquist = sampleRate / 2;
    const effectiveLow = Math.max(0, Math.min(lowHz, nyquist * 0.8));
    const effectiveHigh = Math.min(highHz, nyquist * 0.98);
    if (effectiveLow >= effectiveHigh) return Float64Array.from(signal);
    const taper = Math.max(25, 0.05 * (effectiveHigh - effectiveLow));
    for (let bin = 1; bin < size / 2; bin += 1) {
      const freq = bin * sampleRate / size;
      let mask = 0;
      if (freq >= effectiveLow && freq <= effectiveHigh) mask = 1;
      else if (freq >= effectiveLow - taper && freq < effectiveLow) mask = 0.5 * (1 + Math.cos(Math.PI * (effectiveLow - freq) / taper));
      else if (freq > effectiveHigh && freq <= effectiveHigh + taper) mask = 0.5 * (1 + Math.cos(Math.PI * (freq - effectiveHigh) / taper));
      re[bin] *= mask; im[bin] *= mask;
      const mirror = size - bin;
      re[mirror] *= mask; im[mirror] *= mask;
    }
    ifft(re, im);
    return re.slice(0, n);
  }

  function acousticFeatures(signal, sampleRate) {
    const audio = Float64Array.from(signal);
    const filtered = bandpassFilter(audio, sampleRate, 300, 3400);
    const hopSeconds = 0.010;
    const windowSeconds = 0.025;
    const hop = Math.max(1, Math.round(sampleRate * hopSeconds));
    const windowLength = Math.max(2, Math.round(sampleRate * windowSeconds));
    const count = 1 + Math.max(0, Math.floor((audio.length - windowLength) / hop));
    const weights = hannWindow(windowLength);
    const normFactor = Math.max(weights.reduce((sum, value) => sum + value * value, 0), 1e-12);
    const fftSize = nextPowerOfTwo(windowLength);
    const nyquist = sampleRate / 2;
    const freqResolution = sampleRate / fftSize;
    const highLow = Math.min(2000, nyquist * 0.7);
    const highHigh = Math.min(4000, nyquist);
    const parsevalWeights = new Float64Array(fftSize / 2 + 1).fill(2);
    parsevalWeights[0] = 1;
    if (fftSize % 2 === 0) parsevalWeights[parsevalWeights.length - 1] = 1;
    const energyDb = new Float64Array(count);
    const highBandDb = new Float64Array(count);
    const fusedDb = new Float64Array(count);
    const flux = new Float64Array(count);
    let previousSpectrum = null;
    for (let frame = 0; frame < count; frame += 1) {
      const start = frame * hop;
      let energy = 0;
      const rawRe = new Float64Array(fftSize);
      const rawIm = new Float64Array(fftSize);
      for (let i = 0; i < windowLength && start + i < audio.length; i += 1) {
        const windowValue = weights[i];
        const filteredValue = filtered[start + i] * windowValue;
        const rawValue = audio[start + i] * windowValue;
        energy += filteredValue * filteredValue;
        rawRe[i] = rawValue;
      }
      const rms = Math.sqrt(energy / normFactor + 1e-12);
      energyDb[frame] = 20 * Math.log10(rms);
      fft(rawRe, rawIm);
      const magnitudes = new Float64Array(fftSize / 2 + 1);
      let magNorm = 0;
      let highPower = 0;
      for (let bin = 0; bin <= fftSize / 2; bin += 1) {
        const magnitude = Math.hypot(rawRe[bin], rawIm[bin]);
        magnitudes[bin] = magnitude;
        magNorm += magnitude * magnitude;
        const freq = bin * freqResolution;
        if (freq >= highLow && freq <= highHigh) highPower += magnitude * magnitude * parsevalWeights[bin];
      }
      highBandDb[frame] = 10 * Math.log10(highPower / (fftSize * normFactor) + 1e-12);
      fusedDb[frame] = Math.max(energyDb[frame], highBandDb[frame] - 6);
      magNorm = Math.sqrt(magNorm) + 1e-12;
      let spectralFlux = 0;
      for (let bin = 0; bin <= fftSize / 2; bin += 1) {
        const current = magnitudes[bin] / magNorm;
        const previous = previousSpectrum ? previousSpectrum[bin] : current;
        spectralFlux += Math.max(current - previous, 0);
      }
      flux[frame] = spectralFlux;
      previousSpectrum = Float64Array.from(magnitudes, value => value / magNorm);
    }
    return { audio, filtered, energyDb, highBandDb, fusedDb, flux, hop, hopSeconds, windowLength, windowSeconds: windowLength / sampleRate, duration: audio.length / sampleRate, sampleRate, count };
  }

  function localMinMax(values) {
    let low = Infinity, high = -Infinity;
    for (const value of values) { if (value < low) low = value; if (value > high) high = value; }
    const span = high - low;
    return span < 1e-12 ? new Float64Array(values.length) : values.map(value => (value - low) / span);
  }

  function percentile(values, fraction) {
    const sorted = Array.from(values).sort((a, b) => a - b);
    if (!sorted.length) return 0;
    const pos = clamp(fraction, 0, 1) * (sorted.length - 1);
    const low = Math.floor(pos), high = Math.ceil(pos);
    return sorted[low] + (sorted[high] - sorted[low]) * (pos - low);
  }

  function timeToFrame(time, hopSeconds, halfWindow, count) {
    return clamp(Math.round((time - halfWindow) / hopSeconds), 0, count - 1);
  }
  function frameToTime(frame, hopSeconds, halfWindow, duration) {
    return clamp(frame * hopSeconds + halfWindow, 0, duration);
  }
  function refineEdge(features, anchorTime, direction) {
    const { fusedDb, hopSeconds, windowSeconds, count } = features;
    const halfWindow = windowSeconds / 2;
    const anchor = timeToFrame(anchorTime, hopSeconds, halfWindow, count);
    const radius = Math.max(1, Math.round(0.12 / hopSeconds));
    const start = Math.max(0, anchor - radius);
    const end = Math.min(count, anchor + radius + 1);
    const blockMax = Math.max(...fusedDb.slice(start, end));
    const threshold = Math.max(blockMax - 18, percentile(fusedDb, 0.05) + 3);
    const limit = Math.max(1, Math.round(0.24 / hopSeconds));
    const step = direction === 'left' ? -1 : 1;
    let frame = anchor, lastActive = anchor, misses = 0;
    while (Math.abs(frame - anchor) < limit) {
      const candidate = frame + step;
      if (candidate < 0 || candidate >= count) break;
      if (fusedDb[candidate] < threshold) { misses += 1; if (misses >= 2) break; }
      else { misses = 0; lastActive = candidate; }
      frame = candidate;
    }
    return lastActive;
  }
  function resamplePosteriors(posteriors, targetCount, sourceHop, targetHop, halfWindow, frameCenter) {
    if (!posteriors?.length) return null;
    const classes = posteriors.length, sourceCount = posteriors[0].length;
    const result = Array.from({ length: classes }, () => new Float64Array(targetCount));
    for (let frame = 0; frame < targetCount; frame += 1) {
      const time = frameCenter ? frame * targetHop + halfWindow : frame * targetHop;
      const sourcePosition = time / sourceHop;
      const lower = clamp(Math.floor(sourcePosition), 0, sourceCount - 1);
      const upper = clamp(lower + 1, 0, sourceCount - 1);
      const ratio = clamp(sourcePosition - lower, 0, 1);
      for (let cls = 0; cls < classes; cls += 1) result[cls][frame] = posteriors[cls][lower] * (1 - ratio) + posteriors[cls][upper] * ratio;
    }
    return result;
  }
  function decodeDspState(signal, sampleRate, characters, anchors, options = {}) {
    const features = acousticFeatures(signal, sampleRate);
    const halfWindow = features.windowSeconds / 2;
    const left = refineEdge(features, anchors[0], 'left');
    const right = Math.max(left, refineEdge(features, anchors[anchors.length - 1], 'right'));
    const wordEnergy = features.fusedDb.slice(left, right + 1);
    const wordFlux = features.flux.slice(left, right + 1);
    const energyNorm = options.localNorm ? localMinMax(wordEnergy) : localMinMax(features.fusedDb);
    const fluxNorm = options.localNorm ? localMinMax(wordFlux) : localMinMax(features.flux);
    const offset = options.localNorm ? 0 : left;
    const cost = new Float64Array(features.count).fill(Number.NaN);
    for (let frame = left; frame <= right; frame += 1) cost[frame] = energyNorm[frame - offset] - 0.25 * fluxNorm[frame - offset];
    let post = null;
    if (options.posteriors?.length) post = resamplePosteriors(options.posteriors, features.count, options.posteriorHop || 0.02, features.hopSeconds, halfWindow, options.frameCenter !== false);
    const boundaries = [left];
    for (let index = 0; index < anchors.length - 1; index += 1) {
      const low = Math.max(left, timeToFrame(anchors[index], features.hopSeconds, halfWindow, features.count));
      const high = Math.min(right, timeToFrame(anchors[index + 1], features.hopSeconds, halfWindow, features.count));
      if (high > low) {
        let bestFrame = low, bestCost = Infinity;
        for (let frame = low; frame <= high; frame += 1) {
          let candidate = cost[frame];
          if (post && post.length >= anchors.length) candidate = 0.5 * candidate + 0.5 * Math.abs(post[index][frame] - post[index + 1][frame]);
          if (candidate < bestCost) { bestCost = candidate; bestFrame = frame; }
        }
        boundaries.push(bestFrame);
      } else boundaries.push(Math.round((low + high) / 2));
    }
    boundaries.push(right);
    boundaries.sort((a, b) => a - b);
    const spans = characters.split('').map((character, index) => ({
      character,
      start: frameToTime(boundaries[index], features.hopSeconds, halfWindow, features.duration),
      end: frameToTime(boundaries[index + 1], features.hopSeconds, halfWindow, features.duration),
    }));
    return { features, left, right, boundaries, cost, post, spans, halfWindow };
  }

  // ---------------------------------------------------------------------------
  // Module 1: energy-valley microscope
  // ---------------------------------------------------------------------------
  function initDsp(bundle) {
    const state = {
      presetId: 'shaoshui', characters: '烧水', anchors: [0.33, 0.64], posteriors: null,
      posteriorHop: 0.02, audioBuffer: null, signal: null, sampleRate: 8000, decoded: null,
      localNorm: true, frameCenter: true, dragIndex: -1, playStart: 0, playRequest: 0,
      ambientTime: null, ambientStart: performance.now(), ambientClock: performance.now(), manualUntil: 0, lastPresetCycle: 0,
    };
    app.dsp = state;
    const waveCanvas = $('#canvas-wave'), bandsCanvas = $('#canvas-bands'), valleyCanvas = $('#canvas-valley');
    const charsInput = $('#dsp-chars-input'), toggleLocal = $('#toggle-local-norm'), toggleCenter = $('#toggle-frame-center');
    const chipLocal = $('#chip-local-norm'), chipCenter = $('#chip-frame-center');

    function loadPreset(id) {
      const preset = bundle.dsp_presets.find(item => item.id === id) || bundle.dsp_presets[0];
      state.presetId = preset.id;
      state.characters = preset.characters;
      state.anchors = [...preset.anchor_times];
      state.posteriors = preset.posteriors;
      state.posteriorHop = preset.posterior_hop;
      state.signal = Float32Array.from(preset.signal);
      state.sampleRate = preset.sample_rate;
      state.audioBuffer = null;
      state.decoded = null;
      charsInput.value = preset.characters;
      setActiveSegment('#dsp-preset-group', 'preset', preset.id);
      refreshDsp(true);
    }

    function loadSignal(signal, sampleRate, characters = state.characters) {
      const duration = signal.length / sampleRate;
      const text = charsInput.value.trim() || characters || '词';
      const count = Math.max(1, text.length);
      state.signal = Float32Array.from(signal);
      state.sampleRate = sampleRate;
      state.characters = text;
      state.posteriors = null;
      state.anchors = Array.from({ length: count }, (_, index) => duration * (0.28 + 0.44 * index / Math.max(count - 1, 1)));
      state.audioBuffer = null;
      state.decoded = null;
      refreshDsp(true);
    }

    function refreshDsp(reset = false) {
      if (!state.signal?.length) return;
      const chars = charsInput.value.trim() || state.characters;
      if (chars.length !== state.anchors.length) {
        state.characters = chars;
        const duration = state.signal.length / state.sampleRate;
        state.anchors = Array.from({ length: Math.max(1, chars.length) }, (_, index) => duration * (0.28 + 0.44 * index / Math.max(chars.length - 1, 1)));
      }
      const duration = state.signal.length / state.sampleRate;
      state.anchors = state.anchors.map((time, index) => {
        const minimum = index === 0 ? 0.01 : state.anchors[index - 1] + 0.008;
        return clamp(time, minimum, duration - 0.01);
      });
      state.decoded = decodeDspState(state.signal, state.sampleRate, chars, state.anchors, { localNorm: state.localNorm, frameCenter: state.frameCenter, posteriors: state.posteriors, posteriorHop: state.posteriorHop });
      if (reset) state.audioBuffer = null;
      renderDsp();
    }

    function plotMetrics(width, height, duration, margins = [46, 18]) {
      return {
        x: time => margins[0] + time / duration * (width - margins[0] - margins[1]),
        time: x => clamp((x - margins[0]) / (width - margins[0] - margins[1]) * duration, 0, duration),
        left: margins[0], right: width - margins[1], top: 18, bottom: height - 24,
      };
    }

    function renderWave(canvas, decoded) {
      const surface = setupCanvas(canvas);
      if (!surface) return;
      const { ctx, width, height } = surface;
      const { audio } = decoded.features;
      const duration = decoded.features.duration;
      const plot = plotMetrics(width, height, duration);
      drawPlotFrame(ctx, width, height, plot.left, width - plot.right, plot.top, height - plot.bottom);
      const center = (plot.top + plot.bottom) / 2;
      const amplitude = (plot.bottom - plot.top) * 0.44;
      ctx.strokeStyle = 'rgba(0,113,227,.16)';
      ctx.beginPath(); ctx.moveTo(plot.left, center); ctx.lineTo(plot.right, center); ctx.stroke();
      const columns = Math.max(1, Math.floor(plot.right - plot.left));
      ctx.strokeStyle = 'rgba(0,113,227,.72)'; ctx.lineWidth = 1; ctx.beginPath();
      for (let column = 0; column < columns; column += 1) {
        const from = Math.floor(column / columns * audio.length);
        const to = Math.max(from + 1, Math.floor((column + 1) / columns * audio.length));
        let low = 1, high = -1;
        for (let i = from; i < to; i += 1) { low = Math.min(low, audio[i]); high = Math.max(high, audio[i]); }
        const x = plot.left + column;
        const ripple = 1 + 0.035 * Math.sin(state.ambientClock / 820 + column / columns * Math.PI * 8);
        ctx.moveTo(x, center - high * ripple * amplitude); ctx.lineTo(x, center - low * ripple * amplitude);
      }
      ctx.stroke();
      const wordStart = frameToTime(decoded.left, decoded.features.hopSeconds, decoded.halfWindow, duration);
      const wordEnd = frameToTime(decoded.right, decoded.features.hopSeconds, decoded.halfWindow, duration);
      ctx.fillStyle = 'rgba(43,156,148,.08)';
      ctx.fillRect(plot.x(wordStart), plot.top, plot.x(wordEnd) - plot.x(wordStart), plot.bottom - plot.top);
      ctx.strokeStyle = 'rgba(43,156,148,.65)'; ctx.setLineDash([4, 4]);
      ctx.strokeRect(plot.x(wordStart), plot.top + 1, plot.x(wordEnd) - plot.x(wordStart), plot.bottom - plot.top - 2);
      ctx.setLineDash([]);
      decoded.spans.slice(1).forEach(span => {
        const x = plot.x(span.start);
        ctx.strokeStyle = palette.coral; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(x, plot.top); ctx.lineTo(x, plot.bottom); ctx.stroke();
        ctx.fillStyle = palette.coral; ctx.beginPath(); ctx.arc(x, plot.top + 2, 3, 0, Math.PI * 2); ctx.fill();
      });
      state.anchors.forEach((anchor, index) => {
        const x = plot.x(anchor);
        ctx.strokeStyle = palette.ink; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, center - 18); ctx.lineTo(x, center + 18); ctx.stroke();
        ctx.fillStyle = '#fff'; ctx.strokeStyle = palette.ink; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, center, 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = palette.ink; ctx.font = '600 11px Roboto Mono, monospace'; ctx.textAlign = 'center';
        ctx.fillText(state.characters[index] || `#${index + 1}`, x, center - 24);
      });
      ctx.fillStyle = palette.muted; ctx.font = '10px Roboto Mono, monospace'; ctx.textAlign = 'left';
      ctx.fillText('0 s', plot.left, height - 7);
      ctx.textAlign = 'right'; ctx.fillText(`${duration.toFixed(2)} s`, plot.right, height - 7);
      ctx.textAlign = 'center'; ctx.fillText('拖动汉字锚点 → 观察自动回退边界与落刀点', (plot.left + plot.right) / 2, height - 7);
      const playhead = state.playStart
        ? clamp((performance.now() - state.playStart) / 1000, 0, duration)
        : state.ambientTime;
      if (Number.isFinite(playhead)) {
        const x = plot.x(playhead);
        const glow = ctx.createLinearGradient(x - 20, 0, x + 4, 0);
        glow.addColorStop(0, 'rgba(0,113,227,0)');
        glow.addColorStop(1, 'rgba(0,113,227,.18)');
        ctx.fillStyle = glow;
        ctx.fillRect(x - 20, plot.top, 24, plot.bottom - plot.top);
        ctx.strokeStyle = 'rgba(0,113,227,.82)'; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(x, plot.top); ctx.lineTo(x, plot.bottom); ctx.stroke();
        ctx.fillStyle = palette.blue;
        ctx.beginPath(); ctx.arc(x, plot.top + 5, 2.8, 0, Math.PI * 2); ctx.fill();
      }
    }

    function renderBands(canvas, decoded) {
      const surface = setupCanvas(canvas);
      if (!surface) return;
      const { ctx, width, height } = surface;
      const { energyDb, highBandDb, fusedDb } = decoded.features;
      const duration = decoded.features.duration;
      const plot = plotMetrics(width, height, duration);
      drawPlotFrame(ctx, width, height, plot.left, width - plot.right, plot.top, height - plot.bottom);
      const startFrame = state.localNorm ? decoded.left : 0;
      const endFrame = state.localNorm ? decoded.right : energyDb.length - 1;
      const collection = [...energyDb.slice(startFrame, endFrame + 1), ...highBandDb.slice(startFrame, endFrame + 1), ...fusedDb.slice(startFrame, endFrame + 1)].filter(Number.isFinite);
      const low = Math.max(-125, Math.min(...collection));
      const high = Math.max(low + 8, Math.max(...collection));
      const y = value => plot.bottom - (clamp(value, low, high) - low) / (high - low) * (plot.bottom - plot.top);
      const x = frame => plot.left + frame / Math.max(1, energyDb.length - 1) * (plot.right - plot.left);
      const drawSeries = (values, stroke, fill, widthValue = 1.7, dashed = false) => {
        ctx.beginPath();
        values.forEach((value, index) => {
          const wiggle = Math.sin(state.ambientClock / 690 + index / Math.max(values.length - 1, 1) * Math.PI * 5) * 0.42;
          const px = x(index), py = y(value + wiggle);
          if (index === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        });
        if (fill) { ctx.lineTo(plot.right, plot.bottom); ctx.lineTo(plot.left, plot.bottom); ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); }
        ctx.strokeStyle = stroke; ctx.lineWidth = widthValue; ctx.setLineDash(dashed ? [5, 4] : []); ctx.stroke(); ctx.setLineDash([]);
      };
      drawSeries(Array.from(energyDb), palette.blue, 'rgba(0,113,227,.035)', 1.6);
      drawSeries(Array.from(highBandDb), palette.amber, null, 1.7);
      drawSeries(Array.from(fusedDb), palette.teal, 'rgba(43,156,148,.055)', 1.15, true);
      ctx.fillStyle = palette.muted; ctx.font = '10px Roboto Mono, monospace'; ctx.textAlign = 'left';
      ctx.fillText(`${low.toFixed(0)} dB`, 2, plot.bottom); ctx.fillText(`${high.toFixed(0)} dB`, 2, plot.top + 8);
      ctx.textAlign = 'right'; ctx.fillText(state.localNorm ? '局部标尺' : '全局标尺', plot.right, plot.top + 8);
      if (Number.isFinite(state.ambientTime)) {
        const scanX = plot.x(state.ambientTime);
        const scan = ctx.createLinearGradient(scanX - 28, 0, scanX + 4, 0);
        scan.addColorStop(0, 'rgba(43,156,148,0)');
        scan.addColorStop(1, 'rgba(43,156,148,.16)');
        ctx.fillStyle = scan;
        ctx.fillRect(scanX - 28, plot.top, 32, plot.bottom - plot.top);
        ctx.strokeStyle = 'rgba(43,156,148,.42)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(scanX, plot.top); ctx.lineTo(scanX, plot.bottom); ctx.stroke();
      }
    }

    function renderValley(canvas, decoded) {
      const surface = setupCanvas(canvas);
      if (!surface) return;
      const { ctx, width, height } = surface;
      const duration = decoded.features.duration;
      const plot = plotMetrics(width, height, duration);
      drawPlotFrame(ctx, width, height, plot.left, width - plot.right, plot.top, height - plot.bottom);
      const values = [];
      for (let frame = decoded.left; frame <= decoded.right; frame += 1) {
        let value = decoded.cost[frame];
        if (decoded.post?.length >= state.anchors.length) value = 0.5 * value + 0.5 * Math.abs(decoded.post[0][frame] - decoded.post[1][frame]);
        values.push({ frame, value });
      }
      const low = Math.min(...values.map(item => item.value)), high = Math.max(...values.map(item => item.value));
      const normalized = values.map((item, index) => ({
        ...item,
        value: (high - low < 1e-12 ? 0.5 : (item.value - low) / (high - low))
          + Math.sin(state.ambientClock / 760 + index * 0.19) * 0.012,
      }));
      const x = frame => plot.left + frame / Math.max(1, decoded.features.count - 1) * (plot.right - plot.left);
      const y = value => plot.bottom - value * (plot.bottom - plot.top) * 0.88 - 2;
      ctx.beginPath();
      normalized.forEach((item, index) => { if (index === 0) ctx.moveTo(x(item.frame), y(item.value)); else ctx.lineTo(x(item.frame), y(item.value)); });
      ctx.lineTo(x(normalized.at(-1).frame), plot.bottom); ctx.lineTo(x(normalized[0].frame), plot.bottom); ctx.closePath();
      const gradient = ctx.createLinearGradient(0, plot.top, 0, plot.bottom);
      gradient.addColorStop(0, 'rgba(123,107,176,.06)'); gradient.addColorStop(1, 'rgba(123,107,176,.26)');
      ctx.fillStyle = gradient; ctx.fill(); ctx.strokeStyle = palette.violet; ctx.lineWidth = 1.8; ctx.stroke();
      decoded.boundaries.slice(1, -1).forEach(frame => {
        const px = x(frame);
        ctx.strokeStyle = palette.coral; ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.moveTo(px, plot.top); ctx.lineTo(px, plot.bottom); ctx.stroke();
        ctx.fillStyle = palette.coral; ctx.beginPath(); ctx.moveTo(px, plot.top); ctx.lineTo(px - 5, plot.top + 8); ctx.lineTo(px + 5, plot.top + 8); ctx.closePath(); ctx.fill();
      });
      ctx.fillStyle = palette.muted; ctx.font = '10px Roboto Mono, monospace'; ctx.textAlign = 'left';
      ctx.fillText('argmin → 激光落刀', plot.left, height - 7); ctx.textAlign = 'right'; ctx.fillText(`${decoded.boundaries.length - 2} 个内部切点`, plot.right, height - 7);
      if (Number.isFinite(state.ambientTime)) {
        const scanX = plot.x(state.ambientTime);
        ctx.fillStyle = 'rgba(123,107,176,.10)';
        ctx.fillRect(scanX - 18, plot.top, 18, plot.bottom - plot.top);
        ctx.strokeStyle = 'rgba(123,107,176,.48)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(scanX, plot.top); ctx.lineTo(scanX, plot.bottom); ctx.stroke();
      }
    }

    function drawDspCharts() {
      if (!state.decoded) return;
      renderWave(waveCanvas, state.decoded);
      renderBands(bandsCanvas, state.decoded);
      renderValley(valleyCanvas, state.decoded);
    }

    function renderDsp() {
      if (!state.decoded) return;
      drawDspCharts();
      const pyRef = bundle.dsp_presets.find(item => item.id === state.presetId)?.py_reference;
      const referenceStarts = pyRef?.characters?.map(item => item.start_time) || [];
      const drift = referenceStarts.length === state.decoded.spans.length ? Math.abs(state.decoded.spans[0].start - referenceStarts[0]) * 1000 : 0;
      const spanText = state.decoded.spans.map(span => `${span.character} ${span.start.toFixed(3)}–${span.end.toFixed(3)}s`).join(' · ');
      statCards($('#dsp-stats'), [
        { label: '检出词边界', value: `${frameToTime(state.decoded.left, state.decoded.features.hopSeconds, state.decoded.halfWindow, state.decoded.features.duration).toFixed(3)}–${frameToTime(state.decoded.right, state.decoded.features.hopSeconds, state.decoded.halfWindow, state.decoded.features.duration).toFixed(3)} s`, accent: 'blue' },
        { label: '帧数与跳距', value: `${state.decoded.features.count} / ${(state.decoded.features.hopSeconds * 1000).toFixed(0)} ms` },
        { label: '锚点残差', value: pyRef ? `${drift.toFixed(1)} ms` : '实时输入', accent: 'teal' },
        { label: '归一化视野', value: state.localNorm ? '词内局部' : '全局', accent: state.localNorm ? 'teal' : 'coral' },
      ]);
      $('#dsp-callout').innerHTML = `${spanText}<br><span style="opacity:.78">后验对齐：${state.frameCenter ? '帧中心 frame × hop + half_window' : '未对齐 frame × hop'}；局部归一化让 −120 dB 静音不再压低词内能量谷。</span>`;
    }
    $('#dsp-preset-group')?.addEventListener('click', event => {
      state.manualUntil = performance.now() + 12000;
      const button = event.target.closest('[data-preset]');
      if (button) loadPreset(button.dataset.preset);
    });
    charsInput?.addEventListener('input', () => { state.manualUntil = performance.now() + 12000; refreshDsp(); });
    toggleLocal?.addEventListener('change', () => {
      state.manualUntil = performance.now() + 12000;
      state.localNorm = toggleLocal.checked;
      chipLocal?.classList.toggle('is-on', state.localNorm);
      refreshDsp();
    });
    toggleCenter?.addEventListener('change', () => {
      state.manualUntil = performance.now() + 12000;
      state.frameCenter = toggleCenter.checked;
      chipCenter?.classList.toggle('is-on', state.frameCenter);
      refreshDsp();
    });

    waveCanvas?.addEventListener('pointerdown', event => {
      if (!state.decoded) return;
      const rect = waveCanvas.getBoundingClientRect();
      const plot = plotMetrics(rect.width, Number(waveCanvas.dataset.baseHeight), state.decoded.features.duration);
      const time = plot.time(event.clientX - rect.left);
      let nearest = 0, distance = Infinity;
      state.anchors.forEach((anchor, index) => {
        const current = Math.abs(anchor - time);
        if (current < distance) { distance = current; nearest = index; }
      });
      if (distance < 0.08) {
        state.manualUntil = performance.now() + 12000;
        state.dragIndex = nearest;
        waveCanvas.setPointerCapture(event.pointerId);
      }
    });
    waveCanvas?.addEventListener('pointermove', event => {
      if (state.dragIndex < 0 || !state.decoded) return;
      const rect = waveCanvas.getBoundingClientRect();
      const plot = plotMetrics(rect.width, Number(waveCanvas.dataset.baseHeight), state.decoded.features.duration);
      const time = plot.time(event.clientX - rect.left);
      const index = state.dragIndex;
      const lower = index === 0 ? 0.01 : state.anchors[index - 1] + 0.008;
      state.anchors[index] = clamp(time, lower, state.decoded.features.duration - 0.01);
      for (let next = index + 1; next < state.anchors.length; next += 1) {
        if (state.anchors[next] <= state.anchors[next - 1] + 0.008) state.anchors[next] = state.anchors[next - 1] + 0.01;
      }
      refreshDsp();
    });
    const endDrag = event => {
      if (state.dragIndex >= 0) {
        state.dragIndex = -1;
        try { waveCanvas.releasePointerCapture(event.pointerId); } catch (_) {}
      }
    };
    waveCanvas?.addEventListener('pointerup', endDrag);
    waveCanvas?.addEventListener('pointercancel', endDrag);

    async function loadAudioFile(file) {
      state.manualUntil = performance.now() + 12000;
      const ctx = new AudioContext();
      try {
        const buffer = await ctx.decodeAudioData(await file.arrayBuffer());
        let signal = buffer.getChannelData(0);
        let sampleRate = buffer.sampleRate;
        if (signal.length > sampleRate * 20) signal = signal.slice(0, sampleRate * 20);
        const decodedBuffer = ctx.createBuffer(1, signal.length, sampleRate);
        decodedBuffer.copyToChannel(Float32Array.from(signal), 0);
        await ctx.close();
        loadSignal(signal, sampleRate, state.characters);
        state.audioBuffer = decodedBuffer;
        $('#dsp-preset-group')?.querySelectorAll('.seg-btn').forEach(button => button.classList.remove('is-active'));
        $('#dsp-callout').textContent = `已载入 ${file.name}，音频只在本机浏览器内解码与计算，不上传任何服务器。`;
      } catch (error) {
        $('#dsp-callout').textContent = `音频解码失败：${error.message}`;
      }
    }

    $('#wav-file-input')?.addEventListener('change', async event => {
      const file = event.target.files?.[0];
      if (file) await loadAudioFile(file);
    });
    const dropzone = $('#wav-dropzone');
    ['dragenter', 'dragover'].forEach(type => dropzone?.addEventListener(type, event => {
      event.preventDefault(); dropzone.classList.add('is-dragover');
    }));
    ['dragleave', 'drop'].forEach(type => dropzone?.addEventListener(type, event => {
      event.preventDefault(); dropzone.classList.remove('is-dragover');
    }));
    dropzone?.addEventListener('drop', async event => {
      const file = event.dataTransfer?.files?.[0];
      if (file) await loadAudioFile(file);
    });

    async function playAudio() {
      if (!state.signal) return;
      state.manualUntil = performance.now() + 12000;
      const ctx = new AudioContext();
      if (!state.audioBuffer) {
        state.audioBuffer = ctx.createBuffer(1, state.signal.length, state.sampleRate);
        state.audioBuffer.copyToChannel(Float32Array.from(state.signal), 0);
      }
      const source = ctx.createBufferSource();
      source.buffer = state.audioBuffer;
      source.connect(ctx.destination);
      state.playStart = performance.now();
      source.start();
      const animate = () => {
        renderWave(waveCanvas, state.decoded);
        if (performance.now() - state.playStart < state.decoded.features.duration * 1000) {
          state.playRequest = requestAnimationFrame(animate);
        } else {
          state.playStart = 0; renderDsp(); ctx.close();
        }
      };
      cancelAnimationFrame(state.playRequest);
      animate();
    }
    $('#btn-play-audio')?.addEventListener('click', playAudio);

    $('#btn-record-mic')?.addEventListener('click', async () => {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        $('#dsp-callout').textContent = '当前浏览器不支持麦克风录制。'; return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const recorder = new MediaRecorder(stream);
        const chunks = [];
        recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
        recorder.onstop = async () => {
          stream.getTracks().forEach(track => track.stop());
          const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
          const ctx = new AudioContext();
          try {
            const buffer = await ctx.decodeAudioData(await blob.arrayBuffer());
            const signal = buffer.getChannelData(0);
            loadSignal(signal, buffer.sampleRate, state.characters);
            state.audioBuffer = ctx.createBuffer(1, signal.length, buffer.sampleRate);
            state.audioBuffer.copyToChannel(Float32Array.from(signal), 0);
          } catch (error) { $('#dsp-callout').textContent = `录音解码失败：${error.message}`; }
        };
        recorder.start();
        $('#dsp-callout').textContent = '正在录制 2 秒，请读一个两字词。';
        setTimeout(() => recorder.stop(), 2000);
      } catch (error) { $('#dsp-callout').textContent = `无法开启麦克风：${error.message}`; }
    });

    registerAmbientDrawer('dsp', now => {
      if (!state.decoded) return;
      state.ambientClock = now;
      const duration = state.decoded.features.duration;
      state.ambientTime = ((now - state.ambientStart) / 1000) % duration;
      if (!state.lastPresetCycle) state.lastPresetCycle = now;
      if (now > state.manualUntil && now - state.lastPresetCycle > 5600) {
        const current = bundle.dsp_presets.findIndex(item => item.id === state.presetId);
        loadPreset(bundle.dsp_presets[(current + 1) % bundle.dsp_presets.length].id);
        state.lastPresetCycle = now;
      }
      drawDspCharts();
    });
    window.addEventListener('resize', () => { if (app.dsp?.decoded) renderDsp(); });
    loadPreset('shaoshui');
  }

  // ---------------------------------------------------------------------------
  // Module 2: decoder and calibration
  // ---------------------------------------------------------------------------
  function initDecoder(bundle) {
    const data = bundle.calibration_decoder;
    const state = { utteranceId: 'multi_phrase', normalization: 'robust', spike: 0, cap: 2, margin: 1.5, minProbability: 0.45, priorFix: true, manualUntil: 0, lastAmbientUi: 0, lastOptionCycle: 0 };
    app.decoder = state;

    function normalizeScores(scores, method) {
      if (method === 'zscore') {
        const average = mean(scores), scale = Math.max(std(scores), 1e-12);
        return scores.map(value => (value - average) / scale);
      }
      const center = median(scores);
      const mad = median(scores.map(value => Math.abs(value - center)));
      let scale = 1.4826 * mad;
      if (scale < 1e-12) scale = Math.max(std(scores), 1e-12);
      return scores.map(value => (value - center) / scale);
    }

    function averageDescendingRanks(values) {
      const order = values.map((value, index) => ({ value, index })).sort((a, b) => b.value - a.value || a.index - b.index);
      const ranks = new Array(values.length);
      for (let start = 0; start < order.length;) {
        let end = start + 1;
        while (end < order.length && order[end].value === order[start].value) end += 1;
        const rank = 0.5 * (start + end - 1);
        for (let i = start; i < end; i += 1) ranks[order[i].index] = rank;
        start = end;
      }
      return ranks;
    }

    function computeItems() {
      const utterance = data.utterances.find(item => item.id === state.utteranceId) || data.utterances[0];
      const raws = utterance.candidates.map(candidate => candidate.raw + (candidate.spike_target ? state.spike : 0));
      const z = normalizeScores(raws, state.normalization);
      const ranks = averageDescendingRanks(raws);
      const topRaw = Math.max(...raws);
      const params = data.calibrator_params;
      const probabilities = utterance.candidates.map((candidate, index) => {
        const features = [raws[index], z[index], topRaw - raws[index], ranks[index] / Math.max(raws.length - 1, 1)];
        let linear = params.weights[0];
        for (let feature = 0; feature < features.length; feature += 1) linear += params.weights[feature + 1] * (features[feature] - params.mean[feature]) / params.scale[feature];
        if (state.priorFix) linear -= params.log_prior_offset;
        return { candidate, raw: raws[index], z: z[index], prob: 1 / (1 + Math.exp(-clamp(linear, -35, 35))) };
      });
      const order = probabilities.map((item, index) => ({ ...item, index })).sort((a, b) => b.prob - a.prob || a.index - b.index);
      const topZ = Math.max(...z);
      const selected = [];
      for (const item of order) {
        if (selected.length >= state.cap) break;
        if (item.prob < state.minProbability) break;
        if (topZ - item.z > state.margin) continue;
        selected.push(item);
      }
      const selectedNames = new Set(selected.map(item => item.candidate.phrase));
      const expected = new Set(utterance.expected || []);
      return { utterance, probabilities, selected, selectedNames, expected, order };
    }

    function renderDecoderCanvas(model, grow = 1, now = performance.now()) {
      const surface = setupCanvas($('#canvas-decoder'));
      if (!surface) return;
      const { ctx, width, height } = surface;
      const items = [...model.probabilities].sort((a, b) => b.raw - a.raw);
      const rows = items.length, left = 88, right = 34, top = 18;
      const rowHeight = (height - top - 16) / rows;
      const zMax = Math.max(...model.probabilities.map(candidate => Math.abs(candidate.z)), 1);
      ctx.font = '11px Roboto Mono, monospace'; ctx.textBaseline = 'middle';
      items.forEach((item, index) => {
        const y = top + index * rowHeight + rowHeight / 2;
        const phrase = item.candidate.phrase;
        const isExpected = model.expected.has(phrase), isSelected = model.selectedNames.has(phrase);
        const isSpike = item.candidate.spike_target && state.spike > 0;
        if (isSpike) { ctx.fillStyle = 'rgba(230,106,78,.07)'; ctx.fillRect(6, y - rowHeight / 2 + 2, width - 12, rowHeight - 4); }
        ctx.fillStyle = isExpected ? palette.teal : palette.ink;
        ctx.font = isExpected ? '600 11px Noto Sans SC, sans-serif' : '11px Noto Sans SC, sans-serif';
        ctx.textAlign = 'left'; ctx.fillText(phrase, 10, y);
        const zBar = Math.abs(item.z) / zMax * (width - left - right) * 0.44 * grow;
        ctx.fillStyle = item.z >= 0 ? 'rgba(0,113,227,.32)' : 'rgba(123,107,176,.25)';
        roundedRect(ctx, left, y - 7, zBar, 6, 3); ctx.fill();
        const probBar = item.prob * (width - left - right) * 0.44 * grow;
        ctx.fillStyle = isSelected ? palette.teal : item.candidate.spike_target ? palette.coral : 'rgba(199,125,26,.55)';
        roundedRect(ctx, left + (width - left - right) * 0.50, y - 7, probBar, 6, 3); ctx.fill();
        ctx.fillStyle = isSelected ? palette.teal : palette.muted; ctx.textAlign = 'right'; ctx.font = '10px Roboto Mono, monospace';
        ctx.fillText(`z ${item.z.toFixed(2)}  p ${item.prob.toFixed(2)}`, width - 8, y);
        if (isSpike) { ctx.strokeStyle = palette.coral; ctx.setLineDash([3, 3]); ctx.strokeRect(6.5, y - rowHeight / 2 + 2.5, width - 13, rowHeight - 5); ctx.setLineDash([]); }
        if (isSelected) {
          const pulse = 0.5 + 0.5 * Math.sin(now / 420 + index);
          ctx.fillStyle = `rgba(43,156,148,${0.03 + pulse * 0.035})`;
          ctx.fillRect(4, y - rowHeight / 2 + 1, width - 8, rowHeight - 2);
        }
      });
      ctx.fillStyle = palette.muted; ctx.textAlign = 'left'; ctx.font = '10px Roboto Mono, monospace'; ctx.fillText('Z / 概率', width - 170, 9);
    }

    function renderDecoder() {
      const model = computeItems();
      renderDecoderCanvas(model);
      const expectedHits = [...model.expected].filter(name => model.selectedNames.has(name)).length;
      const falsePositives = [...model.selectedNames].filter(name => !model.expected.has(name)).length;
      const missed = [...model.expected].filter(name => !model.selectedNames.has(name));
      statCards($('#decoder-stats'), [
        { label: '选择集合', value: model.selected.length ? model.selected.map(item => item.candidate.phrase).join(' / ') : '拒识', accent: 'teal' },
        { label: '命中 / 期望', value: `${expectedHits} / ${model.expected.size}`, accent: expectedHits === model.expected.size ? 'teal' : 'coral' },
        { label: '误报候选', value: String(falsePositives), accent: falsePositives ? 'coral' : 'teal' },
        { label: '归一化', value: state.normalization === 'robust' ? 'Median-MAD' : 'Z-score', accent: 'blue' },
      ]);
      const selectedText = model.selected.length ? model.selected.map(item => item.candidate.phrase).join('、') : '无';
      const missedText = missed.length ? `漏掉：${missed.join('、')}` : '没有漏掉目标词';
      const rescue = model.utterance.id === 'continue_rescue' && model.selectedNames.has('烧水');
      $('#decoder-callout').innerHTML = `当前集合：<strong>${selectedText}</strong>。${missedText}。${rescue ? ' v3 使用 <code>continue</code> 跳过被 z-space margin 拒绝的干扰项，因此保住了后面的真目标。' : ''}`;
      renderSurface(); renderReliability(); updateSliderLabels();
    }

    function updateSliderLabels() {
      $('#val-spike').textContent = `+${state.spike.toFixed(2)}`;
      $('#val-cap').textContent = String(state.cap);
      $('#val-margin').textContent = state.margin.toFixed(1);
      $('#val-minp').textContent = state.minProbability.toFixed(2);
      $('#surface-point-label').textContent = `cap ${state.cap} · margin ${state.margin.toFixed(1)} · min p ${state.minProbability.toFixed(2)}`;
    }

    function renderReliability(updateStats = true, now = performance.now()) {
      const surface = setupCanvas($('#canvas-reliability'));
      if (!surface) return;
      const { ctx, width, height } = surface;
      const reliability = state.priorFix ? data.reliability.corrected : data.reliability.uncorrected;
      const plot = { left: 42, right: width - 20, top: 18, bottom: height - 28 };
      drawPlotFrame(ctx, width, height, plot.left, 20, plot.top, 28);
      const x = value => plot.left + value * (plot.right - plot.left);
      const y = value => plot.bottom - value * (plot.bottom - plot.top);
      ctx.strokeStyle = 'rgba(29,29,31,.22)'; ctx.setLineDash([5, 5]);
      ctx.beginPath(); ctx.moveTo(x(0), y(0)); ctx.lineTo(x(1), y(1)); ctx.stroke(); ctx.setLineDash([]);
      const travel = (now / 2800) % 1;
      ctx.fillStyle = 'rgba(0,113,227,.18)';
      ctx.beginPath(); ctx.arc(x(travel), y(travel), 8, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = palette.blue;
      ctx.beginPath(); ctx.arc(x(travel), y(travel), 3, 0, Math.PI * 2); ctx.fill();
      const barWidth = (plot.right - plot.left) / reliability.bins.length * 0.62;
      reliability.bins.forEach(bin => {
        if (!bin.count) return;
        const bx = x(bin.center) - barWidth / 2;
        const livingOffset = Math.sin(now / 560 + bin.center * 7) * 0.007;
        const confValue = clamp(bin.avg_conf + livingOffset, 0, 1);
        const accValue = clamp(bin.avg_acc + livingOffset, 0, 1);
        const top = y(Math.max(confValue, accValue));
        ctx.fillStyle = 'rgba(0,113,227,.13)';
        ctx.fillRect(bx, top, barWidth, Math.abs(y(confValue) - y(accValue)));
        ctx.fillStyle = palette.blue; ctx.fillRect(bx, y(confValue) - 2, barWidth, 4);
        ctx.fillStyle = palette.teal; ctx.beginPath(); ctx.arc(x(bin.center), y(accValue), 3.5, 0, Math.PI * 2); ctx.fill();
      });
      ctx.fillStyle = palette.muted; ctx.font = '10px Roboto Mono, monospace'; ctx.textAlign = 'left'; ctx.fillText('置信度', plot.left, height - 8);
      ctx.textAlign = 'right'; ctx.fillText('准确率', plot.right, height - 8);
      if (updateStats) statCards($('#calib-stats'), [
        { label: 'ECE', value: fmt(reliability.ece, 4), accent: reliability.ece < 0.05 ? 'teal' : 'coral' },
        { label: 'Brier', value: fmt(reliability.brier, 4), accent: reliability.brier < 0.1 ? 'teal' : 'amber' },
        { label: '先验修正', value: state.priorFix ? '已启用' : '未启用', accent: state.priorFix ? 'teal' : 'coral' },
        { label: '校准状态', value: state.priorFix ? '贴近对角线' : '概率虚高', accent: state.priorFix ? 'teal' : 'coral' },
      ]);
    }
    function operatingPoint() {
      let best = data.operating_surface[0], bestDistance = Infinity;
      for (const point of data.operating_surface) {
        const distance = Math.abs(point.cap - state.cap) * 10 + Math.abs(point.margin - state.margin) + Math.abs(point.min_probability - state.minProbability) * 5;
        if (distance < bestDistance) { bestDistance = distance; best = point; }
      }
      return best;
    }

    function renderSurface(now = performance.now()) {
      const surface = setupCanvas($('#canvas-surface'));
      if (!surface) return;
      const { ctx, width, height } = surface;
      const plot = { left: 46, right: width - 18, top: 16, bottom: height - 26 };
      drawPlotFrame(ctx, width, height, plot.left, 18, plot.top, 26);
      const x = value => plot.left + clamp(value, 0, 0.6) / 0.6 * (plot.right - plot.left);
      const y = value => plot.bottom - clamp(value, 0, 1) * (plot.bottom - plot.top);
      data.operating_surface.forEach(point => {
        ctx.fillStyle = `rgba(230,106,78,${0.16 + point.far * 0.5})`;
        ctx.beginPath(); ctx.arc(x(point.far), y(point.f1), 2.4, 0, Math.PI * 2); ctx.fill();
      });
      const point = operatingPoint();
      ctx.strokeStyle = palette.blue; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x(point.far), y(point.f1), 6, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = palette.blue; ctx.beginPath(); ctx.arc(x(point.far), y(point.f1), 3.2, 0, Math.PI * 2); ctx.fill();
      const ring = 8 + 5 * (0.5 + 0.5 * Math.sin(now / 520));
      ctx.strokeStyle = 'rgba(0,113,227,.16)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(x(point.far), y(point.f1), ring, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = palette.muted; ctx.font = '10px Roboto Mono, monospace'; ctx.textAlign = 'left'; ctx.fillText('FAR →', plot.left, height - 7);
      ctx.save(); ctx.translate(12, (plot.top + plot.bottom) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText('F1 ↑', 0, 0); ctx.restore();
    }

    $('#utt-preset-group')?.addEventListener('click', event => {
      state.manualUntil = performance.now() + 12000;
      const button = event.target.closest('[data-utt]');
      if (!button) return;
      state.utteranceId = button.dataset.utt; state.spike = 0; $('#slider-spike').value = '0'; renderDecoder();
    });
    $('#norm-method-group')?.addEventListener('click', event => {
      state.manualUntil = performance.now() + 12000;
      const button = event.target.closest('[data-norm]');
      if (!button) return;
      state.normalization = button.dataset.norm; renderDecoder();
    });
    $('#slider-spike')?.addEventListener('input', event => { state.manualUntil = performance.now() + 12000; state.spike = Number(event.target.value); renderDecoder(); });
    $('#slider-cap')?.addEventListener('input', event => { state.manualUntil = performance.now() + 12000; state.cap = Number(event.target.value); renderDecoder(); });
    $('#slider-margin')?.addEventListener('input', event => { state.manualUntil = performance.now() + 12000; state.margin = Number(event.target.value); renderDecoder(); });
    $('#slider-minp')?.addEventListener('input', event => { state.manualUntil = performance.now() + 12000; state.minProbability = Number(event.target.value); renderDecoder(); });
    $('#toggle-prior-fix')?.addEventListener('change', event => {
      state.manualUntil = performance.now() + 12000;
      state.priorFix = event.target.checked;
      $('#chip-prior-fix')?.classList.toggle('is-on', state.priorFix);
      renderDecoder();
    });
    registerAmbientDrawer('decoder', (now) => {
      const autoAllowed = now > state.manualUntil;
      if (autoAllowed) {
        if (!state.lastOptionCycle) state.lastOptionCycle = now;
        if (now - state.lastOptionCycle > 6100) {
          const current = data.utterances.findIndex(item => item.id === state.utteranceId);
          state.utteranceId = data.utterances[(current + 1) % data.utterances.length].id;
          state.normalization = state.normalization === 'robust' ? 'zscore' : 'robust';
          setActiveSegment('#utt-preset-group', 'utt', state.utteranceId);
          setActiveSegment('#norm-method-group', 'norm', state.normalization);
          state.lastOptionCycle = now;
          renderDecoder();
        }
        state.spike = 0.38 + 0.86 * (0.5 + 0.5 * Math.sin(now / 1450));
        $('#slider-spike').value = state.spike.toFixed(2);
      }
      const model = computeItems();
      const grow = 0.965 + 0.035 * Math.sin(now / 380);
      renderDecoderCanvas(model, grow, now);
      renderReliability(false, now);
      renderSurface(now);
      if (autoAllowed && now - state.lastAmbientUi > 260) {
        state.lastAmbientUi = now;
        updateSliderLabels();
        renderDecoder();
      }
    });
    window.addEventListener('resize', () => { if (app.decoder) renderDecoder(); });
    renderDecoder();
  }

  // ---------------------------------------------------------------------------
  // Module 3: localisation lie detector and frequency bias
  // ---------------------------------------------------------------------------
  function initEvaluation(bundle) {
    const data = bundle.eval_lie_detector;
    const state = { predicted: data.timeline.predicted_spans.map(span => ({ ...span })), dragId: null, dragOffset: 0, dragDuration: 0, freqPreset: 'head_biased', manualUntil: 0, lastAmbientUi: 0, lastFreqCycle: 0 };
    app.eval = state;
    const canvas = $('#canvas-hungarian');
    const duration = data.timeline.duration;

    function iou(aStart, aEnd, bStart, bEnd) {
      const intersection = Math.max(0, Math.min(aEnd, bEnd) - Math.max(aStart, bStart));
      const union = Math.max(aEnd, bEnd) - Math.min(aStart, bStart);
      return union > 0 ? intersection / union : 0;
    }

    function computeMetrics() {
      const expected = data.timeline.expected_spans;
      const valid = state.predicted.filter(span => Number.isFinite(span.start) && Number.isFinite(span.end));
      const knownPairs = [], phraseMatches = [];
      expected.forEach(exp => {
        const candidates = valid.filter(pred => pred.phrase === exp.phrase);
        if (!candidates.length) return;
        let best = candidates[0], bestIou = -1;
        candidates.forEach(pred => {
          const overlap = iou(exp.start, exp.end, pred.start, pred.end);
          if (overlap > bestIou) { bestIou = overlap; best = pred; }
        });
        knownPairs.push(bestIou); phraseMatches.push({ exp, pred: best, iou: bestIou });
      });
      const threshold = data.timeline.iou_threshold;
      const matched = knownPairs.filter(value => value >= threshold).length;
      const precision = matched / Math.max(state.predicted.length, 1);
      const recall = matched / Math.max(expected.length, 1);
      const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
      const naiveMatched = phraseMatches.length;
      const naivePrecision = naiveMatched / Math.max(valid.length, 1);
      const naiveRecall = naiveMatched / Math.max(expected.length, 1);
      const naiveF1 = naivePrecision + naiveRecall > 0 ? 2 * naivePrecision * naiveRecall / (naivePrecision + naiveRecall) : 0;
      return { expected, valid, phraseMatches, matched, precision, recall, f1, naiveMatched, naivePrecision, naiveRecall, naiveF1 };
    }

    function drawSpan(ctx, x, y, width, heightValue, fill, stroke, label, sublabel) {
      roundedRect(ctx, x, y, Math.max(width, 3), heightValue, 6);
      ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = palette.ink; ctx.font = '600 11px Noto Sans SC, sans-serif'; ctx.textAlign = 'center';
      if (width > 28) ctx.fillText(label, x + width / 2, y + 15);
      if (sublabel && width > 56) { ctx.fillStyle = palette.muted; ctx.font = '9px Roboto Mono, monospace'; ctx.fillText(sublabel, x + width / 2, y + 27); }
    }

    function renderEvalCards(metrics) {
      const floor = data.floor_trap_demo;
      const host = $('#eval-compare-grid');
      if (!host) return;
      host.innerHTML = `
        <div class="compare-card compare-card--naive">
          <h4><span>天真指标</span><span class="mono">${metrics.naiveF1.toFixed(2)}</span></h4>
          <div class="metric-line"><span>同词命中（忽略 IoU）</span><strong>${metrics.naiveMatched}</strong></div>
          <div class="metric-line"><span>precision / recall</span><strong>${metrics.naivePrecision.toFixed(2)} / ${metrics.naiveRecall.toFixed(2)}</strong></div>
          <div class="metric-line"><span>no-position 进入分母</span><strong>否</strong></div>
          <div class="metric-line"><span>负样本地板分</span><strong>${floor.naive_sentence_recall_complete.toFixed(2)}</strong></div>
          <p class="metric-note">跳过错位与不可定位预测，指标看起来更漂亮。</p>
        </div>
        <div class="compare-card compare-card--honest">
          <h4><span>v3 诚实指标</span><span class="mono">${metrics.f1.toFixed(2)}</span></h4>
          <div class="metric-line"><span>IoU ≥ ${data.timeline.iou_threshold} 命中</span><strong>${metrics.matched}</strong></div>
          <div class="metric-line"><span>precision / recall</span><strong>${metrics.precision.toFixed(2)} / ${metrics.recall.toFixed(2)}</strong></div>
          <div class="metric-line"><span>no-position 进入分母</span><strong>是</strong></div>
          <div class="metric-line"><span>正样本完整召回</span><strong>${floor.v3_sentence_recall_complete.toFixed(2)}</strong></div>
          <p class="metric-note">零重叠同词与不可定位预测都承担代价，微平均不再遮住长尾。</p>
        </div>`;
    }

    function renderSpanControls() {
      const host = $('#span-controls');
      if (!host) return;
      host.innerHTML = state.predicted.map(span => {
        const tagClass = span.char_source === 'reading' ? 'tag-reading' : span.char_source === 'text-fallback' ? 'tag-fallback' : 'tag-nopos';
        return `<button class="span-pill" data-span-id="${span.id}" type="button"><strong>${span.phrase}</strong><span class="span-pill__tag ${tagClass}">${span.char_source}</span></button>`;
      }).join('');
      $$('.span-pill', host).forEach(button => button.addEventListener('click', () => {
        state.manualUntil = performance.now() + 12000;
        const span = state.predicted.find(item => item.id === button.dataset.spanId);
        if (!span) return;
        const cycle = ['reading', 'text-fallback', 'no-position'];
        const next = cycle[(cycle.indexOf(span.char_source) + 1) % cycle.length];
        span.char_source = next;
        if (next === 'no-position') { span.start = null; span.end = null; }
        else if (!Number.isFinite(span.start)) {
          const expected = data.timeline.expected_spans.find(item => item.phrase === span.phrase);
          span.start = expected ? expected.start : 0.2; span.end = expected ? expected.end : 0.7;
        }
        renderHungarian();
      }));
    }

    function renderHungarian(updateDom = true) {
      const surface = setupCanvas(canvas);
      if (!surface) return;
      const { ctx, width, height } = surface;
      const left = 46, right = width - 20;
      const x = time => left + clamp(time, 0, duration) / duration * (right - left);
      const metrics = computeMetrics();
      ctx.strokeStyle = 'rgba(29,29,31,.12)'; ctx.lineWidth = 1;
      for (let second = 0; second <= duration; second += 1) {
        const px = x(second);
        ctx.beginPath(); ctx.moveTo(px, 18); ctx.lineTo(px, height - 28); ctx.stroke();
        ctx.fillStyle = palette.muted; ctx.font = '9px Roboto Mono, monospace'; ctx.textAlign = 'center'; ctx.fillText(`${second}s`, px, height - 9);
      }
      ctx.fillStyle = palette.muted; ctx.font = '10px Roboto Mono, monospace'; ctx.textAlign = 'left';
      ctx.fillText('expected', 6, 45); ctx.fillText('predicted', 6, 170);
      metrics.expected.forEach(span => drawSpan(ctx, x(span.start), 28, x(span.end) - x(span.start), 34, 'rgba(43,156,148,.12)', palette.teal, span.phrase, `${span.start.toFixed(2)}–${span.end.toFixed(2)}s`));
      const rowY = 153;
      state.predicted.forEach(span => {
        const hasPosition = Number.isFinite(span.start) && Number.isFinite(span.end);
        if (!hasPosition) {
          ctx.fillStyle = 'rgba(230,106,78,.09)'; roundedRect(ctx, right - 98, rowY + 32, 96, 20, 5); ctx.fill();
          ctx.fillStyle = palette.coral; ctx.font = '9px Roboto Mono, monospace'; ctx.textAlign = 'center'; ctx.fillText(`${span.phrase} · no-pos`, right - 50, rowY + 45);
          return;
        }
        const fill = span.char_source === 'reading' ? 'rgba(0,113,227,.11)' : span.char_source === 'text-fallback' ? 'rgba(199,125,26,.13)' : 'rgba(230,106,78,.12)';
        const stroke = span.char_source === 'reading' ? palette.blue : span.char_source === 'text-fallback' ? palette.amber : palette.coral;
        drawSpan(ctx, x(span.start), rowY, x(span.end) - x(span.start), 32, fill, stroke, span.phrase, span.char_source);
      });
      metrics.phraseMatches.forEach(pair => {
        const fromX = (x(pair.exp.start) + x(pair.exp.end)) / 2, toX = (x(pair.pred.start) + x(pair.pred.end)) / 2;
        const topY = 62, bottomY = 153, controlY = (topY + bottomY) / 2;
        ctx.beginPath(); ctx.moveTo(fromX, topY); ctx.bezierCurveTo(fromX, controlY, toX, controlY, toX, bottomY);
        ctx.strokeStyle = pair.iou >= data.timeline.iou_threshold ? 'rgba(43,156,148,.62)' : 'rgba(230,106,78,.65)';
        ctx.lineWidth = pair.iou >= data.timeline.iou_threshold ? 1.8 : 1.4; ctx.setLineDash(pair.iou >= data.timeline.iou_threshold ? [] : [4, 4]); ctx.stroke(); ctx.setLineDash([]);
        const labelX = (fromX + toX) / 2, labelY = controlY - 2;
        ctx.fillStyle = '#fff'; roundedRect(ctx, labelX - 22, labelY - 7, 44, 14, 5); ctx.fill(); ctx.strokeStyle = 'rgba(29,29,31,.08)'; ctx.stroke();
        ctx.fillStyle = pair.iou >= data.timeline.iou_threshold ? palette.teal : palette.coral; ctx.font = '9px Roboto Mono, monospace'; ctx.textAlign = 'center'; ctx.fillText(`IoU ${pair.iou.toFixed(2)}`, labelX, labelY + 3);
      });
      ctx.fillStyle = palette.muted; ctx.font = '10px Roboto Mono, monospace'; ctx.textAlign = 'right';
      ctx.fillText(`honest F1 ${metrics.f1.toFixed(2)} · naive F1 ${metrics.naiveF1.toFixed(2)}`, right, 10);
      const scanTime = (performance.now() / 32) % duration;
      const scanX = x(scanTime);
      const scan = ctx.createLinearGradient(scanX - 24, 0, scanX + 4, 0);
      scan.addColorStop(0, 'rgba(0,113,227,0)');
      scan.addColorStop(1, 'rgba(0,113,227,.10)');
      ctx.fillStyle = scan;
      ctx.fillRect(scanX - 24, 18, 28, height - 46);
      ctx.strokeStyle = 'rgba(0,113,227,.28)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(scanX, 18); ctx.lineTo(scanX, height - 28); ctx.stroke();
      if (updateDom) { renderEvalCards(metrics); renderSpanControls(); }
    }
    function predictedAt(event) {
      const rect = canvas.getBoundingClientRect(), width = rect.width, left = 46, right = width - 20;
      const x = event.clientX - rect.left, y = event.clientY - rect.top;
      if (y < 145 || y > 190) return null;
      return state.predicted.find(span => {
        if (!Number.isFinite(span.start)) return false;
        const startX = left + span.start / duration * (right - left);
        const endX = left + span.end / duration * (right - left);
        return x >= startX && x <= endX;
      }) || null;
    }
    canvas?.addEventListener('pointerdown', event => {
      state.manualUntil = performance.now() + 12000;
      const span = predictedAt(event);
      if (!span) return;
      const rect = canvas.getBoundingClientRect(), width = rect.width;
      const time = (event.clientX - rect.left - 46) / (width - 66) * duration;
      state.dragId = span.id; state.dragOffset = time - span.start; state.dragDuration = span.end - span.start;
      canvas.setPointerCapture(event.pointerId); canvas.style.cursor = 'grabbing';
    });
    canvas?.addEventListener('pointermove', event => {
      if (!state.dragId) { canvas.style.cursor = predictedAt(event) ? 'grab' : 'default'; return; }
      const span = state.predicted.find(item => item.id === state.dragId);
      if (!span) return;
      const rect = canvas.getBoundingClientRect(), width = rect.width;
      const time = (event.clientX - rect.left - 46) / (width - 66) * duration;
      const newStart = clamp(time - state.dragOffset, 0, duration - state.dragDuration);
      span.start = newStart; span.end = newStart + state.dragDuration;
      renderHungarian();
    });
    const finishDrag = event => {
      if (state.dragId) {
        state.dragId = null; canvas.style.cursor = 'grab';
        try { canvas.releasePointerCapture(event.pointerId); } catch (_) {}
      }
    };
    canvas?.addEventListener('pointerup', finishDrag);
    canvas?.addEventListener('pointercancel', finishDrag);
    $('#btn-reset-spans')?.addEventListener('click', () => {
      state.manualUntil = performance.now() + 12000; state.predicted = data.timeline.predicted_spans.map(span => ({ ...span })); renderHungarian(); });
    $('#btn-drift-demo')?.addEventListener('click', () => {
      state.manualUntil = performance.now() + 12000;
      const span = state.predicted.find(item => item.id === 'p1');
      if (span) { span.start = 3.15; span.end = 3.82; }
      renderHungarian();
    });

    function renderFrequency(now = performance.now(), updateStats = true) {
      const preset = data.frequency_presets[state.freqPreset];
      const surface = setupCanvas($('#canvas-freq'));
      if (!surface) return;
      const { ctx, width, height } = surface;
      const plot = { left: 45, right: width - 18, top: 18, bottom: height - 30 };
      drawPlotFrame(ctx, width, height, plot.left, 18, plot.top, 30);
      const counts = preset.points.map(point => Math.log10(Math.max(point.count, 1)));
      const minCount = Math.min(...counts), maxCount = Math.max(...counts);
      const x = value => plot.left + (Math.log10(Math.max(value, 1)) - minCount) / Math.max(maxCount - minCount, 1e-9) * (plot.right - plot.left);
      const y = value => plot.bottom - value * (plot.bottom - plot.top);
      const classColors = { head: palette.blue, body: palette.teal, tail: palette.coral };
      const fillColors = { head: 'rgba(0,113,227,.12)', body: 'rgba(43,156,148,.12)', tail: 'rgba(230,106,78,.12)' };
      preset.points.forEach((point, index) => {
        const cx = x(point.count);
        const cy = y(clamp(point.recall + Math.sin(now / 620 + index * 0.8) * 0.012, 0, 1));
        ctx.fillStyle = fillColors[point.freq_class] || 'rgba(123,107,176,.12)'; ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = classColors[point.freq_class] || palette.violet; ctx.lineWidth = 1.5; ctx.stroke();
        ctx.fillStyle = palette.muted; ctx.font = '9px Noto Sans SC, sans-serif'; ctx.textAlign = 'center'; ctx.fillText(point.word, cx, cy - 11);
      });
      const xs = counts, ys = preset.points.map(point => point.recall), xMean = mean(xs), yMean = mean(ys);
      const slope = xs.reduce((sum, value, index) => sum + (value - xMean) * (ys[index] - yMean), 0) / Math.max(xs.reduce((sum, value) => sum + (value - xMean) ** 2, 0), 1e-9);
      const intercept = yMean - slope * xMean;
      ctx.strokeStyle = palette.violet; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(plot.left, y(clamp(slope * minCount + intercept, 0, 1))); ctx.lineTo(plot.right, y(clamp(slope * maxCount + intercept, 0, 1))); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = palette.muted; ctx.font = '10px Roboto Mono, monospace'; ctx.textAlign = 'left'; ctx.fillText('词频（log）→', plot.left, height - 9);
      const travel = (now / 3200) % 1;
      const trackX = plot.left + travel * (plot.right - plot.left);
      const trackLog = minCount + travel * (maxCount - minCount);
      const trackY = y(clamp(slope * trackLog + intercept, 0, 1));
      ctx.fillStyle = 'rgba(123,107,176,.15)';
      ctx.beginPath(); ctx.arc(trackX, trackY, 8, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = palette.violet;
      ctx.beginPath(); ctx.arc(trackX, trackY, 3, 0, Math.PI * 2); ctx.fill();
      ctx.textAlign = 'right'; ctx.fillText('单字召回率 ↑', plot.right, 16);
      if (updateStats) statCards($('#freq-stats'), [
        { label: 'Micro recall', value: pct(preset.micro_recall), accent: 'blue' },
        { label: 'Macro recall', value: pct(preset.macro_recall), accent: 'teal' },
        { label: 'Tail recall', value: pct(preset.tail_recall), accent: preset.tail_recall < 0.6 ? 'coral' : 'teal' },
        { label: 'Spearman ρ', value: preset.spearman.toFixed(3), accent: 'amber' },
      ]);
      if (updateStats) {
        $('#freq-callout').innerHTML = state.freqPreset === 'head_biased'
          ? `Micro ${pct(preset.micro_recall)} 看似接近均衡模型，但长尾词召回只有 <strong>${pct(preset.tail_recall)}</strong>。频率与召回几乎完全正相关，说明模型把大部分能力押在高频词上。`
          : `Micro ${pct(preset.micro_recall)} 与 Macro ${pct(preset.macro_recall)} 接近，Tail 召回保持 <strong>${pct(preset.tail_recall)}</strong>，长尾方言词不再被平均分遮住。`;
      }
    }

    $('#freq-preset-group')?.addEventListener('click', event => {
      state.manualUntil = performance.now() + 12000;
      const button = event.target.closest('[data-freq]');
      if (!button) return;
      state.freqPreset = button.dataset.freq; renderFrequency();
    });
    registerAmbientDrawer('eval', (now) => {
      if (now > state.manualUntil && !state.dragId) {
        if (!state.lastFreqCycle) state.lastFreqCycle = now;
        if (now - state.lastFreqCycle > 4700) {
          state.freqPreset = state.freqPreset === 'head_biased' ? 'balanced' : 'head_biased';
          setActiveSegment('#freq-preset-group', 'freq', state.freqPreset);
          state.lastFreqCycle = now;
          renderFrequency(now);
        }
        const p0 = state.predicted.find(span => span.id === 'p0');
        const p1 = state.predicted.find(span => span.id === 'p1');
        const p2 = state.predicted.find(span => span.id === 'p2');
        if (p0) { p0.start = 0.52 + 0.10 * Math.sin(now / 1750); p0.end = p0.start + 0.64; }
        if (p1) { p1.start = 1.62 + 0.28 * Math.sin(now / 2100); p1.end = p1.start + 0.66; }
        if (p2) { p2.start = 2.88 + 0.22 * Math.cos(now / 1900); p2.end = p2.start + 0.80; }
      }
      renderHungarian(false);
      renderFrequency(now, false);
      if (now - state.lastAmbientUi > 280) {
        state.lastAmbientUi = now;
        renderEvalCards(computeMetrics());
      }
    });
    window.addEventListener('resize', () => { renderHungarian(); renderFrequency(); });
    renderHungarian(); renderFrequency();
  }

  // ---------------------------------------------------------------------------
  // Module 4: corpus plan sandbox and leakage X-ray
  // ---------------------------------------------------------------------------
  function initPlan(bundle) {
    const state = {
      mode: 'v3', injected: false, defaultWords: bundle.manifest_sandbox.default_words, manualUntil: 0, lastAutoCycle: 0, contract: null,
      confusions: { ...bundle.manifest_sandbox.default_confusions }, backgrounds: [...bundle.manifest_sandbox.default_backgrounds],
    };
    app.plan = state;
    const textarea = $('#lexicon-textarea');
    const positions = ['head', 'body', 'tail'];
    const templates = {
      head: ['请{word}', '帮我{word}', '现在{word}'],
      body: ['请帮我{word}一下', '先{word}再继续', '麻烦把{word}处理一下'],
      tail: ['请帮我处理{word}', '现在开始{word}', '可以帮我{word}吗'],
    };

    function lexicon() {
      return [...new Set(textarea.value.split(/\r?\n/).map(value => value.trim()).filter(Boolean))];
    }

    function buildRecords(words) {
      const records = [];
      words.forEach((word, index) => {
        const wordPositions = state.mode === 'v3'
          ? [positions[index % 3], positions[(index + 1) % 3], positions[(index + 2) % 3]]
          : [positions[index % 3]];
        wordPositions.forEach((position, offset) => {
          const pool = templates[position];
          records.push({ id: `positive-${index}-${offset}`, word, position, text: pool[(index + offset) % pool.length].replace('{word}', word), subset: 'positive' });
        });
      });
      const confusionPool = state.injected ? [...Object.values(state.confusions), '请开灯', '先开灯再说'] : Object.values(state.confusions);
      words.forEach((word, index) => {
        records.push({ id: `hard-negative-${index}`, word, position: positions[index % 3], text: `不要${confusionPool[index % confusionPool.length] || '随便说说'}`, subset: 'hard-negative' });
      });
      const backgroundPool = state.injected ? [...state.backgrounds, '请打开电视', '先开灯'] : state.backgrounds;
      backgroundPool.forEach((background, index) => {
        records.push({ id: `background-${index}`, word: '', position: positions[index % 3], text: `今天的天气适合${background}`, subset: 'background' });
      });
      return records;
    }

    function analyze(words, records) {
      const conflicts = [];
      words.forEach(shorter => words.forEach(longer => {
        if (shorter !== longer && longer.includes(shorter)) conflicts.push({ shorter, longer });
      }));
      const hardNegativeLeakages = [], backgroundLeakages = [];
      records.forEach(record => {
        words.forEach(word => {
          if (record.subset !== 'positive' && record.text.includes(word)) {
            const item = { sampleId: record.id, phrase: word, text: record.text };
            if (record.subset === 'hard-negative') hardNegativeLeakages.push(item);
            if (record.subset === 'background') backgroundLeakages.push(item);
          }
        });
      });
      const coverage = Object.fromEntries(words.map(word => [word, { head: 0, body: 0, tail: 0 }]));
      records.filter(record => record.subset === 'positive').forEach(record => {
        if (coverage[record.word]?.[record.position] !== undefined) coverage[record.word][record.position] += 1;
      });
      return { conflicts, hardNegativeLeakages, backgroundLeakages, coverage };
    }

    function renderHeatmap(words, analysis) {
      const host = $('#heatmap-container');
      if (!host) return;
      host.innerHTML = `<table class="heatmap-table">
        <thead><tr><th>词语</th><th>句首</th><th>句中</th><th>句尾</th></tr></thead>
        <tbody>${words.map(word => `<tr><td>${word}</td>${positions.map(position => {
          const count = analysis.coverage[word]?.[position] || 0;
          const className = count === 0 ? 'hm-zero' : count > 1 ? 'hm-high' : 'hm-ok';
          return `<td class="hm-cell ${className}">${count}</td>`;
        }).join('')}</tr>`).join('')}</tbody></table>`;
      const zeros = words.filter(word => positions.some(position => !(analysis.coverage[word]?.[position] > 0))).length;
      $('#heatmap-summary').textContent = state.mode === 'v3' ? `${words.length} 词 · 每词三位置覆盖` : `${words.length} 词 · ${zeros} 词存在空位`;
    }

    function renderXray(words, records, analysis) {
      const alerts = analysis.conflicts.length + analysis.hardNegativeLeakages.length + analysis.backgroundLeakages.length;
      const pass = alerts === 0;
      const items = [];
      analysis.conflicts.forEach(item => items.push(`<div class="xray-item xray-item--alert"><strong>子串碰撞</strong>：“${item.shorter}”包含于“${item.longer}”。热词表需要避免共享子串，否则字级召回无法归因。</div>`));
      analysis.hardNegativeLeakages.slice(0, 5).forEach(item => items.push(`<div class="xray-item xray-item--alert"><strong>硬负样本泄漏</strong>：${item.sampleId} 的文本“${item.text}”包含已注册词“${item.phrase}”。</div>`));
      analysis.backgroundLeakages.slice(0, 5).forEach(item => items.push(`<div class="xray-item xray-item--alert"><strong>背景泄漏</strong>：${item.sampleId} 的文本“${item.text}”包含已注册词“${item.phrase}”。</div>`));
      if (pass) items.push(`<div class="xray-item"><strong>全词表通过</strong>：未发现子串碰撞、硬负样本泄漏或背景泄漏。</div>`);
      const subsetCounts = records.reduce((accumulator, record) => { accumulator[record.subset] = (accumulator[record.subset] || 0) + 1; return accumulator; }, {});
      items.push(`<div class="xray-item"><strong>规划规模</strong>：${records.length} 条记录，positive ${subsetCounts.positive || 0}，hard-negative ${subsetCounts['hard-negative'] || 0}，background ${subsetCounts.background || 0}。</div>`);
      $('#xray-report').innerHTML = `<div class="xray-status ${pass ? 'xray-status--pass' : 'xray-status--fail'}"><span>${pass ? '绿灯 · DATA_CONTRACT 通过' : '红灯 · 发现数据契约隐患'}</span><span class="mono">${alerts} alert${alerts === 1 ? '' : 's'}</span></div><div class="xray-list">${items.join('')}</div>`;
    }

    function drawContractFrame(now = performance.now()) {
      if (!state.contract) return;
      const surface = setupCanvas($('#canvas-contract'));
      if (!surface) return;
      const { ctx, width, height } = surface;
      const { records, analysis } = state.contract;
      const counts = records.reduce((accumulator, record) => {
        accumulator[record.subset] = (accumulator[record.subset] || 0) + 1;
        return accumulator;
      }, {});
      const rows = [
        { label: '正向记录', count: counts.positive || 0, note: '目标词必须出现', leaks: 0, color: palette.teal },
        { label: '硬负样本', count: counts['hard-negative'] || 0, note: '期望目标集合为空', leaks: analysis.hardNegativeLeakages.length, color: palette.amber },
        { label: '背景语料', count: counts.background || 0, note: '不得包含注册词', leaks: analysis.backgroundLeakages.length, color: palette.violet },
      ];
      const left = Math.max(24, width * 0.04);
      const leftWidth = Math.min(190, width * 0.24);
      const center = width * 0.54;
      const right = width - Math.max(26, width * 0.04);
      const statusWidth = Math.min(180, width * 0.25);
      const top = 56;
      const rowGap = 66;
      ctx.font = '10px Roboto Mono, monospace';
      ctx.textAlign = 'left'; ctx.fillStyle = palette.muted;
      ctx.fillText('合成记录', left, 24);
      ctx.textAlign = 'center'; ctx.fillText('validate_plan', center, 24);
      ctx.textAlign = 'right'; ctx.fillText('契约结果', right, 24);

      rows.forEach((row, index) => {
        const y = top + index * rowGap;
        const failed = row.leaks > 0;
        const stateColor = failed ? palette.coral : row.color;
        roundedRect(ctx, left, y - 18, leftWidth, 38, 10);
        ctx.fillStyle = failed ? 'rgba(230,106,78,.09)' : 'rgba(29,29,31,.035)';
        ctx.fill();
        ctx.fillStyle = palette.ink; ctx.font = '600 11px Noto Sans SC, sans-serif'; ctx.textAlign = 'left';
        ctx.fillText(`${row.label}  ${row.count}`, left + 12, y - 2);
        ctx.fillStyle = palette.muted; ctx.font = '9px Roboto Mono, monospace';
        ctx.fillText(row.note, left + 12, y + 13);

        const lineStart = left + leftWidth + 18;
        const lineEnd = right - statusWidth - 22;
        const lineGradient = ctx.createLinearGradient(lineStart, 0, lineEnd, 0);
        lineGradient.addColorStop(0, failed ? 'rgba(230,106,78,.16)' : 'rgba(43,156,148,.14)');
        lineGradient.addColorStop(1, failed ? 'rgba(230,106,78,.72)' : 'rgba(43,156,148,.58)');
        ctx.strokeStyle = lineGradient;
        ctx.lineWidth = failed ? 2 : 1.5;
        ctx.setLineDash(failed ? [6, 5] : []);
        ctx.beginPath(); ctx.moveTo(lineStart, y); ctx.lineTo(lineEnd, y); ctx.stroke();
        ctx.setLineDash([]);
        const phase = ((now / (failed ? 640 : 980)) + index * 0.27) % 1;
        for (let dot = 0; dot < 3; dot += 1) {
          const travel = (phase + dot / 3) % 1;
          const px = lineStart + travel * (lineEnd - lineStart);
          ctx.fillStyle = failed ? `rgba(230,106,78,${0.25 + travel * 0.65})` : `rgba(43,156,148,${0.18 + travel * 0.55})`;
          ctx.beginPath(); ctx.arc(px, y, failed ? 2.2 : 1.8, 0, Math.PI * 2); ctx.fill();
        }

        const pulse = 0.5 + 0.5 * Math.sin(now / 480 + index);
        ctx.strokeStyle = failed ? 'rgba(230,106,78,.22)' : 'rgba(43,156,148,.20)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(center, y, 15 + pulse * 5, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = failed ? palette.coral : palette.teal;
        ctx.beginPath(); ctx.arc(center, y, 5.5, 0, Math.PI * 2); ctx.fill();

        roundedRect(ctx, right - statusWidth, y - 16, statusWidth, 34, 9);
        ctx.fillStyle = failed ? 'rgba(230,106,78,.10)' : 'rgba(43,156,148,.08)';
        ctx.fill();
        ctx.fillStyle = stateColor; ctx.font = '600 11px Noto Sans SC, sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(failed ? `泄漏 × ${row.leaks}` : '通过', right - statusWidth / 2, y + 4);
      });

      const leaks = [
        ...analysis.hardNegativeLeakages.map(item => ({ ...item, source: '硬负' })),
        ...analysis.backgroundLeakages.map(item => ({ ...item, source: '背景' })),
      ];
      ctx.textAlign = 'left'; ctx.font = '10px Roboto Mono, monospace';
      if (leaks.length) {
        ctx.fillStyle = palette.coral;
        ctx.fillText(`已拦截：${leaks.slice(0, 4).map(item => `${item.source}「${item.phrase}」`).join(' · ')}`, left, height - 16);
      } else {
        ctx.fillStyle = palette.teal;
        ctx.fillText('全部路径通过：负样本没有混入已注册词。', left, height - 16);
      }
      ctx.textAlign = 'right'; ctx.fillStyle = palette.muted;
      ctx.fillText(analysis.conflicts.length ? `另有 ${analysis.conflicts.length} 对子串碰撞` : '扫描中', right, height - 16);
    }

    function refreshPlan() {
      const words = lexicon();
      if (!words.length) { $('#heatmap-container').innerHTML = '<div class="xray-item">请输入至少一个词。</div>'; $('#xray-report').innerHTML = ''; return; }
      const records = buildRecords(words), analysis = analyze(words, records);
      state.contract = { words, records, analysis };
      const alerts = analysis.conflicts.length + analysis.hardNegativeLeakages.length + analysis.backgroundLeakages.length;
      $('#contract-summary').textContent = alerts ? `${alerts} 条路径需要处理` : '全部路径通过';
      renderHeatmap(words, analysis); renderXray(words, records, analysis); drawContractFrame();
    }

    $('#plan-mode-group')?.addEventListener('click', event => {
      state.manualUntil = performance.now() + 12000;
      const button = event.target.closest('[data-pmode]');
      if (!button) return;
      state.mode = button.dataset.pmode; refreshPlan();
    });
    textarea?.addEventListener('input', () => { state.manualUntil = performance.now() + 12000; refreshPlan(); });
    $('#btn-plan-run')?.addEventListener('click', () => { state.manualUntil = performance.now() + 12000; refreshPlan(); });
    $('#btn-plan-9')?.addEventListener('click', () => { state.manualUntil = performance.now() + 12000; textarea.value = state.defaultWords.slice(0, 9).join('\n'); refreshPlan(); });
    $('#btn-plan-10')?.addEventListener('click', () => { state.manualUntil = performance.now() + 12000; textarea.value = state.defaultWords.join('\n'); state.injected = false; refreshPlan(); });
    $('#btn-plan-leak')?.addEventListener('click', () => {
      state.manualUntil = performance.now() + 16000;
      state.injected = true;
      const current = lexicon();
      if (!current.includes('灯')) current.push('灯');
      if (!current.includes('开灯')) current.push('开灯');
      textarea.value = current.join('\n'); refreshPlan();
    });
    registerAmbientDrawer('plan', (now) => {
      if (!state.lastAutoCycle) {
        state.lastAutoCycle = now;
        return;
      }
      if (now > state.manualUntil && now - state.lastAutoCycle > 2800) {
        state.mode = state.mode === 'v3' ? 'naive_mod' : 'v3';
        setActiveSegment('#plan-mode-group', 'pmode', state.mode);
        state.lastAutoCycle = now;
        refreshPlan();
      }
      drawContractFrame(now);
    });
    refreshPlan();
  }

  async function boot() {
    try {
      const response = await fetch('data/synthbench_bundle.json', { cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      app.bundle = await response.json();
      initDsp(app.bundle); initDecoder(app.bundle); initEvaluation(app.bundle); initPlan(app.bundle);
      document.body.dataset.ready = 'true';
      installScrollStory();
      startAmbientLoop();
    } catch (error) {
      console.error(error);
      const notice = document.createElement('div');
      notice.className = 'boot-error';
      notice.innerHTML = `<strong>Studio 数据载入失败</strong><br>${error.message}<br>请通过本地 HTTP 服务打开页面，例如 <code>python -m http.server 8765 --directory studio</code>。`;
      document.querySelector('main')?.prepend(notice);
    }
  }
  boot();
})();
