/**
 * PulseGrid — core detector algorithm.
 *
 * Deliberately dependency-free and DOM-free: a Gaussian sampler, a tiny
 * 4 -> N -> 1 feed-forward network, and hand-written backpropagation.
 * Loaded as a plain <script> in index.html (attaches to window.PulseGridMLP)
 * and required directly by the Node test suite in /tests/mlp.test.js —
 * the UMD wrapper below is what makes both work from the same file.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.PulseGridMLP = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  /** Box-Muller Gaussian sample with the given mean and standard deviation. */
  function gauss(mean, std) {
    const u = 1 - Math.random();
    const v = Math.random();
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    return mean + z * std;
  }

  /** Clamp v into [a, b]. */
  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  /** He-ish random weight init, scaled by fan-in. */
  function randInit(n) {
    return (Math.random() * 2 - 1) * Math.sqrt(1 / n);
  }

  /** Build a fresh nIn -> nHid -> nOut network with random weights. */
  function makeMLP(nIn, nHid, nOut) {
    return {
      nIn, nHid, nOut,
      W1: Array.from({ length: nHid }, () => Array.from({ length: nIn }, () => randInit(nIn))),
      b1: Array.from({ length: nHid }, () => 0),
      W2: Array.from({ length: nOut }, () => Array.from({ length: nHid }, () => randInit(nHid))),
      b2: Array.from({ length: nOut }, () => 0)
    };
  }

  function tanh(x) { return Math.tanh(x); }
  function dtanh(y) { return 1 - y * y; }
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

  /** One forward pass. Returns both the hidden activations and the output. */
  function forward(net, x) {
    const h = net.W1.map((row, i) => tanh(row.reduce((s, w, j) => s + w * x[j], 0) + net.b1[i]));
    const o = net.W2.map((row, i) => sigmoid(row.reduce((s, w, j) => s + w * h[j], 0) + net.b2[i]));
    return { h, o };
  }

  /**
   * One stochastic-gradient-descent step: forward pass, MSE loss,
   * backpropagation, in-place weight update. Returns the sample's loss.
   */
  function trainStep(net, x, yTrue, lr) {
    const { h, o } = forward(net, x);
    const dOut = o.map((oi, i) => (oi - yTrue[i]) * oi * (1 - oi));
    const dHid = h.map((hi, i) => {
      let s = 0;
      for (let k = 0; k < net.nOut; k++) s += dOut[k] * net.W2[k][i];
      return s * dtanh(hi);
    });
    for (let i = 0; i < net.nOut; i++) {
      for (let j = 0; j < net.nHid; j++) net.W2[i][j] -= lr * dOut[i] * h[j];
      net.b2[i] -= lr * dOut[i];
    }
    for (let i = 0; i < net.nHid; i++) {
      for (let j = 0; j < net.nIn; j++) net.W1[i][j] -= lr * dHid[i] * x[j];
      net.b1[i] -= lr * dHid[i];
    }
    return o.reduce((s, oi, i) => s + (oi - yTrue[i]) ** 2, 0) / o.length;
  }

  /**
   * Generate n labelled synthetic samples in the same normalised feature
   * space the live mesh uses: 4 features, each ~N(0, 0.35) when normal;
   * 1-3 features shifted by ~N(±2.4, 0.6) when anomalous.
   */
  function synthesizeDataset(n) {
    const X = [], Y = [];
    for (let i = 0; i < n; i++) {
      const anomalous = Math.random() < 0.5;
      const vec = [gauss(0, 0.35), gauss(0, 0.35), gauss(0, 0.35), gauss(0, 0.35)];
      if (anomalous) {
        const nShift = 1 + Math.floor(Math.random() * 3);
        const idxs = [0, 1, 2, 3].sort(() => Math.random() - 0.5).slice(0, nShift);
        idxs.forEach((k) => { vec[k] = (Math.random() < 0.5 ? -1 : 1) * gauss(2.4, 0.6); });
      }
      X.push(vec);
      Y.push([anomalous ? 1 : 0]);
    }
    return { X, Y };
  }

  return { gauss, clamp, randInit, makeMLP, tanh, dtanh, sigmoid, forward, trainStep, synthesizeDataset };
});
