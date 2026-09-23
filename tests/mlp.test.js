const test = require("node:test");
const assert = require("node:assert/strict");
const {
  gauss, clamp, makeMLP, forward, trainStep, synthesizeDataset
} = require("../assets/js/mlp.js");

test("clamp bounds a value into [a, b]", () => {
  assert.equal(clamp(5, 0, 1), 1);
  assert.equal(clamp(-5, 0, 1), 0);
  assert.equal(clamp(0.5, 0, 1), 0.5);
});

test("gauss produces roughly the requested mean over many samples", () => {
  const n = 20000;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += gauss(10, 2);
  const mean = sum / n;
  assert.ok(Math.abs(mean - 10) < 0.15, `sample mean ${mean} too far from 10`);
});

test("makeMLP builds a network with the requested shape", () => {
  const net = makeMLP(4, 8, 1);
  assert.equal(net.W1.length, 8);
  assert.equal(net.W1[0].length, 4);
  assert.equal(net.W2.length, 1);
  assert.equal(net.W2[0].length, 8);
  assert.equal(net.b1.length, 8);
  assert.equal(net.b2.length, 1);
});

test("forward returns an output in (0, 1) via the sigmoid head", () => {
  const net = makeMLP(4, 8, 1);
  const { o, h } = forward(net, [0.1, -0.2, 0.3, -0.4]);
  assert.equal(h.length, 8);
  assert.equal(o.length, 1);
  assert.ok(o[0] > 0 && o[0] < 1);
});

test("trainStep reduces loss on a single fixed example over repeated steps", () => {
  const net = makeMLP(4, 8, 1);
  const x = [1.5, -1.2, 0.8, -0.9];
  const y = [1];
  let firstLoss = null;
  let lastLoss = null;
  for (let i = 0; i < 200; i++) {
    const loss = trainStep(net, x, y, 0.1);
    if (i === 0) firstLoss = loss;
    lastLoss = loss;
  }
  assert.ok(lastLoss < firstLoss, `loss should drop: ${firstLoss} -> ${lastLoss}`);
  assert.ok(lastLoss < 0.05, `loss should converge low, got ${lastLoss}`);
});

test("synthesizeDataset returns n samples with matching, valid labels", () => {
  const { X, Y } = synthesizeDataset(120);
  assert.equal(X.length, 120);
  assert.equal(Y.length, 120);
  X.forEach((vec) => assert.equal(vec.length, 4));
  Y.forEach((label) => assert.ok(label[0] === 0 || label[0] === 1));
});

test("a network trained on the synthetic dataset beats a coin flip on holdout", () => {
  const { X, Y } = synthesizeDataset(400);
  const trainX = X.slice(0, 320), trainY = Y.slice(0, 320);
  const testX = X.slice(320), testY = Y.slice(320);
  const net = makeMLP(4, 8, 1);
  for (let epoch = 0; epoch < 120; epoch++) {
    for (let i = 0; i < trainX.length; i++) trainStep(net, trainX[i], trainY[i], 0.06);
  }
  let correct = 0;
  testX.forEach((x, i) => {
    const { o } = forward(net, x);
    if ((o[0] > 0.5 ? 1 : 0) === testY[i][0]) correct++;
  });
  const accuracy = correct / testX.length;
  assert.ok(accuracy > 0.75, `expected holdout accuracy > 0.75, got ${accuracy}`);
});
