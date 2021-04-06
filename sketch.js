// Daniel Shiffman
// http://codingtra.in
// http://patreon.com/codingtrain

// Linear Regression with TensorFlow.js
// Video: https://www.youtube.com/watch?v=dLp10CFIvxI

let x_vals;
let y_vals;
let check;

let m, b, slope;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

function initialize(){
  m = tf.variable(tf.scalar(random(1)));
  b = tf.variable(tf.scalar(random(1)));
  x_vals = [];
  y_vals = [];
  background(0);
  check = false;
}

function setup() {
  createCanvas(700, 700);
  submitButton = createButton('submit');
  submitButton.position(width + 100, 400, 65);
  submitButton.mousePressed(predictAndScore);

  clearButton = createButton('clear');
  clearButton.position(width + 100, 450, 65);
  clearButton.mousePressed(initialize);
  initialize();
}

function loss(pred, labels) {
  return pred.sub(labels).square().mean();
}

function predict(x) {
  const xs = tf.tensor1d(x);
  const ys = xs.mul(m).add(b);
  return ys;
}

function addToVals() {
  let x = map(mouseX, 0, width, 0, 1);
  let y = map(mouseY, 0, height, 1, 0);
  x_vals.push(x);
  y_vals.push(y);
}

function predictAndScore(){
  check = true;
  const lineX = [min(x_vals), max(x_vals)];
  // const lineX = [0, 1];

  const ys = tf.tidy(() => predict(lineX));
  let lineY = ys.dataSync();
  ys.dispose();

  let x1 = map(lineX[0], 0, 1, 0, width);
  let x2 = map(lineX[1], 0, 1, 0, width);

  let y1 = map(lineY[0], 0, 1, height, 0);
  let y2 = map(lineY[1], 0, 1, height, 0);

  let y1_coord = map(lineY[0], 0, 1, 0, height);
  let y2_coord = map(lineY[1], 0, 1, 0, height);

  if (x2 - x1 === 0) slope = 0;
  else slope = (y2_coord-y1_coord)/(x2 - x1);
  if (slope > 0.2) console.log(slope, "INCREASING");
  else if (slope < -0.2) console.log(slope, "DECREASING");
  else console.log(slope, "CONFUSING");

  strokeWeight(5);
  stroke(255, 0, 0);
  line(x1, y1, x2, y2);
}

function draw() {
  tf.tidy(() => {
    if (x_vals.length > 0) {
      const ys = tf.tensor1d(y_vals);
      optimizer.minimize(() => loss(predict(x_vals), ys));
    }
  });

  if (mouseIsPressed === true) {
    if (mouseX <= width) addToVals();
  }

  if (check) return;

  background(0);
  stroke(255);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], 0, 1, 0, width);
    let py = map(y_vals[i], 0, 1, height, 0);
    let prevx = map(x_vals[i-1], 0, 1, 0, width);
    let prevy = map(y_vals[i-1], 0, 1, height, 0);
    // strokeWeight(4);
    // line(px, py, prevx, prevy);
    strokeWeight(6);
    point(px, py);
  }
}
