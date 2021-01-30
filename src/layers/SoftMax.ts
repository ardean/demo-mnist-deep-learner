import Layer from "./Layer";
import * as mathjs from "mathjs";
import * as config from "../config";
import * as util from "../utils/util";

export default class SoftMax implements Layer {
  biases: number[];
  weights: any;
  inputLength: number;
  outputLength: number;
  filterNum: number;

  _inputShape: number[];
  _outputLength: number;
  _input: number[];
  _totals: number[];

  constructor(inputLength: number, outputLength: number, filterNum: number) {
    this.filterNum = filterNum;
    this.weights = mathjs.divide(mathjs.random([inputLength, outputLength]), inputLength);
    this.biases = (mathjs.zeros(outputLength)).toArray();
    this.inputLength = inputLength;
    this.outputLength = outputLength;
  }

  forward(input: number[][][]) {
    const flattened: number[] = [];

    this._inputShape = [input.length, input[0].length, input[0][0].length];
    this._outputLength = this.outputLength;

    input.forEach(array => {
      flattened.push(...mathjs.flatten(array));
    });

    this._input = flattened;

    const totals = mathjs.add(mathjs.multiply(flattened, this.weights), this.biases);
    this._totals = totals;

    const expValues = mathjs.exp(totals);
    return mathjs.divide(expValues, mathjs.sum(expValues));
  }

  backprop(gradient: number[], learningRate: number) {
    for (let i = 0; i < gradient.length; i++) {
      if (gradient[i] === 0) continue;

      const totalsExp = mathjs.exp(this._totals as number[]) as number[];
      const totalsExpSum = mathjs.sum(totalsExp);

      let d_out_d_t = mathjs.divide(mathjs.multiply(-totalsExp[i], totalsExp), totalsExpSum ** 2) as number[];
      d_out_d_t[i] = totalsExp[i] * (totalsExpSum - totalsExp[i]) / (totalsExpSum ** 2);

      let d_t_d_w = this._input;
      let d_t_d_b = 1;
      let d_t_d_inputs = this.weights;

      let d_L_d_t = mathjs.multiply(gradient[i], d_out_d_t) as number[];

      let d_L_d_w = mathjs.multiply(mathjs.transpose([d_t_d_w]), [d_L_d_t]) as number[][];
      let d_L_d_b = mathjs.multiply(d_L_d_t, d_t_d_b);
      let d_L_d_inputs = mathjs.multiply(d_t_d_inputs, d_L_d_t);

      this.weights = mathjs.subtract(this.weights, mathjs.multiply(learningRate, d_L_d_w));
      this.biases = mathjs.subtract(this.biases, mathjs.multiply(learningRate, d_L_d_b)) as number[];

      return mathjs.reshape(d_L_d_inputs, [13, 13, this.filterNum]);
    }
  }

  async saveToFile() {
    await Promise.all([
      util.writeJSON(`${config.trainingFolder}/biases.json`, this.biases),
      util.writeJSON(`${config.trainingFolder}/weights.json`, this.weights)
    ]);
  }

  async readFromFile() {
    const [biases, weights] = await Promise.all([
      util.readJSON(`${config.trainingFolder}/biases.json`),
      util.readJSON(`${config.trainingFolder}/weights.json`)
    ]);

    this.biases = biases;
    this.weights = weights;
  }
}