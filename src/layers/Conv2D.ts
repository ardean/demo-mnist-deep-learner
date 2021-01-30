import Layer from "./Layer";
import * as mathjs from "mathjs";
import * as util from "../utils/util";
import * as config from "../config";

export default class Conv2D implements Layer {
  private numFilters: number;
  private filters: mathjs.MathArray;
  private _input: number[][];

  constructor(numFilters: number) {
    this.numFilters = numFilters;

    this.filters = mathjs.divide(mathjs.random([numFilters, 3, 3]), 9);
  }

  *iterateRegions(image: number[][]) {
    const height = image.length;
    const width = image[0].length;

    for (let i = 0; i < height - 2; i++) {
      for (let j = 0; j < width - 2; j++) {
        const imgRegion = [];
        imgRegion.push([image[i][j], image[i][j + 1], image[i][j + 2]]);
        imgRegion.push([image[i + 1][j], image[i + 1][j + 1], image[i + 1][j + 2]]);
        imgRegion.push([image[i + 2][j], image[i + 2][j + 1], image[i + 2][j + 2]]);
        yield { imgRegion, i, j }
      }
    }
  }

  forward(input) {
    this._input = input;
    const height = input.length;
    const width = input[0].length;
    let output: number[][][] = [];
    for (let i = 0; i < height - 2; i++) {
      const matrix = mathjs.zeros(width - 2, this.numFilters);
      output.push(matrix.toArray());
    }

    for (let value of this.iterateRegions(input)) {
      const { imgRegion, i, j } = value;
      output[i][j] = [];
      for (let k = 0; k < this.numFilters; k++) {
        output[i][j].push(mathjs.sum(mathjs.dotMultiply(imgRegion, this.filters[k])));
      }
    }

    return output as any;
  }

  backprop(d_L_d_out: number[][][], learningRate: number) {
    const d_L_d_filters: number[][][] = [];
    const filters: any = this.filters;
    const numFilters = filters.length;
    const width = filters[0].length;
    const height = filters[0][0].length;

    for (let i = 0; i < numFilters; i++) {
      const matrix = mathjs.zeros(width, height);
      d_L_d_filters.push(<number[][]>matrix.toArray());
    }

    for (const { imgRegion, i, j } of this.iterateRegions(this._input)) {
      for (let f = 0; f < this.numFilters; f++) {
        d_L_d_filters[f] = mathjs.add(d_L_d_filters[f], mathjs.multiply(d_L_d_out[i][j][f], imgRegion)) as number[][];
      }
    }

    const subtracted: any = [];

    for (let index = 0; index < d_L_d_filters.length; index++) {
      subtracted.push(mathjs.subtract(this.filters[index], mathjs.multiply(d_L_d_filters[index], learningRate)));
    }

    this.filters = subtracted;
  }

  async saveToFile() {
    await util.writeJSON(`${config.trainingFolder}/filters.json`, this.filters);
  }

  async readFromFile() {
    this.filters = await util.readJSON(`${config.trainingFolder}/filters.json`);
  }
}