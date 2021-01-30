import Layer from "./Layer";
import * as mathjs from "mathjs";

export default class MaxPool implements Layer {
  _input: number[][][];
  constructor() { }

  listRegions(image: number[][][]) {
    const height = Math.floor(image.length / 2);
    const width = Math.floor(image[0].length / 2);

    const regions = [];
    for (let i = 0; i < height; i++) {
      for (let j = 0; j < width; j++) {
        const imgRegion = [];
        imgRegion.push([image[i * 2][j * 2], image[i * 2][j * 2 + 1]]);
        imgRegion.push([image[i * 2 + 1][j * 2], image[i * 2 + 1][j * 2 + 1]]);
        regions.push({ imgRegion, i, j });
      }
    }

    return regions;
  }

  forward(input: number[][][]): number[][][] {
    this._input = input;

    let output: number[][][] = [];
    const height = Math.floor(input.length / 2);
    const width = Math.floor(input[0].length / 2);
    const numFilters = input[0][0].length;

    for (let i = 0; i < height; i++) {
      const matrix = mathjs.zeros(width, numFilters);
      output.push(<number[][]>matrix.toArray());
    }

    const regions = this.listRegions(input);
    for (const value of regions) {
      const { imgRegion, i, j } = value;

      const maxes = [];
      for (let k = 0; k < 2; k++) {
        maxes.push(mathjs.max(imgRegion[k], 0));
      }
      output[i][j] = mathjs.max(maxes, 0);
    }
    return output;
  }

  backprop(d_L_d_out: number[][][]): number[][][] {
    let d_L_d_input: number[][][] = [];
    for (let i = 0; i < this._input.length; i++) {
      const matrix = mathjs.zeros(this._input[0].length, this._input[0][0].length);
      d_L_d_input.push(<number[][]>matrix.toArray());
    }

    const regions = this.listRegions(this._input);
    for (const value of regions) {
      const { imgRegion, i, j } = value;

      const height = imgRegion.length;
      const width = imgRegion[0].length;
      const numFilters = imgRegion[0][0].length;

      const maxes = [];
      for (let k = 0; k < 2; k++) {
        maxes.push(mathjs.max(imgRegion[k], 0));
      }

      const amax = mathjs.max(maxes, 0);

      for (let k = 0; k < height; k++) {
        for (let l = 0; l < width; l++) {
          for (let m = 0; m < numFilters; m++) {
            if (imgRegion[k][l][m] === amax[m]) {
              d_L_d_input[i * 2 + k][j * 2 + l][m] = d_L_d_out[i][j][m];
            }
          }
        }
      }
    }

    return d_L_d_input;
  }

  async readFromFile() { }

  async saveToFile() { }
}