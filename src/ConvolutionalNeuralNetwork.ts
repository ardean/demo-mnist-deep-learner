import fs from "fs";
import Image from "./Image";
import * as mathjs from "mathjs";
import Layer from "./layers/Layer";
import Conv2D from "./layers/Conv2D";
import MaxPool from "./layers/MaxPool";
import SoftMax from "./layers/SoftMax";

export default class ConvolutionalNeuralNetwork {
  printInterval = 0;
  imagesCount = 100;
  epochCount = 5 * 1;
  trainingImages = this.loadTrainData(this.imagesCount);
  testImages = this.loadTrainData(this.imagesCount);
  filtersCount = 12;

  layers: Layer[] = [
    new Conv2D(this.filtersCount),
    new MaxPool(),
    new SoftMax(13 * 13 * this.filtersCount, 10, this.filtersCount)
  ];

  reversedLayers = this.layers.concat().reverse();

  forward(image: number[][], label: number) {
    const { output } = this.predict(image);
    const loss = -mathjs.log(output[label])
    const acc = output.indexOf(Math.max(...output)) === label ? 1 : 0;
    return { output, loss, acc };
  }

  predict(image: number[][]) {
    const normalizedImage = mathjs.subtract(mathjs.divide(image, 255), 0.5);

    let output = normalizedImage;
    for (const layer of this.layers) {
      output = layer.forward(output);
    }

    const predicted = output.indexOf(Math.max(...output));

    return { output, predicted };
  }

  trainImage(image: number[][], label: number, learningRate: number = 0.005) {
    const { output, loss, acc } = this.forward(image, label);

    let gradient: any = mathjs.zeros(10).toArray() as number[];
    gradient[label] = -1 / output[label];

    for (const layer of this.reversedLayers) {
      gradient = layer.backprop(gradient, learningRate);
    }

    return { loss, acc };
  }

  shuffle(list: any[]) {
    for (let index = list.length - 1; index > 0; index--) {
      const randomIndex = Math.floor(Math.random() * (index + 1));
      [list[index], list[randomIndex]] = [list[randomIndex], list[index]];
    }
    return list;
  }

  async train() {
    let totalLoss = 0;
    let correctDigits = 0;

    for (let epochIndex = 0; epochIndex < this.epochCount; epochIndex++) {
      console.log(`training epoch ${epochIndex + 1}`);

      this.shuffle(this.trainingImages);
      this.trainingImages.map((data, index) => {
        const { image, label } = data;
        const { loss, acc } = this.trainImage(image, label, 0.01);

        totalLoss += loss;
        correctDigits += acc;

        if (index % this.printInterval === this.printInterval - 1) {
          console.log(`[Step ${index + 1}] Past ${this.printInterval} steps: 
                                Average Loss ${totalLoss / this.printInterval} | 
                                Accuracy: ${correctDigits / this.printInterval * 100}%`);
          totalLoss = 0
          correctDigits = 0;
        }
      });
    }

    await Promise.all(this.layers.map(layer => layer.saveToFile()));
  }

  async test() {
    await Promise.all(this.layers.map(layer => layer.readFromFile()));

    let totalLoss = 0;
    let correctDigits = 0;

    this.testImages.map(data => {
      const { image, label } = data;
      const { loss, acc } = this.forward(image, label);

      totalLoss += loss;
      correctDigits += acc;
    });

    console.log(`Test loss: ${totalLoss / this.testImages.length} \nTest accuracy: ${correctDigits / this.testImages.length * 100}%`);
  }

  loadTrainData(amount: number) {
    const dataFileBuffer = fs.readFileSync(__dirname + "/../data/train-images-idx3-ubyte");
    const labelFileBuffer = fs.readFileSync(__dirname + "/../data/train-labels-idx1-ubyte");

    const images: Image[] = [];

    for (let image = 0; image < amount; image++) {
      var imagePixels = [];

      for (var y = 0; y < 28; y++) {
        let row = [];
        for (var x = 0; x < 28; x++) {
          row.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16]);
        }
        imagePixels.push(row);
      }

      images.push({
        image: imagePixels,
        label: labelFileBuffer[image + 8]
      });
    }

    return images;
  }
}