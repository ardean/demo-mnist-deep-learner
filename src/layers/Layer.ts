export default interface Layer {
  forward(input: any): any;

  backprop(gradient: any, learningRate: number): any;

  readFromFile(): Promise<void>;
  saveToFile(): Promise<void>;
}