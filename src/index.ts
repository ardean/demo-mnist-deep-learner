import initApp from "./initApp";
import ConvolutionalNeuralNetwork from "./ConvolutionalNeuralNetwork";

(async () => {
  const convolutionalNeuralNetwork = new ConvolutionalNeuralNetwork();
  await initApp(convolutionalNeuralNetwork);
})();