import express from "express";
import * as config from "./config";
import bodyParser from "body-parser";
import ConvolutionalNeuralNetwork from "./ConvolutionalNeuralNetwork";

export default (convolutionalNeuralNetwork: ConvolutionalNeuralNetwork) => {
  const app = express();

  app.use(bodyParser.json());
  app.use(express.static(`${__dirname}/../public`));

  app.post("/api/train", async (req: express.Request, res: express.Response, next: express.NextFunction) => {
    try {
      console.info("Training started.");
      await convolutionalNeuralNetwork.train();

      res.sendStatus(200);
    } catch (err) {
      next(err);
    }
  });

  app.post("/api/test", async (req: express.Request, res: express.Response, next: express.NextFunction) => {
    try {
      console.info("Testing started.");
      await convolutionalNeuralNetwork.test();

      res.sendStatus(200);
    } catch (err) {
      next(err);
    }
  });

  app.post("/api/predict", async (req: express.Request, res: express.Response, next: express.NextFunction) => {
    try {
      const { image } = req.body;
      const predictedNumber = convolutionalNeuralNetwork.predict(image).predicted;
      res.json(predictedNumber);
    } catch (err) {
      next(err);
    }
  });

  return new Promise<void>(resolve => {
    app.listen(config.port, () => {
      console.log(`App listening http://localhost:${config.port}`);
      resolve();
    });
  });
};