import * as cocoSsd from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs-core';

(async () => {
  const model = await cocoSsd.load();

  const webcam = document.getElementById('webcam') as HTMLImageElement;

  const webcamImage = tf.browser.fromPixels(webcam);

  const batchedImage = webcamImage.expandDims(0);

  const processedImage = batchedImage
    .toFloat()
    .div(tf.scalar(127))
    .sub(tf.scalar(1));

  const predictions = await model.detect(processedImage as any); // tf.Tensor<tf.Rank.R3> !== tf.Tensor<tf.Tensor<ts.Rank>>

  console.log(predictions);
})();
