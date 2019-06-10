import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as tf from '@tensorflow/tfjs';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

const NUM_CLASSES = 4;

const IMAGE_SIZE = 227;

const TOPK = 10;

const classes = ['RIght', 'Left', 'Down', 'Neutral'];

const knn = knnClassifier.create();

const video = document.getElementById('video') as HTMLVideoElement;

(async () => {
  const mobilenet = await mobilenetModule.load();

  const image = tf.browser.fromPixels(video);

  const logits = mobilenet.infer(image);

  knn.addExample(logits, training);

  const res = await knn.predictClass(logits, TOPK);

  console.log(classes[res.classIndex]);

  image.dispose();
})();
