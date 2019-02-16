import * as tf from '@tensorflow/tfjs';
console.log(tf);

const video = document.getElementById("video");
const canvases = document.getElementsByClassName("canvases");
const status = document.getElementById("status");
const loss = document.getElementById("loss");
let truncatedMobilenet;
let model;
const numClasses = 4;
let xs;
let ys;

const draw = (image, canvas) => {
    const [width, height] = [224, 224];
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = image.dataSync();
    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        imageData.data[j + 0] = (data[i * 3 + 0] + 1) * 127;
        imageData.data[j + 1] = (data[i * 3 + 1] + 1) * 127;
        imageData.data[j + 2] = (data[i * 3 + 2] + 1) * 127;
        imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}
const canvasSetup = () => {
    Array.from(canvases).forEach(element => {
        element.addEventListener("click", (event) => {
            const canvas = event.target
            const counter = document.getElementById("counter" + canvas.id);
            counter.innerText = parseInt(counter.innerText) + 1;
            console.log("initializing canvas " + event.target.id);
            const img = capture();
            // console.log(typeof parseInt(canvas.id));
            addExample(truncatedMobilenet.predict(img), parseInt(canvas.id));
            draw(img, canvas);

        })
    });
}

const adjustVideoSize = (width, height) => {
    const aspectRatio = width / height;
    if (width >= height) {
        video.width = aspectRatio * video.height;
    } else if (width < height) {
        video.height = video.width / aspectRatio;
    }
}
const webcamSetup = () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true }).then(function (stream) {
            video.srcObject = stream;
            video.addEventListener('loadeddata', () => {
                adjustVideoSize(
                    video.videoWidth,
                    video.videoHeight);
                video.play();
            }, false);

            console.log("initializing webcam");
        });
    }
}

const loadTruncatedModel = () => {

    return tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json')
        .then((res) => {
            const mobilenet = res;
            const layer = mobilenet.getLayer('conv_pw_13_relu');
            return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
        })
        .catch((err) => {
            console.log(err);
        })
}
const cropImage = (img) => {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);

}
const capture = () => {
    return tf.tidy(() => {
        console.log(video);
        const webcamImage = tf.fromPixels(video);
        const croppedImage = cropImage(webcamImage);
        const batchedImage = croppedImage.expandDims(0);
        return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    })

}
const addExample = (example, label) => {
    const y = tf.tidy(() => tf.oneHot(tf.tensor1d([label]).toInt(), numClasses));
    console.log(xs);
    if (xs == null) {
        xs = tf.keep(example);
        ys = tf.keep(y);
    } else {
        const oldX = xs;
        xs = tf.keep(oldX.concat(example, 0));

        const oldY = ys;
        ys = tf.keep(oldY.concat(y, 0));

        oldX.dispose();
        oldY.dispose();
        y.dispose();
    }
}

const train = () => {

    model = tf.sequential({
        layers: [
            // Flattens the input to a vector so we can use it in a dense layer. While
            // technically a layer, this only performs a reshape (and has no training
            // parameters).
            tf.layers.flatten({ inputShape: truncatedMobilenet.outputs[0].shape.slice(1) }),
            tf.layers.dense({
                units: 100,
                activation: 'relu',
                kernelInitializer: 'varianceScaling',
                useBias: true
            }),
            // The number of units of the last layer should correspond
            // to the number of classes we want to predict.
            tf.layers.dense({
                units: numClasses,
                kernelInitializer: 'varianceScaling',
                useBias: false,
                activation: 'softmax'
            })
        ]
    });

    const optimizer = tf.train.adam(0.0001);
    model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });
    const batchSize =
        Math.floor(xs.shape[0] * 0.4);

    model.fit(xs, ys, {
        batchSize,
        epochs: 20,
        callbacks: {
            onBatchEnd: (batch, logs) => {
                loss.innerText = ('Loss: ' + logs.loss.toFixed(5));
            }
        }
    });

}

const start = () => {

    const predictedClass = tf.tidy(() => {
        const img = capture();
        const embedding = truncatedMobilenet.predict(img);
        const prediction = model.predict(embedding);
        return prediction.as1D().argMax();
    });

    let classID;
    predictedClass.data().then((data) => {
        Array.from(canvases).forEach((element) => {
            element.classList.remove("predicted");
        });

        classID = data[0];
        document.getElementById(classID).classList.add("predicted");
    });

    tf.nextFrame().then(() => {
        start();
    })
        .catch((err) => console.log(err));
}




const init = () => {
    canvasSetup();
    webcamSetup();
    status.innerText = "loading the truncated model";
    loadTruncatedModel()
        .then((res) => {
            console.log("model loaded successfully");
            truncatedMobilenet = res;
            const batchImage = capture();
            console.log(batchImage);
            const prediction = tf.tidy(() => truncatedMobilenet.predict(batchImage));
            console.log(prediction);
        })
        .catch((err) => {
            console.log(err);
        });
}

init();
