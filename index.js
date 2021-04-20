/**
 * Training is done on pre-trained mobilenet model for recoginizing images
 * Flow of data
 * images -> (mobilenet model) -> output -> (new_customized_model) -> classification
 */

let mobilenet;
let model;
const rps_data = new RPSData();                                     // RPS data class
const webcam = new Webcam(document.getElementById('wc'));           // webcam class in Webcam.js
var oneCount=0, twoCount=0, threeCount=0,fourCount =0,fiveCount = 0,sixCount = 0;
var startPredictionsInterval;
var time = 3000;
var timeleft = 3;
var computerScore=0, playerScore=0,finalPScore =0, finalCScore=0;
var images1 = [];
// index1 = 0;
images1[0] = "<div><img src='images/one.jpg' class='img-thumbnail' ></div>";
images1[1] = "<div><img src='images/two.jpg' class='img-thumbnail'></div>";
images1[2] = "<div><img src='images/three.jpg' class='img-thumbnail' ></div>";
images1[3] = "<div><img src='images/four.jpg' class='img-thumbnail' ></div>";
images1[4] = "<div><img src='images/five.jpg' class='img-thumbnail' ></div>";
images1[5] = "<div><img src='images/six.jpg' class='img-thumbnail' ></div>";

async function init(){
    await webcam.setup();
    mobilenet = await loadMobilenet();
    // loading weights takes time mean while we capture the images and throw away results to avoid lag
    tf.tidy(() => mobilenet.predict(webcam.capture()));

}

// loading pre-trained model
async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');                          // get output of 'conv_pw_13_relu' layer
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// new model - inputs are output of the pre-trained model
async function train(){
    rps_data.ys = null
    rps_data.convertLabels(6);

    model = tf.sequential({
        layers:[
            tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
            tf.layers.dense({ units: 100, activation: 'relu'}),
            tf.layers.dense({ units: 6, activation: 'softmax'})                 // classfies into three classes
        ]
    });

    model.compile({optimizer: tf.train.adam(0.0001), loss: 'categoricalCrossentropy', metrics: ['accuracy']});
    let loss = 0;
    model.fit(rps_data.xs, rps_data.ys, {
        epochs: 10,
        callbacks: {
            // onBatchEnd: async (batch, logs) => {
            //     console.log('loss ' + logs.loss.toFixed(5) + ' accuracy ' + logs.acc);
            // }
        }
    })
}

// getting data
function handleButton(element){
	switch(element.id){
		case "0":
			oneCount++;
      while (oneCount < 50) {
        const image = webcam.capture();
      	label = parseInt(element.id);
      	rps_data.addImage(mobilenet.predict(image), label);
        oneCount++;
      }
      $("#0").removeClass("btn-warning").addClass("btn-success");
			// document.getElementById("one").innerText = "One Samples:" + oneCount;
			break;
		case "1":
			twoCount++;
      while (twoCount < 50) {
        const image = webcam.capture();
      	label = parseInt(element.id);
      	rps_data.addImage(mobilenet.predict(image), label);
        twoCount++;
      }
      $("#1").removeClass("btn-warning").addClass("btn-success");
			// document.getElementById("two").innerText = "Two Samples:" + twoCount;
			break;
		case "2":
			threeCount++;
      while (threeCount < 50) {
        const image = webcam.capture();
      	label = parseInt(element.id);
      	rps_data.addImage(mobilenet.predict(image), label);
        threeCount++;
      }
      $("#2").removeClass("btn-warning").addClass("btn-success");
			// document.getElementById("three").innerText = "Three Samples:" + threeCount;
			break;
    case "3":
			fourCount++;
      while (fourCount < 50) {
        const image = webcam.capture();
      	label = parseInt(element.id);
      	rps_data.addImage(mobilenet.predict(image), label);
        fourCount++;
      }
      $("#3").removeClass("btn-warning").addClass("btn-success");
			// document.getElementById("four").innerText = "Four Samples:" + fourCount;
			break;

    case "4":
			fiveCount++;
      while (fiveCount < 50) {
        const image = webcam.capture();
      	label = parseInt(element.id);
      	rps_data.addImage(mobilenet.predict(image), label);
        fiveCount++;
      }
      $("#4").removeClass("btn-warning").addClass("btn-success");
			// document.getElementById("five").innerText = "Five Samples:" + fiveCount;
			break;

    case "5":
      sixCount++;
      while (sixCount < 50) {
        const image = webcam.capture();
        label = parseInt(element.id);
        rps_data.addImage(mobilenet.predict(image), label);
        sixCount++;
      }
      $("#5").removeClass("btn-warning").addClass("btn-success");
      // document.getElementById("six").innerText = "Six Samples:" + sixCount;
      break;
	}


    // const image = webcam.capture();
  	// label = parseInt(element.id);
  	// rps_data.addImage(mobilenet.predict(image), label);         // passing image into mobilenet model and getting output weights #transfer learning
}

// making predictions
async function predicts(makePrediction) {
    if(makePrediction){
        startPredictionsInterval = setInterval(function() {
            doPredictions();
        }, time);
    } else clearInterval(startPredictionsInterval);

}


var previousClass = -1;
async function doPredictions(){
    const predictClassRPS = tf.tidy(() => {
        const image = webcam.capture();
        const mobilenetOutput = mobilenet.predict(image);
        const prediction = model.predict(mobilenetOutput);
        // console.log(prediction);
        return prediction.as1D().argMax();      // return 1D tensor containing prediction
    });

    // players Move
    const predictedClassID = (await predictClassRPS.data())[0];
    var text = "";
    switch (predictedClassID) {
        case 0:
            text = "1";
            break;
        case 1:
            text = "2";
            break;
        case 2:
            text = "3";
            break;
        case 3:
            text = "4";
            break;
        case 4:
            text = "5";
            break;
        case 5:
            text = "6";
            break;

    }
    if (predictedClassID != previousClass){
        document.getElementById("prediction").innerText = text;

        // computers Move
        var computersMove = Math.floor((Math.random() * 6));
        // setRandom(computersMove);
        document.getElementById("numberImg").innerHTML=images1[computersMove];
        document.getElementById("numberTxt").textContent = computersMove +1;
        // get winner
        var result = getWinner(predictedClassID, computersMove);
        document.getElementById("winner").innerText = result;

        // score
        if(result.localeCompare("PLAYER")) computerScore++;
        if(computerScore==3){
          alert('You Have Lost All Your Wickets');
          finalPScore = playerScore;
          document.getElementById("fpscore").innerText = finalPScore;
          startAgain();
        }

        if(result.localeCompare("COMPUTER")) playerScore+= predictedClassID+1;
        document.getElementById("pscore").innerText = playerScore; //+ computerScore + " / " + "playerScore " + playerScore ;
        document.getElementById("cscore").innerText = computerScore;
    }
    // previousClass = predictedClassID;

    // dispose predicted class
    predictClassRPS.dispose();
    await tf.nextFrame();
}


function getWinner(playersMove, computersMove){
    console.log(playersMove, computersMove);

    if(playersMove == computersMove) return "COMPUTER";
    else {
      return "PLAYER";
    }
    // if(playersMove == 0 && computersMove == 2) return "PLAYER";
    // if(playersMove == 1 && computersMove == 0) return "PLAYER";
    // if(playersMove == 2 && computersMove == 1) return "PLAYER";
    // if(playersMove == 0 && computersMove == 1) return "COMPUTER";
    // if(playersMove == 1 && computersMove == 2) return "COMPUTER" ;
    // if(playersMove == 2 && computersMove == 0) return "COMPUTER";

}

function doTraining(){
    if (oneCount > 0 && twoCount > 0 && threeCount > 0 && fourCount>0 && fiveCount>0 && sixCount>0){
        train();
        alert('All Data Sent ');
    } else alert('Capture Some Data');
}

function startPredicting(){
  if (oneCount > 0 && twoCount > 0 && threeCount > 0 && fourCount>0 && fiveCount>0 && sixCount>0)
	{predicts(true);
  start();}
  else{
    alert('Send Some Data');
  }
}

function stopPredicting(){
	predicts(false);
  stop();
  document.getElementById("countdowntimer").textContent = 0;
}

function resetAll(){
    stopPredicting(false);
    oneCount=0;
    twoCount=0;
    threeCount=0;
    fourCount=0;
    fiveCount=0;
    sixCount=0;
    finalPScore = 0;
    finalCScore = 0;
    document.getElementById("fpscore").innerText = finalPScore
    // document.getElementById("one").innerText = "One samples:" + oneCount;
    // document.getElementById("two").innerText = "Two samples:" + twoCount;
    // document.getElementById("three").innerText = "Three samples:" + threeCount;
    // document.getElementById("four").innerText = "Four samples:" + fourCount;
    // document.getElementById("five").innerText = "Five samples:" + fiveCount;
    // document.getElementById("six").innerText = "Six samples:" + sixCount;
    document.getElementById("prediction").innerText = "";
    document.getElementById("pscore").innerText = "";
    document.getElementById("cscore").innerText = "";
    document.getElementById("winner").innerText = "";
    $("#0").removeClass("btn-success").addClass("btn-warning");
    $("#1").removeClass("btn-success").addClass("btn-warning");
    $("#2").removeClass("btn-success").addClass("btn-warning");
    $("#3").removeClass("btn-success").addClass("btn-warning");
    $("#4").removeClass("btn-success").addClass("btn-warning");
    $("#5").removeClass("btn-success").addClass("btn-warning");
    // setRandom("");
    document.getElementById("numberTxt").textContent = 0;
    document.getElementById("countdowntimer").textContent = timeleft;

    init();
}

function startAgain(){
  stopPredicting(false);
  document.getElementById("countdowntimer").textContent = 0;
  document.getElementById("pscore").innerText = "";
  document.getElementById("cscore").innerText = "";
  playerScore = 0;
  computerScore = 0;
  document.getElementById("fpcore").innerText = "0";
}


function setRandom(msg){

    document.getElementById("random").innerText = msg;
}

function start(){
    scoreTimer = setInterval(countDown,1000);
}

function stop(){
    clearInterval(scoreTimer);
}



function countDown(){
timeleft--;
document.getElementById("countdowntimer").textContent = timeleft;
if(timeleft <= 0)
    timeleft = 3;
}

init();
