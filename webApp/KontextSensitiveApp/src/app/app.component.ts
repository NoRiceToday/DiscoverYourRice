import { Variable } from '@angular/compiler/src/render3/r3_ast';
import { ChangeDetectorRef, Component, ElementRef, OnInit, ViewChild } from '@angular/core';
import { interval } from 'rxjs';


enum labels {
  Sitting ='sitting',
  Standing = 'standing',
  Walking = 'walking'
}

var values : number[][] = [[], [], [], [], [], [], [], [], []];
declare function callPlayer(frame_id:string, func:string, args?:Array<string>): any;

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})

export class AppComponent {
  title = 'KontextSensitiveApp';
  currentLabel = labels.Sitting;
  RandomForestClassifier: any;
  
  firstTimeStamp : Date | undefined;

  constructor(private cdr:ChangeDetectorRef) {
  }
  
  ngOnInit() {
    values = [[], [], [], [], [], [], [], [], []];
    if (window.DeviceMotionEvent) {
      window.addEventListener('devicemotion', this.deviceMotionHandler, true);
    }
  }

 deviceMotionHandler(evt: DeviceMotionEvent) {
  // first get values
  var currValues = {
        alpha: evt.rotationRate!.alpha,
        beta: evt.rotationRate!.beta,
        gamma: evt.rotationRate!.gamma,
        x: evt.accelerationIncludingGravity!.x,
        x0: evt.acceleration!.x,
        y: evt.accelerationIncludingGravity!.y,
        y0: evt.acceleration!.y,
        z: evt.accelerationIncludingGravity!.z,
        z0: evt.acceleration!.z,
      };

  
  var currentTime = new Date();
  // this will gather the motion values
  gatherValues(evt, values);

  if (this.firstTimeStamp == null || this.firstTimeStamp == undefined) {
    this.firstTimeStamp = currentTime;
  }
  var timeDiffValue = timeDiff(this.firstTimeStamp, currentTime);

  // we want to gather data of atleast 1 second, which is the size of our trainings buckets
  if (timeDiffValue >= 1) {
    // calculate means and var's - reduce noise and also did it in our training
    var calcInput = calculateMeanAndVar(values);
      
      // now classify it
    if (calcInput != null) {
      var prediction = new this.RandomForestClassifier().predict(calcInput);

    if (prediction == 0) {
      this.currentLabel = labels.Sitting;
      updateSitting();
    } else if (prediction == 1) {
      this.currentLabel =  labels.Standing;
      updateStanding();
    } else {
      this.currentLabel =  labels.Walking;
      updateWalking();
    }
    
    // reset for next measurement
    this.firstTimeStamp = undefined;
    values =  [[], [], [], [], [], [], [], [], []];
    }
  }
}
}


function updateSitting() {
  callPlayer('youtubeVideo', 'playVideo');
  //console.log('Video should have started');
}

function updateStanding() {
  // standing should do nothing to the video.
}

function updateWalking() {
  callPlayer('youtubeVideo', 'pauseVideo')
  //console.log('Video should have paused')
}

function gatherValues(evt: DeviceMotionEvent | null, values: number[][]) {
  if (evt != null) {
    if (evt.rotationRate != null) {
      values[0].push(evt.rotationRate.alpha as number);
      values[1].push(evt.rotationRate.beta as number);
      values[2].push(evt.rotationRate.gamma as number);
    }
    if (evt.accelerationIncludingGravity != null) {
      values[3].push(evt.accelerationIncludingGravity.x as number);
      values[5].push(evt.accelerationIncludingGravity.y as number);
      values[7].push(evt.accelerationIncludingGravity.z as number);
    }
    if (evt.acceleration != null) {
      values[4].push(evt.acceleration.x as number);
      values[6].push(evt.acceleration.y as number);
      values[8].push(evt.acceleration.z as number);
    }
  }
}

function timeDiff(timeStamp1 : Date, timeStamp2 : Date){
  return Math.floor((timeStamp2.getTime() - timeStamp1.getTime()) / 1000);
}

function calculateMeanAndVar(inputValues: number[][]) {
  var meanAndVarArray = new Array(18);

  for (var i = 0; i < meanAndVarArray.length; i++) {
    meanAndVarArray[i] = 0.0;
  }

  // calculate mean
  for (let i = 0; i < inputValues.length; i++) {
    meanAndVarArray[i] = inputValues[i].reduce((a: number,b: number) => a + b) / inputValues[i].length;
  }

  // calculate variance
  for (let i = 0; i < inputValues.length; i++) {
    meanAndVarArray[i + 8] =
      inputValues[i].reduce(
        (a: number, b: number) =>
          Math.pow(a - meanAndVarArray[i], 2) + Math.pow(b - meanAndVarArray[i], 2)
      ) / inputValues[i].length;
  }

  return meanAndVarArray;

}


