/*
* Copyright (c) 2015 Samsung Electronics Co., Ltd.
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are
* met:
*
* * Redistributions of source code must retain the above copyright
* notice, this list of conditions and the following disclaimer.
* * Redistributions in binary form must reproduce the above
* copyright notice, this list of conditions and the following disclaimer
* in the documentation and/or other materials provided with the
* distribution.
* * Neither the name of Samsung Electronics Co., Ltd. nor the names of its
* contributors may be used to endorse or promote products derived from
* this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
* OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
* SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
* LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
* DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
* THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

var SAAgent,
    SASocket,
    connectionListener,
    responseTxt = document.getElementById("responseTxt");

var counter = 0;

var ppg;
var ppg_time;
var hrm;
var rr_interval;

var ppg_arr = new Array();
var ptime_arr = new Array();
var rr_arr = new Array();
var hrm_arr = new Array();
var count_arr = new Array();
var selected_rr = new Array();

/* Make Provider application running in background */
//console.log('Heart Rate: here i am');
//tizen.systeminfo.getCapability();
//tizen.humanactivitymonitor.start('HRM', onchangedCB);

tizen.application.getCurrentApplication().hide();
//tizen.power.request('SCREEN', 'CPU_AWAKE');
tizen.power.request('SCREEN', 'SCREEN_NORMAL');
//tizen.power.request("CPU", "CPU_AWAKE");


var HRMrawsensor = tizen.sensorservice.getDefaultSensor("HRM_RAW");

/*
	setTimeout(function(){
		HRMrawsensor.getHRMRawSensorData(onGetSuccessCB, onerrorCB);
		tizen.humanactivitymonitor.getHumanActivityData('HRM', onsuccessCB2, onerrorCB);
	}, 1000); */


function onGetSuccessCB(sensorData)
{
    var date = new Date(Date.now());
	var hours = date.getHours();
	var minutes = "0" + date.getMinutes();
	var seconds = "0" + date.getSeconds();   	
	var ms = "0" +date.getMilliseconds();
	var Time = hours + ':' + minutes.substr(-2) + ':' + seconds.substr(-2) + ':' + ms.substr(-3);
	
	
	tizen.humanactivitymonitor.getHumanActivityData('HRM', onsuccessCB2, onerrorCB);
	//console.log("data:" + sensorData.lightIntensity + " timeStamp:"+ Time + " counter : " + counter);
	counter++;
	
	ppg_arr.push(sensorData.lightIntensity);
	ptime_arr.push(Time);
	count_arr.push(counter);
	
	/*
	if (counter % 20 == 0){
		console.log(ppg_arr.length, ": ", ppg_arr[ppg_arr.length-1], ptime_arr.length, ": ", ptime_arr[ptime_arr.length-1], count_arr.length, ":", count_arr[count_arr.length-1]), 
		console.log(hrm_arr.length, ": ", hrm_arr[hrm_arr.length-1], rr_arr.length, ": ", rr_arr[rr_arr.length-1]);
	}
*/
}


function onsuccessCB1()
{
   console.log("HRMRaw sensor start");
   HRMrawsensor.getHRMRawSensorData(onGetSuccessCB, onerrorCB);
   HRMrawsensor.setChangeListener(onGetSuccessCB, 300);
}


function onsuccessCB2(hrmInfo) {
    var rrInterval = hrmInfo.rRInterval;
    var hrm = hrmInfo.heartRate;
    console.log('Heart rate: ' + hrmInfo.heartRate + " counter : " + counter);
    console.log('Peak-to-peak interval: ' + rrInterval + ' milliseconds' + ' counter : ' + counter);


    
    
    if (hrm > 0){
        hrm_arr.push(hrm);
    }
    if (hrm_arr.length == 10){
        var result = hrm_arr.reduce(function add(sum, currValue) {
      	  return sum + currValue;
      	}, 0);
        
        var average = result / hrm_arr.length;
        
    	hrm_arr.push(average)
    	SASocket.sendData(SAAgent.channelIds[0], hrm_arr);
    	console.log("HRM has been sent");
    	hrm_arr = [];
    }

    rr_arr.push(rrInterval);
    if (selected_rr.length == 0 && rrInterval != 0){
        selected_rr.push(rrInterval);
        console.log(selected_rr.length);
        console.log('rr interval: ' + selected_rr[selected_rr.length-1] + ' length : ' + selected_rr.length);
        console.log("");
    }
    else{
    	if (selected_rr[selected_rr.length-1] != rrInterval && rrInterval != 0){
    		selected_rr.push(rrInterval);	
    		console.log("");
    		console.log('rr interval: ' + selected_rr[selected_rr.length-1] + ' length : ' + selected_rr.length);
    		console.log("");
    	}
    }
    
    if (selected_rr.length == 100){
    	SASocket.sendData(SAAgent.channelIds[0], selected_rr);
    	selected_rr = [];
    	console.log("100 RR intervals have been sent");
    }
    else if (selected_rr.length%10 == 0 && selected_rr.length != 0){
    	SASocket.sendData(SAAgent.channelIds[0], selected_rr);
    	selected_rr = [];
    	console.log("10 RR intervals have been sent");
    }
    
}

function onerrorCB(error) {
    console.log('Error occurred: ' + error.message);
}


function onchangedCB(hrmInfo) {	
	
    //console.log('Heart Rate: ' + hrmInfo.heartRate);
    //console.log('Peak-to-peak interval: ' + hrmInfo.rRInterval + ' milliseconds');
    
	/*
    counter++;
    if (counter > 1000) {
        // Stop the sensor after detecting a few changes 
        tizen.humanactivitymonitor.stop('HRM');
    }*/
   
}





function createHTML(log_string)
{
    var content = document.getElementById("toast-content");
    content.innerHTML = log_string;
    tau.openPopup("#toast");
}

connectionListener = {
    /* Remote peer agent (Consumer) requests a service (Provider) connection */
    onrequest: function (peerAgent) {

        createHTML("peerAgent: peerAgent.appName<br />" +
                    "is requsting Service conncetion...");

        /* Check connecting peer by appName*/
        if (peerAgent.appName === "HelloAccessoryConsumer") {
            SAAgent.acceptServiceConnectionRequest(peerAgent);
            createHTML("Service connection request accepted.");

        } else {
            SAAgent.rejectServiceConnectionRequest(peerAgent);
            createHTML("Service connection request rejected.");

        }
    },

    /* Connection between Provider and Consumer is established */
    onconnect: function (socket) {
        var onConnectionLost,
            dataOnReceive;

        createHTML("Service connection established");

        /* Obtaining socket */
        SASocket = socket;

        onConnectionLost = function onConnectionLost (reason) {
            createHTML("Service Connection disconnected due to following reason:<br />" + reason);
        };

        /* Inform when connection would get lost */
        SASocket.setSocketStatusListener(onConnectionLost);

        dataOnReceive =  function dataOnReceive (channelId, data) {

        	tizen.humanactivitymonitor.start('HRM', onchangedCB);
        	HRMrawsensor.start(onsuccessCB1);
        	
            if (!SAAgent.channelIds[0]) {
                createHTML("Something goes wrong...NO CHANNEL ID!");
                return;
            }
            //newData[0] = data + " :: " + new Date();
            //newData[1] = 'example1';

            /* Send new data to Consumer */
            //SASocket.sendData(SAAgent.channelIds[0], "");
           // createHTML("Send massage:<br />" + newData);
        };

        /* Set listener for incoming data from Consumer */
        SASocket.setDataReceiveListener(dataOnReceive);
    },
    onerror: function (errorCode) {
        createHTML("Service connection error<br />errorCode: " + errorCode);
    }
};

function requestOnSuccess (agents) {
    var i = 0;

    for (i; i < agents.length; i += 1) {
        if (agents[i].role === "PROVIDER") {
            createHTML("Service Provider found!<br />" +
                        "Name: " +  agents[i].name);
            SAAgent = agents[i];
            break;
        }
    }

    /* Set listener for upcoming connection from Consumer */
    SAAgent.setServiceConnectionListener(connectionListener);
};


function requestOnError (e) {
    createHTML("requestSAAgent Error" +
                "Error name : " + e.name + "<br />" +
                "Error message : " + e.message);
};

/* Requests the SAAgent specified in the Accessory Service Profile */
webapis.sa.requestSAAgent(requestOnSuccess, requestOnError);


(function () {
    /* Basic Gear gesture & buttons handler */
    window.addEventListener('tizenhwkey', function(ev) {
        var page,
            pageid;

        if (ev.keyName === "back") {
            page = document.getElementsByClassName('ui-page-active')[0];
            pageid = page ? page.id : "";
            if (pageid === "main") {
                try {
                    tizen.application.getCurrentApplication().exit();
                } catch (ignore) {
                }
            } else {
                window.history.back();
            }
        }
    });
}());

(function(tau) {
    var toastPopup = document.getElementById('toast');

    toastPopup.addEventListener('popupshow', function(ev){
        setTimeout(function () {
            tau.closePopup();
        }, 3000);
    }, false);
})(window.tau);
