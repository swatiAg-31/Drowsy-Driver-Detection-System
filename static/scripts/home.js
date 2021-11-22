
function hide() {
  document.getElementById('loader').style.display = 'none';
};

function hide2() {
  document.getElementById('myPopup').style.display = 'none';
};

function display() {
  var x = document.getElementById("usr_video");
  var y = document.getElementById("manual_video");
  y.style.display = "none"; 
  x.style.display = "block"; 
}

function stop(){
  var x = document.getElementById("usr_video");
  x.src = "";
  var y = document.getElementById("manual_video");
  y.style.display = "block";
}
function submits() {
   alert("Submitted");
 }
