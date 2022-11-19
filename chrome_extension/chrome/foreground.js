//var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0
var prev_dict = {}
var current_width = 300
var questionid = document.querySelector("#question").getAttribute("data-questionid")
var oldTop;
var oldHeight;
var oldWidth;
// var first = true
// var done = false
window.setInterval(everyHalfSecond, 500)

// 创建Sidebar
var fixedContainer = document.createElement("div")
fixedContainer.id = "fixedContainer"

// To make it draggable
//fixedContainer.classList.add("fullmight")

// fixedContainer.setAttribute("style",`height:${document.querySelector('body').scrollHeight - document.querySelector('.top-bar').scrollHeight - 10}px`)

// Fake
var fakeContainer = document.createElement("div")
fakeContainer.id = "fakeContainer"
fixedContainer.setAttribute("style","overflow:hidden;")

// Now I want to read the sidebar.html
fetch('http://localhost:8000/sidebar', {
    headers: {
        'Access-Control-Allow-Origin': '*', 
        "Accept": "application/json", 
        "Content-Type": "application/json"}, 
    mode: 'cors'}).then(function(response) {
    return response.text();
  }).then(function(response) {
    // creates the header for the sidebar
    fixedContainer.innerHTML = response
    var origin = fixedContainer.querySelector("#origin")
    list_of_answers = ''
    fixedContainer.querySelector("#traffic-light-green").style.display = "none"
    // Check all the answers
    var url = "http://localhost:8000/"
    var allanswers = document.querySelectorAll(".answer")
    
    // splits the text
    allanswers.forEach((eachAnswer, index) => {
        text = eachAnswer.querySelector(".js-post-body")
        question = document.body.querySelector("h1").querySelector("a")
        var id = eachAnswer.id.split("-")[1]
        httpPost(url, "summarybonankou" + text.innerHTML + "bonankou" + question.innerHTML + "bonankou" + id, eachAnswer)
        var answer_time
        if (eachAnswer.querySelector(".user-action-time a")) {
            answer_time = `<div class="user-action-time">` + eachAnswer.querySelector(".user-action-time a").innerHTML + `</div>`
        } else {
            answer_time = eachAnswer.querySelector(".user-action-time").outerHTML
        }
        var allFlairs = eachAnswer.querySelectorAll(".-flair")
        var reputation = allFlairs[allFlairs.length-1].innerHTML
        var temp = eachAnswer.querySelectorAll(".user-info")
        var active = index==0?"active":""
        if (temp[temp.length - 1].querySelector("img")) {
            var image = temp[temp.length - 1].querySelector("img").outerHTML
        } else {
            var image = ''
        }
        if (temp[temp.length - 1].querySelector(".user-details") && temp[temp.length - 1].querySelector(".user-details").querySelector("a")) {
            var name = temp[temp.length - 1].querySelector(".user-details").querySelector("a").innerHTML
        } else {
            var name = "Unknown author"
        }
        var string = genearteHtml(answer_time = answer_time, image = image, reputation = reputation, name = name, id = id, active = active)
        origin.appendChild(htmlToElement(string))
        // click on the answer
        fixedContainer.querySelector(`[id="${id}"]`).addEventListener("click", (e) => {
            var elem = document.querySelector(`#answer-${id}`)
            setTimeout(function() {
              window.scrollTo({top:elem.offsetTop, left:0, behavior:"smooth"})
            }, 0);
            fixedContainer.querySelectorAll(".list-group-item").forEach(eachli => {
                eachli.classList.remove("active")
            })
            fixedContainer.querySelector(`[id="${id}"]`).classList.add("active")
        })
    })
  }).then((eachAnswer)=>{
    document.body.appendChild(fixedContainer)
    currWidth = fixedContainer.offsetWidth
    currHeight = fixedContainer.offsetHeight
    currTop = fixedContainer.offsetTop
    // All the listeners
    fixedContainer.querySelector("#sidebar-header-wrapper").onmousedown = dragMouseDown
    fixedContainer.querySelector("#color-picker").addEventListener("input", (e)=>{
        let root = document.documentElement;
        root.style.setProperty('--rcolor', fixedContainer.querySelector("#color-picker").value.match(/[A-Za-z0-9]{2}/g).map(function(v) { return parseInt(v, 16) })[0]);
        root.style.setProperty('--gcolor', fixedContainer.querySelector("#color-picker").value.match(/[A-Za-z0-9]{2}/g).map(function(v) { return parseInt(v, 16) })[1]);
        root.style.setProperty('--bcolor', fixedContainer.querySelector("#color-picker").value.match(/[A-Za-z0-9]{2}/g).map(function(v) { return parseInt(v, 16) })[2]);
    })
    fixedContainer.querySelector("#traffic-light-red").addEventListener("click", (e)=>{
        // minimizes the sidebar
        oldHeight = fixedContainer.offsetHeight;
        oldTop = fixedContainer.offsetTop;
        fixedContainer.style.top = "95.3vh"
        fixedContainer.style.height = "4.7vh"
        fixedContainer.querySelector("#traffic-light-red").style.display = "none"
        fixedContainer.querySelector("#traffic-light-green").style.display = "block"
    })
    fixedContainer.querySelector("#traffic-light-green").addEventListener("click", (e)=>{
        // maximizes the sidebar
        fixedContainer.style.top = oldTop + "px"
        fixedContainer.style.height = oldHeight + "px"
        currWidth = fixedContainer.offsetWidth
        currHeight = fixedContainer.offsetHeight
        fixedContainer.querySelector("#traffic-light-green").style.display = "none"
        fixedContainer.querySelector("#traffic-light-red").style.display = "block"
    })
    fixedContainer.addEventListener("mouseup", (e)=>{
      if (fixedContainer.offsetWidth != currWidth || fixedContainer.offsetHeight != currHeight) {
        currWidth = fixedContainer.offsetWidth
        currHeight = fixedContainer.offsetHeight
        fixedContainer.style.top = 938-fixedContainer.offsetHeight + "px"
      }
    })
    // done = true
  })

// Sidebar.js stuff
var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
tooltipTriggerList.forEach(function (tooltipTriggerEl) {
  new bootstrap.Tooltip(tooltipTriggerEl)
})

// Sidebar stuff done.

//Check if in viewport
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// Seven parameters, one html tag:
function genearteHtml(answer_time, image, reputation, name, id, active="active", sentence="Loading", code_hidden='', mark='', to = "#") {
    // string = `<a id="${id}" href="${to}" class="list-group-item list-group-item-action ${active} ${mark} py-3 lh-tight" aria-current="true">
    //     <div id="summary-info">
    //       <div class="d-flex w-100 align-items-center justify-content-between">
    //         <strong class="mb-1">${sentence}</strong>
    //       </div>
    //     </div>
    //     <div class="answer-tiny-info">
    //       <div class="user-feature">
    //         <div class='user-profile'>
    //                   ${answer_time}
    //                   <div class="tiny-user-image">
    //                       ${image}
    //                       <div class="user-reputation">
    //                         <p>${name}</p>
    //                         ${reputation}
    //                       </div>
    //                   </div>
    //         </div>
    //         <div class='code-exist ${code_hidden}'>
    //         </div>
    //       </div>
    //       <div class="big-check hidden">
    //       </div>
    //     </div>
    //   </a>`
    string = `<a id="${id}" href="${to}" class="list-group-item list-group-item-action ${active} ${mark} py-3 lh-tight" aria-current="true">
        <div id="summary-info">
          <div class="d-flex w-100 align-items-center justify-content-between">
            <strong class="mb-1"></strong>
          </div>
        </div>
        <div class="answer-tiny-info">
          <div class="user-feature">
            <div class='user-profile'>
                      ${answer_time}
                      <div class="tiny-user-image">
                          <div class="user-reputation">
                            ${reputation}
                          </div>
                      </div>
            </div>
          </div>
        <div class='code-exist ${code_hidden}'>
        </div>
        </div>
      </a>`
    return string
}
// 将String转化为html内容。
function htmlToElement(html) {
    var template = document.createElement('template');
    html = html.trim(); // Never return a text node of whitespace as the result
    template.innerHTML = html;
    return template.content.firstChild;
}

// 发送Get请求。
function httpGet(theUrl)
{
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.open( "GET", theUrl, false ); // false for synchronous request
    xmlHttp.send( null );
    return xmlHttp.responseText;
}

// 发送Post请求。
async function httpPost(url, data, eachAnswer)
{
    var id = eachAnswer.id.split("-")[1]
    var xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.setRequestHeader("Accept", "application/json");
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.setRequestHeader('Access-Control-Allow-Origin', '*');
    xhr.onreadystatechange = function () {
    if (xhr.readyState === 4) {
        text = eachAnswer.querySelector(".js-post-body")
        var fullText = ""
        var runCode = ""
        // answer text 
        // needs to wait for DOM to finish
        setTimeout(() => {
          var span = text.getElementsByClassName("extracted")
          for (var i = 0; i < span.length; i++) {
            span[i].classList += " modelHighlighted"
            if (!span[i].innerHTML.includes("snippet") && !span[i].innerHTML.includes("Hide results")) {
              fullText += span[i].innerHTML
            } else {
              runCode += span[i].innerHTML
            }
          }
          var textBody= "<span class='text'>"+fullText+"</span>"
          if (fullText == "" && runCode == "") {
            textBody = "<span class='empty'>Empty</span>"
          }
          if (fullText == "" && runCode != "") {
            textBody = "<span class='codeOnly'>Code Only</span>"
          }
          fixedContainer.querySelector(`[id='${id}'] .mb-1`).innerHTML = textBody
          var badges = fixedContainer.getElementsByClassName("badgecount")
          for (var i = badges.length-1; i >= 0; i--) {
            badges[i].classList += "badgeCountBlack"
          }
        }, 0);
        text.innerHTML = xhr.responseText.split("bonankou")[0] 
        // code available
        if (xhr.responseText.split("bonankou")[2] == "yes") {
            fixedContainer.querySelector(`[id='${id}'] .code-exist`).innerHTML = `<div class='green-lattern'>
                                                                                  </div>
                                                                                  <p><small><small>Code Available</small></small></p>`
        } 
        // else {
        //     fixedContainer.querySelector(`[id='${id}'] .code-exist`).innerHTML = `<div class='red-lattern'>
        //                                                                           </div>
        //                                                                           <p>Text only</p>`
        // }
        // if (eachAnswer.id in prev_dict) {
        //     console.log("selected")
        //     fixedContainer.querySelector(`[id='${id}']`).classList.add("marked")
        //     fixedContainer.querySelector(`[id='${id}'] .big-check`).classList.remove("hidden")
        // }
        eachAnswer.querySelector(".js-post-body").querySelectorAll(".so_tracker_sentence").forEach((each_sentence, index) => {
            var sentence_index
            var answer_id
            each_sentence.classList.forEach((each_class)=>{
                if (each_class.startsWith("index_")) {
                    sentence_index = each_class
                } else if (each_class.startsWith("belong")) {
                    answer_id = "answer-" + each_class.split("_")[1]
                }
            })

            // if (eachAnswer.id in prev_dict) {
            //     console.log(prev_dict[eachAnswer.id])
            //     if (prev_dict[eachAnswer.id].contains(sentence_index)) {
            //         console.log("sentence selected")
            //         each_sentence.classList.add("manually_selected")
            //     }
            // }
            
            // highlight the sentence
            // what is big-check?
            each_sentence.addEventListener("click", (e)=>{
                console.log("Clicked a sentence")
                if (each_sentence.classList.contains("extracted")) {
                    fixedContainer.querySelector(`[id='${id}']`).classList.add("marked")
                    each_sentence.classList.remove("extracted")
                    each_sentence.classList.add("manually_selected")
                    /*document.querySelectorAll(`#${answer_id} .${sentence_index}`).forEach((each_friend)=>{
                        //each_friend.classList.remove("extracted")
                        each_friend.classList.add("manually_selected")
                    })*/
                } else if (each_sentence.classList.contains("manually_selected")) {
                    each_sentence.classList.remove("manually_selected")
                    /*document.querySelectorAll(`#${answer_id} .${sentence_index}`).forEach((each_friend)=>{
                        each_friend.classList.remove("manually_selected")
                    })*/
                    if (nothingLeft(eachAnswer)) {
                        console.log("Nothing left, burn the world")
                        fixedContainer.querySelector(`[id='${id}']`).classList.remove("marked")
                    }
                } else {
                  if (each_sentence.classList.contains("modelHighlighted")) {
                    each_sentence.classList.add("extracted")
                  } else {
                    fixedContainer.querySelector(`[id='${id}']`).classList.add("marked")
                    each_sentence.classList.add("manually_selected")
                    /*document.querySelectorAll(`#${answer_id} .${sentence_index}`).forEach((each_friend)=>{
                        each_friend.classList.add("manually_selected")
                    })*/
                  }
                }
            })
        })
        return xhr.responseText
        // fixedContainer.setAttribute("style","height:0px;")
        // fixedContainer.setAttribute("style",`height:${document.querySelector('body').scrollHeight - document.querySelector('.top-bar').scrollHeight - 10}px`)
    }};
    xhr.send(data);
    // return xhr
}


function nothingLeft(element) {
    result = true
    element.querySelectorAll(".so_tracker_sentence").forEach((each_sentence)=>{
        console.log(each_sentence)
        if (each_sentence.classList.contains("manually_selected")) {
            result = false
        }
    })
    return result
}

// Draggable component.
function dragMouseDown(e) {
    e = e || window.event;
    e.preventDefault();
    // get the mouse cursor position at startup:
    pos3 = e.clientX;
    pos4 = e.clientY;
    current_width = fixedContainer.offsetWidth
    document.onmouseup = closeDragElement;
    // call a function whenever the cursor moves:
    document.onmousemove = elementDrag;
}

function elementDrag(e) {
    e = e || window.event;
    e.preventDefault();
    // calculate the new cursor position:
    pos1 = pos3 - e.clientX
    pos2 = pos4 - e.clientY
    pos3 = e.clientX;
    pos4 = e.clientY;
    // set the element's new position:
    // Uncomment this to make it draggable
    //document.getElementById("fixedContainer").style.top = (0 - fixedContainer.style.height) + "px";
    if (fixedContainer.offsetTop == 893 && fixedContainer.offsetHeight == 45) {
      fixedContainer.style.left = (fixedContainer.offsetLeft - pos1) + "px";
    } else {
      fixedContainer.style.left = (fixedContainer.offsetLeft - pos1) + "px";
      fixedContainer.style.top = (fixedContainer.offsetTop - pos2) + "px";
    }
}

function closeDragElement(e) {
    // stop moving when mouse button is released:
    document.onmouseup = null;
    document.onmousemove = null;
} 

function everyHalfSecond() {
    // // Object.assign(dict, prev_dict)
    // dict = prev_dict
    // console.log(prev_dict)

    // if (first) {
    //     console.log("悲伤", `http://localhost:8000/${questionid}`)
    //     fetch(`http://localhost:8000/${questionid}`, {
    //         headers: {
    //             'Access-Control-Allow-Origin': '*', 
    //             "Accept": "application/json", 
    //             "Content-Type": "application/json"}, 
    //         mode: 'cors'}).then(function(response) {
    //         return response.text();
    //         }).then((dict_string)=>{
    //             dict = JSON.parse(dict_string)
    //             console.log(dict_string)
    //         })
    // }
    // first = false
    // if (done) {
        var dict = {}
        var question = document.querySelector(".question")
        var allanswers = document.querySelectorAll(".answer")
        allanswers.forEach((eachAnswer, index) => {
            text = eachAnswer.querySelector(".js-post-body")
            text.querySelectorAll(".so_tracker_sentence").forEach((each_sentence)=>{

                var sentence_index
                each_sentence.classList.forEach((each_class)=>{
                    if (each_class.startsWith("index_")) {
                        sentence_index = each_class
                    }
                })

                if (!dict.hasOwnProperty(questionid)) {
                    dict[questionid] = {}
                }

                if (each_sentence.classList.contains("manually_selected")) {

                    if (!dict[questionid].hasOwnProperty(eachAnswer.id)) {
                        dict[questionid][eachAnswer.id] = []
                    }

                    if (!dict[questionid][eachAnswer.id].includes(sentence_index.split("_")[1])) {
                        knowledge_extracted = ""
                        document.querySelectorAll(`#${eachAnswer.id} .${sentence_index}`).forEach((each_friend)=>{
                            knowledge_extracted += each_friend.innerHTML
                        })
                        dict[questionid][eachAnswer.id].push(sentence_index.split("_")[1])
                        dict[questionid][eachAnswer.id].push(knowledge_extracted)
                    }
                }
            })
        })
    // }

    if (!compareDictionaries(dict, prev_dict)) {
        //console.log("do something")
        var url = "http://localhost:8000/"
        var data = "updatebonankou" + JSON.stringify(dict)
        tempPost(url, data)
        // var xhr = new XMLHttpRequest();
        // xhr.open("POST", url);
        // xhr.setRequestHeader("Accept", "application/json");
        // xhr.setRequestHeader("Content-Type", "application/json");
        // xhr.setRequestHeader('Access-Control-Allow-Origin', '*');
        // xhr.send(data)
        prev_dict = {}
        Object.assign(prev_dict, dict)
    }
}

function compareDictionaries(d1, d2)
{
    // quick check for the same object
    if( d1 == d2 )
        return true;

    // check for null
    if( d1 == null || d2 == null )
        return false;

    // go through the keys in d1 and check if they're in d2 - also keep a count
    var count = 0;
    for( var key in d1 )
    {
        // check if the key exists
        if( !( key in d2 ) )
            return false;

        // check that the values are the same
        if (d1[key].length != d2[key].length) {
            return false
        }

        for (second_key in d1[key]) {
            if (! (second_key in d2[key])) {
                return false
            }
            
            if (d2[key][second_key].length != d1[key][second_key].length) {
                return false
            }
            d1[key][second_key].forEach((element, second_index) => {
                if (d1[key][second_key][second_index] != d2[key][second_key][second_index]) {
                    return false
                }
            })
        }

        count++;
    }

    // now just make sure d2 has the same number of keys
    var count2 = 0;
    for( key in d2 )
        count2++;

    // return if they're the same size
    return ( count == count2 );
}

async function tempPost(url, data)
{
    var xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.setRequestHeader("Accept", "application/json");
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.setRequestHeader('Access-Control-Allow-Origin', '*');
    xhr.onreadystatechange = function () {
    if (xhr.readyState === 4) {
        return xhr.responseText
        // fixedContainer.setAttribute("style","height:0px;")
        // fixedContainer.setAttribute("style",`height:${document.querySelector('body').scrollHeight - document.querySelector('.top-bar').scrollHeight - 10}px`)
    }};
    xhr.send(data);
    // return xhr
}