chrome.webNavigation.onCommitted.addListener((details) => {
    // ["reload", "link", "typed", "generated"].includes(details.transitionType) &&
    if (
        /^https:\/\/stackoverflow.com\/questions\/\d*/.test(details.url)) {
        console.log(details)
        chrome.tabs.get(details.tabId, current_tab_info => {
            chrome.scripting.executeScript({files: ["./foreground.js"], target: {tabId: details.tabId}}, (result)=>{console.log("foreground.js complete")})
            chrome.scripting.insertCSS({files: ["./foreground.css"], target: {tabId: details.tabId}}, (result)=>{console.log("CSS complete")})
        })
    }
});

chrome.tabs.onUpdated.addListener(function(tabId, changeInfo, tab) {
    // if (changeInfo["status"] == "complete") {
    //     console.log("Update:", tab)
    //     console.log(changeInfo)
    //     // console.log("The id:", tabid)
    //     // I think the second argument is a callback function
    //     chrome.tabs.get(tabId, current_tab_info => {
    //         if (/^https:\/\/stackoverflow.com\/questions\/\d*/.test(current_tab_info.url)) {
    //             chrome.scripting.executeScript({files: ["./foreground.js"], target: {tabId: tabId}}, (result)=>{console.log("foreground.js complete")})
    //             chrome.scripting.insertCSS({files: ["./foreground.css"], target: {tabId: tabId}}, (result)=>{console.log("CSS complete")})
    //         }
    //     })
    // }
})


// Do some testing here
chrome.runtime.onUpdateAvailable.addListener(tab => {
    console.log("onUpdateAvailable", tab)
})

chrome.runtime.onSuspendCanceled.addListener(() => {
    console.log("onSuspendCanceled")
})

chrome.runtime.onSuspend.addListener(() => {
    console.log("onSuspend")
})

chrome.runtime.onStartup.addListener(() => {
    console.log("onStartup")
})

chrome.runtime.onRestartRequired.addListener(reason => {
    console.log("onRestartRequired", reason)
})

chrome.runtime.onMessageExternal.addListener((message, sender, sendresponse) => {
    console.log("onMessageExternal", message)
})

chrome.runtime.onMessage.addListener((message, sender, sendresponse) => {
    console.log("onMessage", message)
})

chrome.runtime.onInstalled.addListener(details => {
    console.log("onInstalled", details.id)
})

chrome.runtime.onBrowserUpdateAvailable.addListener(() => {
    console.log("onBrowserUpdateAvailable")
})
