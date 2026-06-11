// Open the UI in a tab so microphone permissions work properly
// (extension popups close when the browser shows the permission dialog)
chrome.action.onClicked.addListener(() => {
  chrome.tabs.create({ url: chrome.runtime.getURL("popup.html") });
});
