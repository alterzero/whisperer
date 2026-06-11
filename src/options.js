const CONFIG_KEY = "whisperer_config";

const DEFAULTS = {
  liveIntervalMs: 5000,
  maxHistory: 50,
  maxTokens: 16384,
  chunkChars: 20000,
  systemPrompt: "You are a concise meeting notes assistant. Always respond with structured bullet points.",
  sections: [
    { title: "Important Points", instruction: "List the key points discussed" },
    { title: "Decisions", instruction: "List any decisions that were made" },
    { title: "Action Items", instruction: "List any tasks, follow-ups, or action items mentioned" },
  ],
};

const $ = (id) => document.getElementById(id);

const liveIntervalInput = $("live-interval");
const maxHistoryInput = $("max-history");
const maxTokensInput = $("max-tokens");
const chunkCharsInput = $("chunk-chars");
const systemPromptInput = $("system-prompt");
const sectionsList = $("sections-list");
const addSectionBtn = $("add-section-btn");
const saveBtn = $("save-btn");
const resetBtn = $("reset-btn");
const saveStatus = $("save-status");

async function loadConfig() {
  const result = await chrome.storage.local.get(CONFIG_KEY);
  return { ...DEFAULTS, ...result[CONFIG_KEY] };
}

function populateForm(config) {
  liveIntervalInput.value = config.liveIntervalMs;
  maxHistoryInput.value = config.maxHistory;
  maxTokensInput.value = config.maxTokens;
  chunkCharsInput.value = config.chunkChars;
  systemPromptInput.value = config.systemPrompt;
  renderSections(config.sections);
}

function renderSections(sections) {
  sectionsList.textContent = "";
  sections.forEach((s, i) => {
    sectionsList.appendChild(createSectionEntry(s, i));
  });
}

function createSectionEntry(section, index) {
  const entry = document.createElement("div");
  entry.className = "section-entry";
  entry.dataset.index = index;

  const header = document.createElement("div");
  header.className = "section-entry-header";

  const titleInput = document.createElement("input");
  titleInput.type = "text";
  titleInput.value = section.title;
  titleInput.placeholder = "Section title";
  titleInput.dataset.field = "title";

  const removeBtn = document.createElement("button");
  removeBtn.className = "remove-section-btn";
  removeBtn.title = "Remove section";
  removeBtn.textContent = "\u00D7";
  removeBtn.addEventListener("click", () => {
    entry.remove();
  });

  header.append(titleInput, removeBtn);

  const instrInput = document.createElement("textarea");
  instrInput.value = section.instruction;
  instrInput.placeholder = "Instructions for this section";
  instrInput.rows = 1;
  instrInput.dataset.field = "instruction";

  entry.append(header, instrInput);
  return entry;
}

function collectSections() {
  const entries = sectionsList.querySelectorAll(".section-entry");
  const sections = [];
  entries.forEach((entry) => {
    const title = entry.querySelector('[data-field="title"]').value.trim();
    const instruction = entry.querySelector('[data-field="instruction"]').value.trim();
    if (title) sections.push({ title, instruction });
  });
  return sections;
}

function collectConfig() {
  return {
    liveIntervalMs: Math.max(1000, parseInt(liveIntervalInput.value, 10) || DEFAULTS.liveIntervalMs),
    maxHistory: Math.max(5, parseInt(maxHistoryInput.value, 10) || DEFAULTS.maxHistory),
    maxTokens: Math.max(2048, parseInt(maxTokensInput.value, 10) || DEFAULTS.maxTokens),
    chunkChars: Math.max(5000, parseInt(chunkCharsInput.value, 10) || DEFAULTS.chunkChars),
    systemPrompt: systemPromptInput.value.trim() || DEFAULTS.systemPrompt,
    sections: collectSections().length > 0 ? collectSections() : DEFAULTS.sections,
  };
}

async function save() {
  const config = collectConfig();
  await chrome.storage.local.set({ [CONFIG_KEY]: config });
  saveStatus.textContent = "Settings saved";
  setTimeout(() => { saveStatus.textContent = ""; }, 2000);
}

addSectionBtn.addEventListener("click", () => {
  const index = sectionsList.children.length;
  sectionsList.appendChild(createSectionEntry({ title: "", instruction: "" }, index));
});

saveBtn.addEventListener("click", save);

resetBtn.addEventListener("click", async () => {
  await chrome.storage.local.remove(CONFIG_KEY);
  populateForm(DEFAULTS);
  saveStatus.textContent = "Reset to defaults";
  setTimeout(() => { saveStatus.textContent = ""; }, 2000);
});

document.addEventListener("DOMContentLoaded", async () => {
  const config = await loadConfig();
  populateForm(config);
});
