import { messages } from './locales/zh-CN.js';

const languages = ['en', 'zh-CN'];
const storageKey = 'watchwise.language';
const requested = new URL(location.href).searchParams.get('lang');
let saved;
try { saved = localStorage.getItem(storageKey); } catch { /* Storage is optional. */ }
let language = languages.includes(requested) ? requested : languages.includes(saved) ? saved : 'en';

export const locale = () => language;
export function t(english, values = {}) {
  const text = language === 'zh-CN' ? messages[english] ?? english : english;
  return text.replace(/\{(\w+)\}/g, (match, key) => values[key] ?? match);
}

// Capture the static English source once. Translate text and attributes only,
// keeping the original markup, event listeners, and controls intact.
const textBindings = [];
const walker = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_TEXT);
while (walker.nextNode()) {
  const node = walker.currentNode;
  if (node.parentElement.closest('script, style, noscript, option')) continue;
  const key = node.textContent.trim().replace(/\s+/g, ' ');
  if (Object.hasOwn(messages, key)) textBindings.push({ node, key, original: node.textContent });
}
const attributeBindings = [];
for (const element of document.querySelectorAll('[aria-label], [placeholder], [data-question], meta[content]')) {
  for (const attribute of ['aria-label', 'placeholder', 'data-question', 'content']) {
    const original = element.getAttribute(attribute);
    if (Object.hasOwn(messages, original)) attributeBindings.push({ element, attribute, original });
  }
}
const docs = [
  ['.how-intro .text-link', 'https://github.com/PAOPAOtomato/watchwise/blob/main/README.zh-CN.md#工作原理'],
  ['.installation-copy .button', 'https://github.com/PAOPAOtomato/watchwise/blob/main/README.zh-CN.md#试用路由模拟无需摄像头gpu-或-api-密钥'],
].map(([selector, chinese]) => ({ element: document.querySelector(selector), chinese, english: document.querySelector(selector).href }));

export function applyLanguage(next = language, persist = false) {
  if (!languages.includes(next)) return;
  language = next;
  document.documentElement.lang = language;
  for (const { node, key, original } of textBindings) {
    // English whitespace also separates adjacent inline elements in Chinese.
    node.textContent = language === 'en' ? original : original.replace(/\S[\s\S]*\S|\S/, () => t(key));
  }
  for (const { element, attribute, original } of attributeBindings) element.setAttribute(attribute, t(original));
  for (const { element, english, chinese } of docs) element.href = language === 'en' ? english : chinese;
  document.querySelector('#language').value = language;
  if (persist) {
    try { localStorage.setItem(storageKey, language); } catch { /* Still switch for this visit. */ }
    const url = new URL(location.href);
    url.searchParams.set('lang', language);
    history.replaceState(null, '', url);
  }
}
