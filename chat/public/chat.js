const chatLog = document.getElementById('chat-log');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const micButton = document.getElementById('mic-button');
const muteButton = document.getElementById('mute-button');

const API_URL = '/chat/query';
const DEFAULT_TOP_K = 5;
const DEFAULT_STRICT = true;
const DEFAULT_FILTERS = {};
const MAX_INPUT_LENGTH = 2000;

const SANITIZE_CFG = {
    SAFE_FOR_TEMPLATES: true,
    ALLOWED_TAGS: ['br','p','ul','li','strong','em','a','code','pre'],
    ALLOWED_ATTR: ['href', 'target', 'rel'],
    FORBID_TAGS: ['style','iframe','object','embed']
};
const sanitizeHTML = dirty => DOMPurify.sanitize(dirty, SANITIZE_CFG);
const md = window.markdownit({ linkify: true, breaks: true });

let isMuted = true;
let isWaitingResponse = false;

function updateMuteVisual() {
    muteButton.classList.toggle('listening', !isMuted);
    muteButton.title = isMuted ? "Activar lectura de voz" : "Silenciar lectura de voz";
}

function speakText(text) {
    if (!isMuted && 'speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'es-MX';
        window.speechSynthesis.speak(utterance);
    }
}

function formatMessage(text, noLimit = false) {
    text = text.replace(/\\n/g, '\n')
        .replace(/([^\n])\n(-\s)/g, '$1\n\n$2');
    const dirtyHTML = md.render(text);
    const sanitized = sanitizeHTML(dirtyHTML);
    return (noLimit || sanitized.length <= MAX_INPUT_LENGTH) 
        ? sanitized 
        : sanitized.slice(0, MAX_INPUT_LENGTH) + '…';
}

function addMessage(content, sender, isThinking = false) {
    const existingThinking = chatLog.querySelector('[data-thinking="true"]');
    if (existingThinking) existingThinking.remove();

    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message-block', 'message', sender);
    if (isThinking) messageDiv.dataset.thinking = 'true';

    const bubbleDiv = document.createElement('div');
    bubbleDiv.classList.add('bubble');

    if (sender === 'assistant' && !isThinking) {
        const icon = document.createElement('div');
        icon.classList.add('chat-icon', 'assistant-icon');
        icon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 96.016 96.515" width="24" height="24" aria-hidden="true"><g transform="translate(-43.523 -77.029)"><path d="m95.273 77.189c18.602 1.0443 35.605 14.334 41.534 31.937 6.0697 16.701 2.0349 36.534-10.184 49.462-9.3142 10.184-23.242 15.81-37.017 14.944-18.717-0.11389-36.457-12.852-42.854-30.409-5.9703-15.454-3.5328-33.845 6.5508-47.039 9.5905-12.94 25.929-20.268 41.97-18.894m-4.445 15.029c-3.6095 1.7173-3.45 6.4068-4.9218 9.6687-1.3077 3.8257-2.4015 7.7653-3.9153 11.492l-20.32 0.0576c-4.9037 3.6943 2.6344 7.0422 5.0271 9.5779 3.0934 2.8524 6.9692 5.2676 9.543 8.4222-2.1434 6.7971-4.6506 13.494-6.5459 20.358 0.7346 6.4507 6.8476 0.62751 9.5594-1.1687 4.0022-2.8521 8.0455-5.6694 12.098-8.455 6.0558 4.0901 11.78 8.6847 17.907 12.663 6.0936 0.34711 2.6959-6.6111 1.6205-9.8517-1.4024-4.6422-2.9671-9.2263-4.4328-13.846 5.1843-4.6822 10.914-8.813 15.778-13.812 1.5585-5.8634-7.0918-3.3026-10.454-3.8599-3.6817-9e-3 -7.3634-0.0178-11.045-0.0267-2.3787-6.264-4.0403-12.819-6.4622-19.066-0.54835-1.3635-1.9013-2.4309-3.4364-2.1529" fill="#043055" fill-rule="evenodd"/></g></svg>';
        bubbleDiv.prepend(icon);
    }

    if (sender === 'user') {
        const icon = document.createElement('div');
        icon.classList.add('chat-icon', 'user-icon');
        icon.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" viewBox="0 0 96.095 96.668" width="24" height="24" aria-hidden="true"><g transform="translate(-57.15 -100.28)"><path d="m110.38 100.43c15.331 1.2934 29.44 10.887 36.7 24.381 10.612 18.493 7.2145 43.829-8.1777 58.652-16.008 16.384-44.288 18.045-62.371 4.1529-12.983-9.4424-20.35-25.649-19.369-41.618 0.53965-17.56 11.706-34.154 27.641-41.517 7.9383-3.8246 16.858-4.7479 25.577-4.0508m-6.8792 16.731c-9.5257 0.80559-16.916 10.994-14.284 20.283 2.1157 8.5477 11.659 14.142 20.144 11.773 8.3176-2.008 14.1-11.137 11.914-19.526-1.7728-7.8799-9.7042-13.598-17.774-12.53m-7.8317 37.677c-8.9507 0.86168-16.83 9.1322-16.163 18.316-0.8459 6.0926 6.39 5.0925 10.304 5.0547 12.958 0.0131 25.915 4e-3 38.873 6e-3 4.3285-1.6812 2.9971-7.4236 2.1225-10.916-2.5306-7.7304-10.592-13.232-18.746-12.499-5.4618-0.0812-10.932-0.19612-16.391 0.0376" fill="#043055" fill-rule="evenodd"/></g></svg>';
        bubbleDiv.prepend(icon);
    }

    const contentSpan = document.createElement('span');
    const skipLimit = (sender === 'assistant');
    contentSpan.innerHTML = isThinking
        ? '<div class="typing-indicator" aria-live="polite"><span class="sr-only">Escribiendo...</span><span></span><span></span><span></span></div>'
        : formatMessage(content, skipLimit);

    bubbleDiv.appendChild(contentSpan);
    messageDiv.appendChild(bubbleDiv);
    chatLog.appendChild(messageDiv);
    chatLog.scrollTop = chatLog.scrollHeight;

    if (!isThinking && sender === 'assistant') speakText(contentSpan.textContent);
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (!message || isWaitingResponse) return;

    if (message.length > MAX_INPUT_LENGTH) {
        addMessage(`Tu mensaje es demasiado largo. Por favor, reduce el texto a menos de ${MAX_INPUT_LENGTH} caracteres.`, 'assistant');
        return;
    }

    isWaitingResponse = true;
    sendButton.classList.add('waiting');
    addMessage(message, 'user');
    userInput.value = '';
    addMessage('...', 'assistant', true);

    try {
        const payload = {
            question: message,
            top_k: DEFAULT_TOP_K,
            strict: DEFAULT_STRICT,
            filters: DEFAULT_FILTERS
        };
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Error ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        const assistantMessage = data?.answer ?? 'Lo siento, no tengo una respuesta para eso.';

        addMessage(assistantMessage, 'assistant');
    } catch (error) {
        console.error('Error en la solicitud:', error);
        addMessage('Hubo un error al procesar tu solicitud.', 'assistant');
    } finally {
        isWaitingResponse = false;
        sendButton.classList.remove('waiting');
    }
}

const throttledSend = throttle(sendMessage, 2500);
sendButton.addEventListener('click', throttledSend);
userInput.addEventListener('keypress', e => {
    if (e.key === 'Enter') throttledSend();
});

muteButton.addEventListener('click', () => {
    isMuted = !isMuted;
    updateMuteVisual();
    if (isMuted && window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
    }
});

if ('webkitSpeechRecognition' in window && (
    (/Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor)) ||
    (/Edg/.test(navigator.userAgent)) || (/Brave/.test(navigator.userAgent) || navigator.brave))) {
    const recognition = new webkitSpeechRecognition();
    recognition.lang = 'es-MX';
    recognition.continuous = false;
    recognition.interimResults = false;

    micButton.addEventListener('click', () => {
        micButton.classList.add('listening');
        isMuted = false;
        updateMuteVisual();
        recognition.start();
    });

    recognition.onresult = event => {
        const transcript = event.results[0][0].transcript;
        userInput.value = transcript;
        userInput.focus();
        micButton.classList.remove('listening');
    };

    recognition.onerror = () => {
        micButton.classList.remove('listening');
        alert('Hubo un error al reconocer tu voz.');
    };
} else {
    micButton.style.display = 'none';
    const voiceStatus = document.getElementById('voice-status');
    if (voiceStatus) {
        voiceStatus.textContent = 'Reconocimiento de voz no disponible en el navegador.';
    }
}

function throttle(func, limit) {
    let lastCall = 0;
    return function (...args) {
        const now = Date.now();
        if (now - lastCall >= limit) {
            lastCall = now;
            func.apply(this, args);
        }
    };
}

updateMuteVisual();
