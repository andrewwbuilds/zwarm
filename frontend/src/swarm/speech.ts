// A thin typed wrapper over the browser's speech recognition. Chrome and Edge send the audio to
// a Google service, so voice needs internet there. Safari does it on the device.

interface RecognitionAlternative {
  readonly transcript: string;
}

interface RecognitionResult {
  readonly isFinal: boolean;
  readonly length: number;
  readonly [index: number]: RecognitionAlternative;
}

interface RecognitionEvent {
  readonly resultIndex: number;
  readonly results: { readonly length: number; readonly [index: number]: RecognitionResult };
}

interface RecognitionError {
  readonly error: string;
}

interface Recognition {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((event: RecognitionEvent) => void) | null;
  onerror: ((event: RecognitionError) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

type RecognitionConstructor = new () => Recognition;

declare global {
  interface Window {
    SpeechRecognition?: RecognitionConstructor;
    webkitSpeechRecognition?: RecognitionConstructor;
  }
}

export interface Listener {
  stop(): void;
}

export interface ListenCallbacks {
  onInterim: (text: string) => void;
  onFinal: (text: string) => void;
  onError: (message: string) => void;
  onEnd: () => void;
}

const ERRORS: Record<string, string> = {
  "not-allowed": "The browser blocked the microphone. Allow it in the address bar, then try again.",
  "service-not-allowed": "The browser blocked speech recognition. Allow it in site settings, then try again.",
  "no-speech": "No speech heard. Press the microphone and speak.",
  "audio-capture": "No microphone found.",
  network: "Speech recognition needs internet in this browser. Type the command instead.",
};

export const speechSupported = (): boolean => Boolean(window.SpeechRecognition ?? window.webkitSpeechRecognition);

export function listen(callbacks: ListenCallbacks): Listener | null {
  const Constructor = window.SpeechRecognition ?? window.webkitSpeechRecognition;
  if (!Constructor) return null;
  const recognition = new Constructor();
  recognition.lang = "en-US";
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.onresult = (event) => {
    let interim = "";
    for (let i = event.resultIndex; i < event.results.length; i += 1) {
      const result = event.results[i];
      const words = result[0]?.transcript ?? "";
      if (result.isFinal) callbacks.onFinal(words.trim());
      else interim += words;
    }
    if (interim) callbacks.onInterim(interim.trim());
  };
  recognition.onerror = (event) => {
    if (event.error === "aborted") return;
    callbacks.onError(ERRORS[event.error] ?? `Speech recognition failed: ${event.error}.`);
  };
  recognition.onend = callbacks.onEnd;
  recognition.start();
  return { stop: () => recognition.stop() };
}
