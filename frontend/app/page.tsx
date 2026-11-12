"use client";

import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";

const API_URL = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000").replace(/\/+$/, "");
const MAX_BYTES = 5 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png", "webp"];

type Probability = { label: string; probability: number };
type Prediction = {
  prediction: string;
  confidence: number;
  sure: boolean;
  probabilities: Probability[];
};

const EMOJI: Record<string, string> = {
  daisy: "🌼",
  dandelion: "🌾",
  roses: "🌹",
  sunflowers: "🌻",
  tulips: "🌷",
};

// "sunflowers" -> "Sunflowers"
function capitalize(label: string) {
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function percent(p: number) {
  return `${(p * 100).toFixed(1)}%`;
}

function validate(file: File): string | null {
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (!ALLOWED_TYPES.includes(file.type) && !ALLOWED_EXTENSIONS.includes(ext)) {
    return "That file isn't a supported image. Please choose a JPG, PNG or WebP photo.";
  }
  if (file.size > MAX_BYTES) {
    const mb = (file.size / (1024 * 1024)).toFixed(1);
    return `That image is ${mb} MB. Please choose one under 5 MB.`;
  }
  return null;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Free the preview's object URL when it changes or the page unmounts.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function selectFile(selected: File | undefined) {
    if (!selected) return;
    setResult(null);
    const problem = validate(selected);
    if (problem) {
      setError(problem);
      setFile(null);
      setPreviewUrl(null);
      return;
    }
    setError(null);
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
  }

  function onInputChange(e: ChangeEvent<HTMLInputElement>) {
    selectFile(e.target.files?.[0]);
    // Allow re-selecting the same file later.
    e.target.value = "";
  }

  function onDrop(e: DragEvent<HTMLElement>) {
    e.preventDefault();
    setDragging(false);
    if (loading) return;
    selectFile(e.dataTransfer.files?.[0]);
  }

  function reset() {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setLoading(false);
  }

  async function identify() {
    if (!file || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const body = new FormData();
    body.append("image", file);

    let response: Response;
    try {
      response = await fetch(`${API_URL}/predict`, { method: "POST", body });
    } catch {
      setError(
        `Can't reach the classifier at ${API_URL}. Make sure the Flask backend is running (python app.py) and try again.`,
      );
      setLoading(false);
      return;
    }

    try {
      const data = await response.json();
      if (!response.ok) {
        setError(data?.error ?? `The server returned an error (${response.status}).`);
      } else {
        setResult(data as Prediction);
      }
    } catch {
      setError(`The server sent an unexpected response (${response.status}).`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-xl flex-1 flex-col px-4 py-10 sm:py-16">
      <header className="mb-8 text-center">
        <div className="mb-3 text-4xl" aria-hidden>
          🌷🌻🌼
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-stone-900 sm:text-4xl">
          Flower Classifier
        </h1>
        <p className="mt-2 text-stone-600">Upload a photo and I&apos;ll tell you which flower it is.</p>
      </header>

      <section className="rounded-3xl border border-white/60 bg-white/80 p-5 shadow-xl shadow-emerald-900/5 backdrop-blur sm:p-7">
        {!previewUrl ? (
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={`flex w-full flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed px-6 py-14 text-center transition focus:outline-none focus-visible:ring-4 focus-visible:ring-emerald-200 ${
              dragging
                ? "border-emerald-500 bg-emerald-50"
                : "border-stone-300 bg-stone-50/60 hover:border-emerald-400 hover:bg-emerald-50/50"
            }`}
          >
            <svg
              className="h-10 w-10 text-emerald-600"
              fill="none"
              viewBox="0 0 24 24"
              strokeWidth={1.5}
              stroke="currentColor"
              aria-hidden
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
              />
            </svg>
            <span className="font-medium text-stone-800">
              {dragging ? "Drop it here" : "Drag & drop a photo, or click to browse"}
            </span>
            <span className="text-sm text-stone-500">JPG, PNG or WebP · up to 5 MB</span>
          </button>
        ) : (
          <div className="overflow-hidden rounded-2xl bg-stone-100">
            {/* eslint-disable-next-line @next/next/no-img-element -- local blob preview */}
            <img
              src={previewUrl}
              alt="Selected flower"
              className="mx-auto max-h-80 w-full object-contain"
            />
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
          className="hidden"
          onChange={onInputChange}
        />

        {error && (
          <div
            role="alert"
            className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700"
          >
            {error}
          </div>
        )}

        {file && !result && (
          <button
            type="button"
            onClick={identify}
            disabled={loading}
            className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-600 px-5 py-3 font-semibold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-emerald-400"
          >
            {loading && (
              <span
                className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white"
                aria-hidden
              />
            )}
            {loading ? "Identifying…" : "Identify flower"}
          </button>
        )}

        {result && <ResultCard result={result} />}

        {(file || result || error) && (
          <button
            type="button"
            onClick={reset}
            disabled={loading}
            className="mt-3 w-full rounded-xl border border-stone-300 bg-white px-5 py-3 font-medium text-stone-700 transition hover:bg-stone-50 disabled:opacity-50"
          >
            Try another photo
          </button>
        )}
      </section>

      <p className="mt-6 text-center text-sm text-stone-500">
        This model recognizes daisies, dandelions, roses, sunflowers and tulips.
      </p>
    </main>
  );
}

function ResultCard({ result }: { result: Prediction }) {
  const name = capitalize(result.prediction);

  return (
    <div className="mt-5 space-y-5" aria-live="polite">
      {result.sure ? (
        <div className="rounded-2xl bg-emerald-50 p-5 ring-1 ring-emerald-100">
          <p className="text-sm font-medium uppercase tracking-wide text-emerald-700">It looks like</p>
          <p className="mt-1 flex items-center gap-2 text-3xl font-bold text-stone-900">
            <span aria-hidden>{EMOJI[result.prediction] ?? "🌸"}</span>
            {name}
          </p>
          <div className="mt-4 flex items-center justify-between text-sm">
            <span className="text-stone-600">Confidence</span>
            <span className="font-mono font-semibold text-emerald-700">{percent(result.confidence)}</span>
          </div>
          <div className="mt-1.5 h-2.5 w-full overflow-hidden rounded-full bg-emerald-100">
            <div
              className="h-full rounded-full bg-emerald-500 transition-[width] duration-700"
              style={{ width: percent(result.confidence) }}
            />
          </div>
        </div>
      ) : (
        <div className="rounded-2xl bg-amber-50 p-5 ring-1 ring-amber-100">
          <p className="flex items-center gap-2 text-lg font-semibold text-amber-900">
            <span aria-hidden>🤔</span>
            Not sure — this might not be one of the 5 flowers I know
          </p>
          <p className="mt-1 text-sm text-amber-800">
            My best guess is {name} at {percent(result.confidence)}, but that&apos;s too low to be confident.
          </p>
        </div>
      )}

      <div>
        <h2 className="mb-3 text-sm font-semibold text-stone-700">All probabilities</h2>
        <ul className="space-y-2.5">
          {result.probabilities.map((p, i) => (
            <li key={p.label}>
              <div className="mb-1 flex justify-between text-sm">
                <span className={i === 0 ? "font-semibold text-stone-900" : "text-stone-600"}>
                  <span aria-hidden className="mr-1.5">
                    {EMOJI[p.label] ?? "🌸"}
                  </span>
                  {capitalize(p.label)}
                </span>
                <span className="font-mono text-stone-600">{percent(p.probability)}</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-stone-100">
                <div
                  className={`h-full rounded-full transition-[width] duration-700 ${
                    i === 0 ? (result.sure ? "bg-emerald-500" : "bg-amber-400") : "bg-stone-300"
                  }`}
                  style={{ width: percent(p.probability) }}
                />
              </div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
