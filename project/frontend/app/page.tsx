"use client";

import { useMemo, useState } from "react";

// ---------------------------------------------------------------------------
// Config: the six sensor features the model expects, in the exact order and
// with the exact key names (aliases) the backend pipeline was trained on.
// Ranges are derived from the training data's mean ± 3·std (StandardScaler
// stats baked into the pickled pipeline); defaults are the imputer medians.
// ---------------------------------------------------------------------------

type Sensor = {
  key: string; // exact column name expected by the model
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  defaultValue: number;
};

const SENSORS: Sensor[] = [
  { key: "Torque(Nm)", label: "Torque", unit: "Nm", min: 5, max: 45, step: 0.1, defaultValue: 24.7 },
  { key: "Hydraulic_Pressure(bar)", label: "Hydraulic Pressure", unit: "bar", min: 10, max: 195, step: 0.5, defaultValue: 96.6 },
  { key: "Cutting(kN)", label: "Cutting Force", unit: "kN", min: 0.9, max: 4.7, step: 0.01, defaultValue: 2.78 },
  { key: "Coolant_Pressure(bar)", label: "Coolant Pressure", unit: "bar", min: 1.9, max: 8, step: 0.05, defaultValue: 4.92 },
  { key: "Spindle_Speed(RPM)", label: "Spindle Speed", unit: "RPM", min: 8800, max: 31800, step: 10, defaultValue: 20150 },
  { key: "Coolant_Temperature", label: "Coolant Temperature", unit: "°C", min: 0, max: 44, step: 0.1, defaultValue: 21.2 },
];

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type PredictResult = {
  prediction: "Machine_Failure" | "No_Machine_Failure";
  needs_maintenance: boolean;
  probability_failure: number;
  confidence: number;
  needs_manual_review: boolean;
};

export default function Home() {
  const [readings, setReadings] = useState<Record<string, number>>(() =>
    Object.fromEntries(SENSORS.map((s) => [s.key, s.defaultValue]))
  );
  const [result, setResult] = useState<PredictResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (key: string, value: number) => {
    setReadings((prev) => ({ ...prev, [key]: value }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(readings),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const data: PredictResult = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto max-w-5xl px-6 py-12 md:py-16">
      <Header />
      <div className="mt-10 grid grid-cols-1 gap-6 lg:grid-cols-5">
        <section className="lg:col-span-3">
          <Panel title="Sensor Readings" eyebrow="INPUT · 6 CHANNELS">
            <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
              {SENSORS.map((sensor) => (
                <SliderField
                  key={sensor.key}
                  sensor={sensor}
                  value={readings[sensor.key]}
                  onChange={(v) => handleChange(sensor.key, v)}
                />
              ))}
            </div>
            <button
              onClick={handlePredict}
              disabled={loading}
              className="mt-8 w-full rounded-md border border-amber/40 bg-amber/10 py-3 font-display text-sm font-medium tracking-wide text-amber transition hover:bg-amber/20 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "RUNNING INFERENCE…" : "RUN PREDICTION"}
            </button>
            {error && (
              <p className="mt-3 rounded border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
                {error}
              </p>
            )}
          </Panel>
        </section>

        <section className="lg:col-span-2">
          <Panel title="Prediction" eyebrow="OUTPUT · LIVE">
            <OutputGauge result={result} loading={loading} />
          </Panel>
        </section>
      </div>
      <Footer />
    </main>
  );
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

function Header() {
  return (
    <header className="border-b border-line pb-6">
      <p className="font-mono text-xs tracking-[0.2em] text-cyan">PREDICTIVE MAINTENANCE · LIVE INFERENCE</p>
      <h1 className="mt-3 font-display text-3xl font-semibold tracking-tight text-ink md:text-4xl">
        Machine Downtime Predictor
      </h1>
      <p className="mt-2 max-w-2xl text-sm text-ink-muted">
        Adjust the six sensor channels to a live or hypothetical reading and run the trained
        classifier to check whether the machine is heading toward failure.
      </p>
    </header>
  );
}

// ---------------------------------------------------------------------------
// Panel shell
// ---------------------------------------------------------------------------

function Panel({ title, eyebrow, children }: { title: string; eyebrow: string; children: React.ReactNode }) {
  return (
    <div className="h-full rounded-lg border border-line bg-panel p-6">
      <p className="font-mono text-[11px] tracking-[0.15em] text-ink-faint">{eyebrow}</p>
      <h2 className="mt-1 font-display text-lg font-medium text-ink">{title}</h2>
      <div className="mt-5">{children}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Slider field
// ---------------------------------------------------------------------------

function SliderField({
  sensor,
  value,
  onChange,
}: {
  sensor: Sensor;
  value: number;
  onChange: (v: number) => void;
}) {
  const pct = ((value - sensor.min) / (sensor.max - sensor.min)) * 100;
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label className="text-xs font-medium text-ink-muted">{sensor.label}</label>
        <span className="font-mono text-sm text-ink">
          {value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
          <span className="ml-1 text-ink-faint">{sensor.unit}</span>
        </span>
      </div>
      <input
        type="range"
        className="sensor-slider mt-2"
        style={{ ["--thumb-color" as string]: `hsl(${38 - pct * 0.15}, 85%, 58%)` }}
        min={sensor.min}
        max={sensor.max}
        step={sensor.step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
      <div className="mt-1 flex justify-between font-mono text-[10px] text-ink-faint">
        <span>{sensor.min}</span>
        <span>{sensor.max}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Output gauge — semicircular dial + status card
// ---------------------------------------------------------------------------

function OutputGauge({ result, loading }: { result: PredictResult | null; loading: boolean }) {
  const probability = result?.probability_failure ?? 0;
  const angle = useMemo(() => -90 + probability * 180, [probability]);

  const zoneColor = probability >= 0.75 ? "#FF5C5C" : probability >= 0.5 ? "#F2A93B" : "#3ED598";

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 220 130" className="w-full max-w-[260px]">
        <path d="M 10 110 A 100 100 0 0 1 210 110" fill="none" stroke="#262B33" strokeWidth="14" strokeLinecap="round" />
        <path
          d="M 10 110 A 100 100 0 0 1 210 110"
          fill="none"
          stroke={zoneColor}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={`${probability * 314} 314`}
          opacity={0.9}
        />
              </svg>

      <div className="-mt-2 text-center">
        <p className="font-mono text-3xl font-semibold" style={{ color: result ? zoneColor : "#5B6270" }}>
          {result ? `${(probability * 100).toFixed(1)}%` : "—"}
        </p>
        <p className="font-mono text-[11px] tracking-wide text-ink-faint">FAILURE PROBABILITY</p>
      </div>

      <div className="mt-6 w-full space-y-3">
        <StatusRow label="Status" value={loading ? "Running…" : result ? statusLabel(result) : "Awaiting reading"} tone={result ? (result.needs_maintenance ? "danger" : "ok") : "muted"} />
        <StatusRow label="Confidence" value={result ? `${(result.confidence * 100).toFixed(1)}%` : "—"} tone="muted" />
        {result?.needs_manual_review && (
          <p className="rounded border border-amber/30 bg-amber/10 px-3 py-2 text-xs text-amber">
            Low confidence reading — recommend manual review by a technician.
          </p>
        )}
      </div>
    </div>
  );
}

function statusLabel(result: PredictResult) {
  return result.needs_maintenance ? "Maintenance required" : "Normal operation";
}

function StatusRow({ label, value, tone }: { label: string; value: string; tone: "danger" | "ok" | "muted" }) {
  const color = tone === "danger" ? "text-danger" : tone === "ok" ? "text-ok" : "text-ink";
  return (
    <div className="flex items-center justify-between border-t border-line pt-3 text-sm">
      <span className="text-ink-muted">{label}</span>
      <span className={`font-medium ${color}`}>{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Footer
// ---------------------------------------------------------------------------

function Footer() {
  return (
    <footer className="mt-14 border-t border-line pt-5 text-center font-mono text-[11px] text-ink-faint">
      FastAPI · scikit-learn RandomForest · Next.js App Router
    </footer>
  );
}
