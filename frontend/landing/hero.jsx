import React from "react";
import { createRoot } from "react-dom/client";

function Sparkline({ points }) {
  const line = points.map(([x, y]) => `${x},${y / 2}`).join(" ");
  return <svg viewBox="0 0 100 50" role="img" aria-label="近三十个交易日收盘价走势" className="w-[45%] overflow-visible">
    <defs><linearGradient id="helix-area" x2="0" y2="1">
      <stop stopColor="#22d3ee" stopOpacity=".42" />
      <stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
    </linearGradient></defs>
    <polygon points={`0,50 ${line} 100,50`} fill="url(#helix-area)" />
    <polyline points={line} fill="none" stroke="#67e8f9" strokeWidth="1.5" strokeLinejoin="round" />
  </svg>;
}

function Score({ score, days, horizon }) {
  return <div className="helix-score">
    <div className="helix-ring" style={{ "--score": `${score ?? 0}%` }}>
      <div><small>独立测试</small><strong>{score == null ? "—" : score.toFixed(1)}</strong><small>/ 100</small></div>
    </div>
    <div className="helix-score-details">
      <div>有效测试日 <b>{days ?? "—"}</b></div>
      <div>预测周期 <b>{horizon == null ? "—" : `${horizon} 日`}</b></div>
      <div>随机排序 <b>50 分</b></div>
    </div>
  </div>;
}

function Hero({ market, stats, score, days, horizon }) {
  return <>
    <h1>Helix 赫力</h1>
    <p className="helix-tagline">研究场景 · 从假设到验证，<strong>AI自主完成量价研究全流程</strong></p>
    <div className="helix-market-strip">
      <div className="helix-quote">
        <div><small>行情 · {market.symbol}</small><strong>{market.close}</strong><small className={market.tone}>{market.change}</small></div>
        <Sparkline points={market.points} />
      </div>
      <div className="helix-stat-group">
        {stats.map(([label, value]) => <div key={label} className="helix-stat bg-white/5 transition-[transform,background-color] duration-300 hover:-translate-y-1 hover:bg-white/10 motion-reduce:transition-none motion-reduce:hover:translate-y-0">
          <small>{label}</small><strong>{value}</strong>
        </div>)}
      </div>
      <Score score={score} days={days} horizon={horizon} />
    </div>
  </>;
}

const roots = new WeakMap();
export default function mount({ parentElement, data }) {
  const host = parentElement.querySelector(".helix-react-hero");
  if (!roots.has(host)) roots.set(host, createRoot(host));
  roots.get(host).render(<Hero {...data} />);
}
