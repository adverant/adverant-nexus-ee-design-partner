"use client";

import { Cpu, Zap, Settings, HelpCircle } from "lucide-react";

export function Header() {
  return (
    <header className="h-14 border-b border-slate-700 bg-surface-primary flex items-center justify-between px-4">
      {/* Logo and Title */}
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary-600">
          <Cpu className="w-5 h-5 text-white" />
        </div>
        <div>
          <h1 className="text-lg font-semibold text-white">EE Design Partner</h1>
          <p className="text-xs text-slate-400">Hardware/Software Development Automation</p>
        </div>
      </div>

      {/* Status Indicators */}
      <div className="flex items-center gap-6">
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1.5 text-sm">
            <Zap className="w-4 h-4 text-green-400" />
            <span className="text-slate-300">Claude Opus 4</span>
          </div>
          <span className="text-slate-600">|</span>
          <div className="flex items-center gap-1.5 text-sm">
            <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-slate-300">Gemini 2.5 Pro</span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
          <button className="p-2 rounded-lg hover:bg-surface-secondary transition-colors text-slate-400 hover:text-white">
            <HelpCircle className="w-5 h-5" />
          </button>
          <button className="p-2 rounded-lg hover:bg-surface-secondary transition-colors text-slate-400 hover:text-white">
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  );
}