"use client";

/**
 * Dashboard Firmware Route
 *
 * URL: /dashboard/ee-design/firmware?projectId=xxx
 * Redirects to: /?projectId=xxx&tab=testing (bench testing tab)
 */

import { useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";

export default function FirmwarePage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const projectId = searchParams?.get("projectId") ?? null;

  useEffect(() => {
    const params = new URLSearchParams();
    if (projectId) {
      params.set("projectId", projectId);
    }
    params.set("tab", "testing");
    router.replace(`/?${params.toString()}`);
  }, [projectId, router]);

  return (
    <div className="h-screen flex items-center justify-center bg-slate-900">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-slate-400">Loading Firmware tools...</p>
      </div>
    </div>
  );
}
