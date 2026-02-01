"use client";

/**
 * Dashboard Schematic Route
 *
 * This route handles the /dashboard/ee-design/schematic path and redirects
 * to the main app with appropriate query parameters.
 *
 * URL: /dashboard/ee-design/schematic?projectId=xxx
 * Redirects to: /?projectId=xxx&tab=schematic
 */

import { useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";

export default function SchematicPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const projectId = searchParams.get("projectId");

  useEffect(() => {
    // Build the redirect URL
    const params = new URLSearchParams();
    if (projectId) {
      params.set("projectId", projectId);
    }
    params.set("tab", "schematic");

    // Redirect to main app with params
    router.replace(`/?${params.toString()}`);
  }, [projectId, router]);

  // Show loading state while redirecting
  return (
    <div className="h-screen flex items-center justify-center bg-slate-900">
      <div className="text-center">
        <div className="w-8 h-8 border-2 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
        <p className="text-slate-400">Loading schematic viewer...</p>
      </div>
    </div>
  );
}
