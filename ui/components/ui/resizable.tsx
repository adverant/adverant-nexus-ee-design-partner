"use client";

import * as React from "react";
import {
  PanelGroup,
  Panel,
  PanelResizeHandle,
} from "react-resizable-panels";
import { cn } from "@/lib/utils";

const ResizablePanelGroup = React.forwardRef<
  React.ElementRef<typeof PanelGroup>,
  React.ComponentPropsWithoutRef<typeof PanelGroup>
>(({ className, ...props }, ref) => (
  <PanelGroup
    ref={ref}
    className={cn("flex h-full w-full", className)}
    {...props}
  />
));
ResizablePanelGroup.displayName = "ResizablePanelGroup";

const ResizablePanel = React.forwardRef<
  React.ElementRef<typeof Panel>,
  React.ComponentPropsWithoutRef<typeof Panel>
>(({ className, ...props }, ref) => (
  <Panel ref={ref} className={cn("", className)} {...props} />
));
ResizablePanel.displayName = "ResizablePanel";

function ResizableHandle({
  className,
  withHandle = false,
  ...props
}: React.ComponentPropsWithoutRef<typeof PanelResizeHandle> & {
  withHandle?: boolean;
}) {
  return (
    <PanelResizeHandle
      className={cn(
        "relative flex w-px items-center justify-center bg-slate-700 after:absolute after:inset-y-0 after:left-1/2 after:w-1 after:-translate-x-1/2 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary-500 data-[panel-group-direction=vertical]:h-px data-[panel-group-direction=vertical]:w-full data-[panel-group-direction=vertical]:after:left-0 data-[panel-group-direction=vertical]:after:h-1 data-[panel-group-direction=vertical]:after:w-full data-[panel-group-direction=vertical]:after:-translate-y-1/2 data-[panel-group-direction=vertical]:after:translate-x-0 [&[data-panel-group-direction=vertical]>div]:rotate-90 hover:bg-primary-500 transition-colors",
        className
      )}
      {...props}
    >
      {withHandle && (
        <div className="z-10 flex h-4 w-3 items-center justify-center rounded-sm border border-slate-600 bg-slate-800">
          <svg
            className="h-2.5 w-2.5 text-slate-400"
            viewBox="0 0 6 10"
            fill="currentColor"
          >
            <circle cx="1" cy="2" r="1" />
            <circle cx="1" cy="5" r="1" />
            <circle cx="1" cy="8" r="1" />
            <circle cx="5" cy="2" r="1" />
            <circle cx="5" cy="5" r="1" />
            <circle cx="5" cy="8" r="1" />
          </svg>
        </div>
      )}
    </PanelResizeHandle>
  );
}

export { ResizablePanelGroup, ResizablePanel, ResizableHandle };