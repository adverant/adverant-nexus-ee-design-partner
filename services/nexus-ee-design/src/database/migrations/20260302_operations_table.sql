-- Operations persistence table
-- Stores in-flight and historical operations so they survive pod restarts
-- and appear in the Operations Center history with restart capability.
--
-- Key design decisions:
--   * project_id is TEXT (no FK) — Trigger.dev + external ops may not have a
--     matching row in the projects table.
--   * status includes 'interrupted' — set on server startup for any 'running'
--     operations that were orphaned by a pod restart.
--   * parameters/subsystems stored for replay/restart capability.

CREATE TABLE IF NOT EXISTS operations (
  id              TEXT PRIMARY KEY,
  project_id      TEXT NOT NULL,
  project_name    TEXT,
  organization_id TEXT,
  owner           TEXT,

  type            TEXT NOT NULL DEFAULT 'schematic',
  source          TEXT NOT NULL DEFAULT 'ee-design',

  -- Status: running | completed | failed | cancelled | interrupted
  status          TEXT NOT NULL DEFAULT 'running',
  progress        INTEGER DEFAULT 0,
  current_step    TEXT DEFAULT '',
  phase           TEXT,

  started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at    TIMESTAMPTZ,
  duration_ms     INTEGER,

  completed_phases TEXT[] DEFAULT '{}',
  quality_gates   JSONB,
  result_data     JSONB,
  error_message   TEXT,
  interrupt_reason TEXT,

  -- Replay support: original pipeline parameters
  parameters      JSONB DEFAULT '{}',
  subsystems      JSONB DEFAULT '[]',

  -- Last 50 events (for detail view of historical operations)
  event_history   JSONB DEFAULT '[]',

  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_operations_project_id ON operations(project_id);
CREATE INDEX IF NOT EXISTS idx_operations_status ON operations(status);
CREATE INDEX IF NOT EXISTS idx_operations_owner ON operations(owner);
CREATE INDEX IF NOT EXISTS idx_operations_created_at ON operations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_operations_completed_at ON operations(completed_at DESC);

-- Auto-update updated_at trigger
CREATE OR REPLACE FUNCTION update_operations_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_operations_updated_at ON operations;
CREATE TRIGGER trigger_operations_updated_at
  BEFORE UPDATE ON operations
  FOR EACH ROW
  EXECUTE FUNCTION update_operations_updated_at();
