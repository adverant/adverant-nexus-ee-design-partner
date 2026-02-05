-- ============================================================================
-- Migration: 20260205_rls_tenant_isolation.sql
-- Description: Add Row-Level Security (RLS) for multi-tenant isolation
-- Database: nexus_plugins
-- ============================================================================
-- CRITICAL SECURITY: This migration enables RLS on ALL tables to ensure
-- complete tenant isolation. Users can only access their own data.
--
-- Architecture:
-- - projects table: Root entity with owner_id and organization_id
-- - Child tables: Reference projects via project_id
-- - Second-level tables: Reference child tables
--
-- Session Variables (must be set by app before queries):
-- - app.current_user_id: The authenticated user's ID
-- - app.current_organization_id: The user's organization ID
-- ============================================================================

-- ============================================================================
-- STEP 1: Create Helper Functions
-- ============================================================================

-- Get current user ID from session variable
CREATE OR REPLACE FUNCTION public.current_user_id()
RETURNS VARCHAR AS $$
BEGIN
    RETURN NULLIF(current_setting('app.current_user_id', true), '');
EXCEPTION
    WHEN OTHERS THEN RETURN NULL;
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;

COMMENT ON FUNCTION public.current_user_id() IS
'Returns the current user ID from session variable app.current_user_id.
Used by RLS policies for tenant isolation.';

-- Get current organization ID from session variable
CREATE OR REPLACE FUNCTION public.current_organization_id()
RETURNS VARCHAR AS $$
BEGIN
    RETURN NULLIF(current_setting('app.current_organization_id', true), '');
EXCEPTION
    WHEN OTHERS THEN RETURN NULL;
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;

COMMENT ON FUNCTION public.current_organization_id() IS
'Returns the current organization ID from session variable app.current_organization_id.
Used by RLS policies for tenant isolation.';

-- Helper function to check if user owns a project
CREATE OR REPLACE FUNCTION public.user_owns_project(project_uuid UUID)
RETURNS BOOLEAN AS $$
DECLARE
    v_user_id VARCHAR;
    v_org_id VARCHAR;
BEGIN
    v_user_id := public.current_user_id();
    v_org_id := public.current_organization_id();

    -- If no session variables set (admin/system/migration), allow access
    IF v_user_id IS NULL AND v_org_id IS NULL THEN
        RETURN TRUE;
    END IF;

    -- Check ownership: owner, organization member, or collaborator
    RETURN EXISTS (
        SELECT 1 FROM public.projects p
        WHERE p.id = project_uuid
        AND (
            p.owner_id = v_user_id
            OR p.organization_id = v_org_id
            OR (p.collaborators ? v_user_id)  -- JSONB contains check
        )
    );
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;

COMMENT ON FUNCTION public.user_owns_project(UUID) IS
'Returns TRUE if the current user owns or has access to the given project.
Checks: ownership, organization membership, collaborator status.
Returns TRUE if no session context (admin/system access).';

-- ============================================================================
-- STEP 2: Enable RLS on ALL Tables
-- ============================================================================

ALTER TABLE public.projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.schematics ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.pcb_layouts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.simulations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.firmware_projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.ideation_artifacts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.job_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.manufacturing_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.validation_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.hil_test_sequences ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.hil_instruments ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.hil_test_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.hil_captured_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.hil_measurements ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.mapos_iterations ENABLE ROW LEVEL SECURITY;

-- ============================================================================
-- STEP 3: Create RLS Policy for projects (Root Table)
-- ============================================================================

DROP POLICY IF EXISTS projects_tenant_isolation ON public.projects;
CREATE POLICY projects_tenant_isolation ON public.projects
    FOR ALL
    USING (
        -- User owns the project
        (owner_id = public.current_user_id())
        -- Or belongs to the same organization
        OR (organization_id IS NOT NULL AND organization_id = public.current_organization_id())
        -- Or user is a collaborator (JSONB contains check)
        OR (collaborators ? public.current_user_id())
        -- Or no session context (admin/system access)
        OR (public.current_user_id() IS NULL AND public.current_organization_id() IS NULL)
    )
    WITH CHECK (
        (owner_id = public.current_user_id())
        OR (organization_id IS NOT NULL AND organization_id = public.current_organization_id())
        OR (public.current_user_id() IS NULL AND public.current_organization_id() IS NULL)
    );

COMMENT ON POLICY projects_tenant_isolation ON public.projects IS
'Multi-tenant isolation: Users can only see projects they own, belong to their org, or are collaborators on.';

-- ============================================================================
-- STEP 4: Create RLS Policies for Child Tables (project_id reference)
-- ============================================================================

-- Schematics
DROP POLICY IF EXISTS schematics_tenant_isolation ON public.schematics;
CREATE POLICY schematics_tenant_isolation ON public.schematics
    FOR ALL
    USING (public.user_owns_project(project_id))
    WITH CHECK (public.user_owns_project(project_id));

-- PCB Layouts
DROP POLICY IF EXISTS pcb_layouts_tenant_isolation ON public.pcb_layouts;
CREATE POLICY pcb_layouts_tenant_isolation ON public.pcb_layouts
    FOR ALL
    USING (public.user_owns_project(project_id))
    WITH CHECK (public.user_owns_project(project_id));

-- Simulations
DROP POLICY IF EXISTS simulations_tenant_isolation ON public.simulations;
CREATE POLICY simulations_tenant_isolation ON public.simulations
    FOR ALL
    USING (public.user_owns_project(project_id))
    WITH CHECK (public.user_owns_project(project_id));

-- Firmware Projects
DROP POLICY IF EXISTS firmware_projects_tenant_isolation ON public.firmware_projects;
CREATE POLICY firmware_projects_tenant_isolation ON public.firmware_projects
    FOR ALL
    USING (public.user_owns_project(project_id))
    WITH CHECK (public.user_owns_project(project_id));

-- Ideation Artifacts
DROP POLICY IF EXISTS ideation_artifacts_tenant_isolation ON public.ideation_artifacts;
CREATE POLICY ideation_artifacts_tenant_isolation ON public.ideation_artifacts
    FOR ALL
    USING (public.user_owns_project(project_id))
    WITH CHECK (public.user_owns_project(project_id));

-- Job History
DROP POLICY IF EXISTS job_history_tenant_isolation ON public.job_history;
CREATE POLICY job_history_tenant_isolation ON public.job_history
    FOR ALL
    USING (public.user_owns_project(project_id))
    WITH CHECK (public.user_owns_project(project_id));

-- Manufacturing Orders
DROP POLICY IF EXISTS manufacturing_orders_tenant_isolation ON public.manufacturing_orders;
CREATE POLICY manufacturing_orders_tenant_isolation ON public.manufacturing_orders
    FOR ALL
    USING (public.user_owns_project(project_id))
    WITH CHECK (public.user_owns_project(project_id));

-- Validation Results
DROP POLICY IF EXISTS validation_results_tenant_isolation ON public.validation_results;
CREATE POLICY validation_results_tenant_isolation ON public.validation_results
    FOR ALL
    USING (public.user_owns_project(project_id))
    WITH CHECK (public.user_owns_project(project_id));

-- HIL Test Sequences
DROP POLICY IF EXISTS hil_test_sequences_tenant_isolation ON public.hil_test_sequences;
CREATE POLICY hil_test_sequences_tenant_isolation ON public.hil_test_sequences
    FOR ALL
    USING (public.user_owns_project(project_id))
    WITH CHECK (public.user_owns_project(project_id));

-- HIL Instruments
DROP POLICY IF EXISTS hil_instruments_tenant_isolation ON public.hil_instruments;
CREATE POLICY hil_instruments_tenant_isolation ON public.hil_instruments
    FOR ALL
    USING (public.user_owns_project(project_id))
    WITH CHECK (public.user_owns_project(project_id));

-- HIL Test Runs
DROP POLICY IF EXISTS hil_test_runs_tenant_isolation ON public.hil_test_runs;
CREATE POLICY hil_test_runs_tenant_isolation ON public.hil_test_runs
    FOR ALL
    USING (public.user_owns_project(project_id))
    WITH CHECK (public.user_owns_project(project_id));

-- ============================================================================
-- STEP 5: Create RLS Policies for Second-Level Tables
-- ============================================================================

-- HIL Captured Data (references hil_test_runs)
DROP POLICY IF EXISTS hil_captured_data_tenant_isolation ON public.hil_captured_data;
CREATE POLICY hil_captured_data_tenant_isolation ON public.hil_captured_data
    FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM public.hil_test_runs tr
            WHERE tr.id = test_run_id
            AND public.user_owns_project(tr.project_id)
        )
        OR (public.current_user_id() IS NULL AND public.current_organization_id() IS NULL)
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.hil_test_runs tr
            WHERE tr.id = test_run_id
            AND public.user_owns_project(tr.project_id)
        )
        OR (public.current_user_id() IS NULL AND public.current_organization_id() IS NULL)
    );

-- HIL Measurements (references hil_test_runs)
DROP POLICY IF EXISTS hil_measurements_tenant_isolation ON public.hil_measurements;
CREATE POLICY hil_measurements_tenant_isolation ON public.hil_measurements
    FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM public.hil_test_runs tr
            WHERE tr.id = test_run_id
            AND public.user_owns_project(tr.project_id)
        )
        OR (public.current_user_id() IS NULL AND public.current_organization_id() IS NULL)
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.hil_test_runs tr
            WHERE tr.id = test_run_id
            AND public.user_owns_project(tr.project_id)
        )
        OR (public.current_user_id() IS NULL AND public.current_organization_id() IS NULL)
    );

-- MAPOS Iterations (references pcb_layouts)
DROP POLICY IF EXISTS mapos_iterations_tenant_isolation ON public.mapos_iterations;
CREATE POLICY mapos_iterations_tenant_isolation ON public.mapos_iterations
    FOR ALL
    USING (
        EXISTS (
            SELECT 1 FROM public.pcb_layouts pl
            WHERE pl.id = pcb_layout_id
            AND public.user_owns_project(pl.project_id)
        )
        OR (public.current_user_id() IS NULL AND public.current_organization_id() IS NULL)
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.pcb_layouts pl
            WHERE pl.id = pcb_layout_id
            AND public.user_owns_project(pl.project_id)
        )
        OR (public.current_user_id() IS NULL AND public.current_organization_id() IS NULL)
    );

-- ============================================================================
-- STEP 6: Grant Permissions
-- ============================================================================

GRANT EXECUTE ON FUNCTION public.current_user_id() TO PUBLIC;
GRANT EXECUTE ON FUNCTION public.current_organization_id() TO PUBLIC;
GRANT EXECUTE ON FUNCTION public.user_owns_project(UUID) TO PUBLIC;

-- ============================================================================
-- VERIFICATION QUERIES (uncomment to test)
-- ============================================================================
-- SELECT tablename, rowsecurity FROM pg_tables WHERE schemaname = 'public';
-- SELECT tablename, policyname FROM pg_policies WHERE schemaname = 'public';
