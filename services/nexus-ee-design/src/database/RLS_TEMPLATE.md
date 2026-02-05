# Row-Level Security (RLS) Template for Nexus Plugins

## MANDATORY: All Nexus Plugins MUST Implement RLS

This template provides the standard pattern for implementing multi-tenant isolation
in any Nexus plugin. **Failure to implement RLS is a critical security vulnerability.**

---

## Quick Start Checklist

- [ ] Create helper functions (`current_user_id`, `current_organization_id`)
- [ ] Enable RLS on ALL tenant-scoped tables
- [ ] Create policies for root entity (e.g., `projects`)
- [ ] Create policies for child entities (reference root via FK)
- [ ] Update database connection to set session variables
- [ ] Add API-level ownership validation middleware
- [ ] Test with different users to verify isolation

---

## 1. Helper Functions

Add these to your migration SQL:

```sql
-- Get current user ID from session variable
CREATE OR REPLACE FUNCTION public.current_user_id()
RETURNS VARCHAR AS $$
BEGIN
    RETURN NULLIF(current_setting('app.current_user_id', true), '');
EXCEPTION
    WHEN OTHERS THEN RETURN NULL;
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;

-- Get current organization ID from session variable
CREATE OR REPLACE FUNCTION public.current_organization_id()
RETURNS VARCHAR AS $$
BEGIN
    RETURN NULLIF(current_setting('app.current_organization_id', true), '');
EXCEPTION
    WHEN OTHERS THEN RETURN NULL;
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;

-- Helper to check ownership of root entity
-- Customize this for your plugin's root entity
CREATE OR REPLACE FUNCTION public.user_owns_<your_entity>(entity_uuid UUID)
RETURNS BOOLEAN AS $$
DECLARE
    v_user_id VARCHAR;
    v_org_id VARCHAR;
BEGIN
    v_user_id := public.current_user_id();
    v_org_id := public.current_organization_id();

    -- If no session context (admin/system), allow access
    IF v_user_id IS NULL AND v_org_id IS NULL THEN
        RETURN TRUE;
    END IF;

    -- Check ownership
    RETURN EXISTS (
        SELECT 1 FROM public.<your_table> t
        WHERE t.id = entity_uuid
        AND (
            t.owner_id = v_user_id
            OR t.organization_id = v_org_id
            -- Add collaborators check if applicable
        )
    );
END;
$$ LANGUAGE plpgsql STABLE SECURITY DEFINER;
```

---

## 2. Enable RLS on Tables

```sql
-- Enable RLS on ALL tables that contain tenant data
ALTER TABLE public.<table_name> ENABLE ROW LEVEL SECURITY;
```

**Important**: RLS must be enabled on EVERY table that stores user/tenant data.

---

## 3. Root Entity Policy

For your main entity (e.g., projects, documents, resources):

```sql
DROP POLICY IF EXISTS <table>_tenant_isolation ON public.<table>;
CREATE POLICY <table>_tenant_isolation ON public.<table>
    FOR ALL
    USING (
        -- User owns the entity
        (owner_id = public.current_user_id())
        -- Or belongs to the same organization
        OR (organization_id IS NOT NULL AND organization_id = public.current_organization_id())
        -- Or no session context (admin/system access)
        OR (public.current_user_id() IS NULL AND public.current_organization_id() IS NULL)
    )
    WITH CHECK (
        (owner_id = public.current_user_id())
        OR (organization_id IS NOT NULL AND organization_id = public.current_organization_id())
        OR (public.current_user_id() IS NULL AND public.current_organization_id() IS NULL)
    );
```

---

## 4. Child Entity Policy

For tables that reference the root entity:

```sql
DROP POLICY IF EXISTS <child_table>_tenant_isolation ON public.<child_table>;
CREATE POLICY <child_table>_tenant_isolation ON public.<child_table>
    FOR ALL
    USING (public.user_owns_<root_entity>(<foreign_key_column>))
    WITH CHECK (public.user_owns_<root_entity>(<foreign_key_column>));
```

---

## 5. Database Connection - Set Session Variables

In your database connection module, add support for RLS context:

```typescript
// Set RLS context before queries
async function setRLSContext(
  client: PoolClient,
  userId?: string,
  organizationId?: string
): Promise<void> {
  if (userId) {
    await client.query(`SET LOCAL app.current_user_id = '${userId.replace(/'/g, "''")}'`);
  }
  if (organizationId) {
    await client.query(`SET LOCAL app.current_organization_id = '${organizationId.replace(/'/g, "''")}'`);
  }
}
```

**Important**: Use `SET LOCAL` so settings only apply to the current transaction.

---

## 6. API-Level Validation (Defense in Depth)

Even with RLS, add API-level validation:

```typescript
// In your router
router.param('entityId', async (req, res, next, value) => {
  const userId = req.headers['x-user-id'] as string;

  // Require authentication
  if (!userId) {
    return res.status(401).json({ error: 'Authentication required' });
  }

  // Fetch and validate ownership
  const entity = await findById(value);
  if (!entity) {
    return res.status(404).json({ error: 'Not found' });
  }

  // Check ownership
  if (entity.owner_id !== userId) {
    return res.status(403).json({ error: 'Access denied' });
  }

  req.entity = entity;
  next();
});
```

---

## 7. Required Headers

All API requests must include these headers (set by frontend/proxy):

| Header | Description |
|--------|-------------|
| `X-User-ID` | Authenticated user's ID |
| `X-Organization-ID` | User's organization ID |

---

## 8. Testing Checklist

Test these scenarios before deployment:

- [ ] User A cannot see User B's data
- [ ] User A cannot modify User B's data
- [ ] User A cannot delete User B's data
- [ ] API returns 403 for unauthorized access attempts
- [ ] Database returns empty results (not errors) for RLS-filtered queries
- [ ] Admin/system access still works (no session context)
- [ ] Collaborators can access shared resources (if applicable)

---

## Common Mistakes to Avoid

1. **NOT enabling RLS** - Tables without RLS leak all data
2. **Missing helper functions** - Policies fail silently
3. **Not setting session variables** - RLS allows all access
4. **Using `SET` instead of `SET LOCAL`** - Context persists across requests
5. **Forgetting child tables** - Data leaks through related entities
6. **Not validating at API level** - Defense in depth is required

---

## Example Migration Structure

```
migrations/
├── 001_create_tables.sql
├── 002_add_indexes.sql
└── 003_enable_rls.sql  <-- RLS should be its own migration
```

---

## Reference Implementation

See the EE Design Partner plugin for a complete reference:
- Migration: `migrations/20260205_rls_tenant_isolation.sql`
- Connection: `database/connection.ts` (queryWithContext, setRLSContext)
- Routes: `api/routes.ts` (router.param ownership validation)
