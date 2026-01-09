/**
 * EE Design Partner - Database Connection Module
 *
 * PostgreSQL connection pool management with transaction support,
 * query helpers, and proper error handling.
 */

import { Pool, PoolClient, QueryResult, QueryResultRow } from 'pg';
import { config } from '../config.js';
import { log, Logger } from '../utils/logger.js';
import { EEDesignError } from '../utils/errors.js';

// ============================================================================
// Types
// ============================================================================

export interface QueryOptions {
  /** Query timeout in milliseconds */
  timeout?: number;
  /** Request ID for logging correlation */
  requestId?: string;
  /** Operation name for logging */
  operation?: string;
}

export interface TransactionOptions extends QueryOptions {
  /** Isolation level for the transaction */
  isolationLevel?: 'READ COMMITTED' | 'REPEATABLE READ' | 'SERIALIZABLE';
}

export type TransactionCallback<T> = (client: PoolClient) => Promise<T>;

// ============================================================================
// Database Error Class
// ============================================================================

export class DatabaseError extends EEDesignError {
  public readonly pgCode?: string;
  public readonly constraint?: string;
  public readonly table?: string;
  public readonly column?: string;

  constructor(
    message: string,
    originalError: Error & {
      code?: string;
      constraint?: string;
      table?: string;
      column?: string;
    },
    context: { operation: string; requestId?: string }
  ) {
    super(
      message,
      'DATABASE_ERROR',
      mapPgErrorToHttpStatus(originalError.code),
      {
        operation: context.operation,
        requestId: context.requestId,
        pgCode: originalError.code,
        originalMessage: originalError.message,
      }
    );
    this.pgCode = originalError.code;
    this.constraint = originalError.constraint;
    this.table = originalError.table;
    this.column = originalError.column;
  }
}

/**
 * Maps PostgreSQL error codes to HTTP status codes
 */
function mapPgErrorToHttpStatus(pgCode?: string): number {
  if (!pgCode) return 500;

  // Class 23: Integrity Constraint Violation
  if (pgCode.startsWith('23')) {
    switch (pgCode) {
      case '23505': // unique_violation
        return 409; // Conflict
      case '23503': // foreign_key_violation
        return 400; // Bad Request
      case '23502': // not_null_violation
        return 400; // Bad Request
      case '23514': // check_violation
        return 400; // Bad Request
      default:
        return 400;
    }
  }

  // Class 42: Syntax Error or Access Rule Violation
  if (pgCode.startsWith('42')) {
    return 500;
  }

  // Class 53: Insufficient Resources
  if (pgCode.startsWith('53')) {
    return 503; // Service Unavailable
  }

  // Class 57: Operator Intervention
  if (pgCode.startsWith('57')) {
    return 503;
  }

  // Class 08: Connection Exception
  if (pgCode.startsWith('08')) {
    return 503;
  }

  return 500;
}

// ============================================================================
// Connection Pool
// ============================================================================

let pool: Pool | null = null;
let dbLogger: Logger;

/**
 * Initialize the database connection pool
 */
function initializePool(): Pool {
  const { postgres } = config;

  dbLogger = log.child({ service: 'database' });

  const newPool = new Pool({
    host: postgres.host,
    port: postgres.port,
    database: postgres.database,
    user: postgres.user,
    password: postgres.password,
    ssl: postgres.ssl ? { rejectUnauthorized: false } : false,

    // Pool configuration
    max: 20, // Maximum number of clients in the pool
    min: 2, // Minimum number of clients to keep idle
    idleTimeoutMillis: 30000, // Close idle clients after 30 seconds
    connectionTimeoutMillis: 10000, // Return an error after 10 seconds if connection cannot be established
    maxUses: 7500, // Close a connection after it has been used 7500 times

    // Statement timeout
    statement_timeout: 30000, // 30 seconds default statement timeout
  });

  // Pool event handlers
  newPool.on('connect', (client) => {
    dbLogger.debug('New database client connected', {
      totalCount: newPool.totalCount,
      idleCount: newPool.idleCount,
      waitingCount: newPool.waitingCount,
    });

    // Set session-level configuration
    client.query('SET timezone = \'UTC\'').catch((err) => {
      dbLogger.warn('Failed to set timezone', { error: err.message });
    });
  });

  newPool.on('acquire', () => {
    dbLogger.debug('Client acquired from pool', {
      totalCount: newPool.totalCount,
      idleCount: newPool.idleCount,
      waitingCount: newPool.waitingCount,
    });
  });

  newPool.on('release', () => {
    dbLogger.debug('Client released to pool', {
      totalCount: newPool.totalCount,
      idleCount: newPool.idleCount,
      waitingCount: newPool.waitingCount,
    });
  });

  newPool.on('remove', () => {
    dbLogger.debug('Client removed from pool', {
      totalCount: newPool.totalCount,
      idleCount: newPool.idleCount,
      waitingCount: newPool.waitingCount,
    });
  });

  newPool.on('error', (err) => {
    dbLogger.error('Unexpected pool error', err, {
      operation: 'pool_error',
    });
  });

  dbLogger.info('Database connection pool initialized', {
    host: postgres.host,
    port: postgres.port,
    database: postgres.database,
    user: postgres.user,
    maxConnections: 20,
  });

  return newPool;
}

// ============================================================================
// Public API
// ============================================================================

/**
 * Get the database connection pool.
 * Initializes the pool on first call.
 */
export function getPool(): Pool {
  if (!pool) {
    pool = initializePool();
  }
  return pool;
}

/**
 * Execute a query against the database.
 *
 * @param text - SQL query string with $1, $2 style parameters
 * @param values - Parameter values for the query
 * @param options - Query options
 * @returns Query result
 */
export async function query<T extends QueryResultRow = QueryResultRow>(
  text: string,
  values?: unknown[],
  options: QueryOptions = {}
): Promise<QueryResult<T>> {
  const pool = getPool();
  const startTime = Date.now();
  const { timeout, requestId, operation = 'query' } = options;

  const queryLogger = dbLogger.child({
    requestId,
    operation,
  });

  // Build query configuration
  const queryConfig: { text: string; values?: unknown[]; query_timeout?: number } = {
    text,
    values,
  };

  if (timeout) {
    queryConfig.query_timeout = timeout;
  }

  try {
    queryLogger.debug('Executing query', {
      queryLength: text.length,
      paramCount: values?.length ?? 0,
    });

    const result = await pool.query<T>(queryConfig);

    const duration = Date.now() - startTime;
    queryLogger.debug('Query completed', {
      rowCount: result.rowCount,
      duration,
    });

    return result;
  } catch (error) {
    const duration = Date.now() - startTime;
    const pgError = error as Error & {
      code?: string;
      constraint?: string;
      table?: string;
      column?: string;
    };

    queryLogger.error('Query failed', pgError, {
      duration,
      pgCode: pgError.code,
    });

    throw new DatabaseError(
      `Database query failed: ${pgError.message}`,
      pgError,
      { operation, requestId }
    );
  }
}

/**
 * Execute multiple queries within a transaction.
 *
 * The transaction is automatically committed on success or rolled back on error.
 *
 * @param callback - Async function receiving a client to execute queries
 * @param options - Transaction options
 * @returns Result of the callback function
 */
export async function withTransaction<T>(
  callback: TransactionCallback<T>,
  options: TransactionOptions = {}
): Promise<T> {
  const pool = getPool();
  const startTime = Date.now();
  const { isolationLevel, requestId, operation = 'transaction' } = options;

  const txLogger = dbLogger.child({
    requestId,
    operation,
  });

  const client = await pool.connect();

  try {
    // Begin transaction with optional isolation level
    const beginStatement = isolationLevel
      ? `BEGIN TRANSACTION ISOLATION LEVEL ${isolationLevel}`
      : 'BEGIN';

    await client.query(beginStatement);
    txLogger.debug('Transaction started', { isolationLevel });

    // Execute the callback
    const result = await callback(client);

    // Commit transaction
    await client.query('COMMIT');

    const duration = Date.now() - startTime;
    txLogger.debug('Transaction committed', { duration });

    return result;
  } catch (error) {
    // Rollback on any error
    try {
      await client.query('ROLLBACK');
      txLogger.debug('Transaction rolled back');
    } catch (rollbackError) {
      txLogger.error('Rollback failed', rollbackError as Error);
    }

    const duration = Date.now() - startTime;
    const pgError = error as Error & {
      code?: string;
      constraint?: string;
      table?: string;
      column?: string;
    };

    txLogger.error('Transaction failed', pgError, {
      duration,
      pgCode: pgError.code,
    });

    // Re-throw if already a DatabaseError
    if (error instanceof DatabaseError) {
      throw error;
    }

    throw new DatabaseError(
      `Transaction failed: ${pgError.message}`,
      pgError,
      { operation, requestId }
    );
  } finally {
    client.release();
  }
}

/**
 * Execute a query within a transaction client.
 * Helper for use inside withTransaction callbacks.
 *
 * @param client - Transaction client
 * @param text - SQL query string
 * @param values - Parameter values
 * @returns Query result
 */
export async function clientQuery<T extends QueryResultRow = QueryResultRow>(
  client: PoolClient,
  text: string,
  values?: unknown[]
): Promise<QueryResult<T>> {
  return client.query<T>(text, values);
}

/**
 * Check if the database connection is healthy.
 *
 * @returns True if the database is reachable and responding
 */
export async function healthCheck(): Promise<boolean> {
  try {
    const result = await query('SELECT 1 as health_check', [], {
      timeout: 5000,
      operation: 'health_check',
    });
    return result.rows.length === 1 && result.rows[0].health_check === 1;
  } catch (error) {
    dbLogger.error('Health check failed', error as Error);
    return false;
  }
}

/**
 * Get database connection pool statistics.
 *
 * @returns Pool statistics
 */
export function getPoolStats(): {
  totalCount: number;
  idleCount: number;
  waitingCount: number;
} {
  const p = getPool();
  return {
    totalCount: p.totalCount,
    idleCount: p.idleCount,
    waitingCount: p.waitingCount,
  };
}

/**
 * Close the database connection pool.
 * Should be called during graceful shutdown.
 */
export async function closePool(): Promise<void> {
  if (pool) {
    dbLogger.info('Closing database connection pool');
    await pool.end();
    pool = null;
    dbLogger.info('Database connection pool closed');
  }
}

/**
 * Execute a raw query (for schema migrations, etc.)
 * Use with caution - no parameter escaping.
 *
 * @param sql - Raw SQL to execute
 * @returns Query result
 */
export async function rawQuery<T extends QueryResultRow = QueryResultRow>(
  sql: string
): Promise<QueryResult<T>> {
  const pool = getPool();
  return pool.query<T>(sql);
}

// ============================================================================
// Query Builder Helpers
// ============================================================================

/**
 * Build a parameterized INSERT statement with RETURNING.
 *
 * @param table - Table name
 * @param data - Object with column names as keys
 * @returns Object with text and values for query()
 */
export function buildInsert(
  table: string,
  data: Record<string, unknown>
): { text: string; values: unknown[] } {
  const columns = Object.keys(data);
  const values = Object.values(data);
  const placeholders = columns.map((_, i) => `$${i + 1}`);

  const text = `
    INSERT INTO ${table} (${columns.join(', ')})
    VALUES (${placeholders.join(', ')})
    RETURNING *
  `;

  return { text, values };
}

/**
 * Build a parameterized UPDATE statement with RETURNING.
 *
 * @param table - Table name
 * @param data - Object with column names as keys to update
 * @param whereColumn - Column name for WHERE clause
 * @param whereValue - Value for WHERE clause
 * @returns Object with text and values for query()
 */
export function buildUpdate(
  table: string,
  data: Record<string, unknown>,
  whereColumn: string,
  whereValue: unknown
): { text: string; values: unknown[] } {
  const columns = Object.keys(data);
  const values = Object.values(data);

  const setClause = columns
    .map((col, i) => `${col} = $${i + 1}`)
    .join(', ');

  const whereParamIndex = columns.length + 1;
  values.push(whereValue);

  const text = `
    UPDATE ${table}
    SET ${setClause}
    WHERE ${whereColumn} = $${whereParamIndex}
    RETURNING *
  `;

  return { text, values };
}

/**
 * Build a parameterized SELECT statement.
 *
 * @param table - Table name
 * @param columns - Array of column names (or '*')
 * @param where - Object with column/value pairs for WHERE clause
 * @param orderBy - Optional ORDER BY clause
 * @param limit - Optional LIMIT
 * @param offset - Optional OFFSET
 * @returns Object with text and values for query()
 */
export function buildSelect(
  table: string,
  columns: string[] | '*',
  where?: Record<string, unknown>,
  orderBy?: string,
  limit?: number,
  offset?: number
): { text: string; values: unknown[] } {
  const selectClause = columns === '*' ? '*' : columns.join(', ');
  const values: unknown[] = [];
  let text = `SELECT ${selectClause} FROM ${table}`;

  if (where && Object.keys(where).length > 0) {
    const conditions = Object.entries(where).map(([col, val], i) => {
      values.push(val);
      return `${col} = $${i + 1}`;
    });
    text += ` WHERE ${conditions.join(' AND ')}`;
  }

  if (orderBy) {
    text += ` ORDER BY ${orderBy}`;
  }

  if (limit !== undefined) {
    values.push(limit);
    text += ` LIMIT $${values.length}`;
  }

  if (offset !== undefined) {
    values.push(offset);
    text += ` OFFSET $${values.length}`;
  }

  return { text, values };
}

// ============================================================================
// Exports
// ============================================================================

export default {
  getPool,
  query,
  withTransaction,
  clientQuery,
  healthCheck,
  getPoolStats,
  closePool,
  rawQuery,
  buildInsert,
  buildUpdate,
  buildSelect,
};
