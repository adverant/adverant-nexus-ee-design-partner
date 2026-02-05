/**
 * CLI Access Middleware
 *
 * Grants full access to terminal/CLI clients including Claude Code CLI.
 * Automatically identifies CLI requests and assigns appropriate permissions.
 */

import { Request, Response, NextFunction } from 'express';
import { config } from '../config.js';
import { log } from '../utils/logger.js';

// Permission levels for CLI access
export interface CLIPermissions {
  read: boolean;
  write: boolean;
  execute: boolean;
  admin: boolean;
  userId: string;
  userAgent: string;
  clientIP: string;
  isCLI: boolean;
}

// Extend Express Request to include CLI permissions
declare global {
  namespace Express {
    interface Request {
      cliPermissions?: CLIPermissions;
    }
  }
}

/**
 * Detects if the request is from a CLI/terminal client
 */
function isCLIClient(req: Request): boolean {
  const userAgent = (req.headers['user-agent'] || '').toLowerCase();
  const { allowedUserAgents } = config.cliAccess;

  // Check for known CLI user agents
  for (const agent of allowedUserAgents) {
    if (userAgent.includes(agent.toLowerCase())) {
      return true;
    }
  }

  // Check for explicit CLI header
  if (req.headers['x-cli-access'] === 'true') {
    return true;
  }

  // Check for Claude Code CLI header
  if (req.headers['x-claude-code'] === 'true') {
    return true;
  }

  // Check for Nexus CLI header
  if (req.headers['x-nexus-cli'] === 'true') {
    return true;
  }

  // SECURITY FIX: Empty User-Agent is NOT automatically trusted
  // Must have explicit CLI headers or trusted IP to be considered CLI client
  // An empty User-Agent alone should not grant CLI access
  if (!userAgent || userAgent === '') {
    return false;
  }

  return false;
}

/**
 * Gets the client IP address from the request
 */
function getClientIP(req: Request): string {
  const forwarded = req.headers['x-forwarded-for'];
  if (forwarded) {
    const ips = (typeof forwarded === 'string' ? forwarded : forwarded[0]).split(',');
    return ips[0].trim();
  }
  return req.socket.remoteAddress || req.ip || 'unknown';
}

/**
 * Checks if the client IP is trusted
 */
function isTrustedIP(clientIP: string): boolean {
  const { trustedIPs } = config.cliAccess;

  // Normalize the IP for comparison
  const normalizedIP = clientIP.replace('::ffff:', '');

  for (const trustedIP of trustedIPs) {
    if (normalizedIP === trustedIP || normalizedIP.includes(trustedIP)) {
      return true;
    }
  }

  return false;
}

/**
 * CLI Access Middleware
 *
 * Attaches CLI permissions to the request object for downstream handlers.
 * Grants full access to recognized CLI clients from trusted IPs.
 */
export function cliAccessMiddleware(req: Request, res: Response, next: NextFunction): void {
  if (!config.cliAccess.enabled) {
    next();
    return;
  }

  const userAgent = req.headers['user-agent'] || '';
  const clientIP = getClientIP(req);
  const isCLI = isCLIClient(req);
  const trusted = isTrustedIP(clientIP);

  // Build permissions object
  const permissions: CLIPermissions = {
    read: false,
    write: false,
    execute: false,
    admin: false,
    userId: (req.headers['x-user-id'] as string) || 'anonymous',
    userAgent,
    clientIP,
    isCLI,
  };

  if (isCLI && trusted) {
    // Grant full permissions to trusted CLI clients
    permissions.read = config.cliAccess.permissions.read;
    permissions.write = config.cliAccess.permissions.write;
    permissions.execute = config.cliAccess.permissions.execute;
    permissions.admin = config.cliAccess.permissions.admin;
    permissions.userId = (req.headers['x-user-id'] as string) || config.cliAccess.systemUserId;

    // Set the user ID header if not already set
    if (!req.headers['x-user-id']) {
      req.headers['x-user-id'] = permissions.userId;
    }

    log.debug('CLI access granted', {
      userId: permissions.userId,
      clientIP,
      userAgent: userAgent.substring(0, 50),
      path: req.path,
      method: req.method,
    });
  } else if (isCLI) {
    // CLI client from untrusted IP - grant read-only by default
    permissions.read = true;

    log.debug('CLI access (read-only) from untrusted IP', {
      clientIP,
      userAgent: userAgent.substring(0, 50),
      path: req.path,
    });
  }

  // Attach permissions to request
  req.cliPermissions = permissions;

  next();
}

/**
 * Require CLI read permission
 */
export function requireCLIRead(req: Request, res: Response, next: NextFunction): void {
  if (!req.cliPermissions?.read && !req.cliPermissions?.admin) {
    res.status(403).json({
      error: 'CLI read permission required',
      code: 'CLI_READ_REQUIRED',
    });
    return;
  }
  next();
}

/**
 * Require CLI write permission
 */
export function requireCLIWrite(req: Request, res: Response, next: NextFunction): void {
  if (!req.cliPermissions?.write && !req.cliPermissions?.admin) {
    res.status(403).json({
      error: 'CLI write permission required',
      code: 'CLI_WRITE_REQUIRED',
    });
    return;
  }
  next();
}

/**
 * Require CLI execute permission
 */
export function requireCLIExecute(req: Request, res: Response, next: NextFunction): void {
  if (!req.cliPermissions?.execute && !req.cliPermissions?.admin) {
    res.status(403).json({
      error: 'CLI execute permission required',
      code: 'CLI_EXECUTE_REQUIRED',
    });
    return;
  }
  next();
}

/**
 * Require CLI admin permission
 */
export function requireCLIAdmin(req: Request, res: Response, next: NextFunction): void {
  if (!req.cliPermissions?.admin) {
    res.status(403).json({
      error: 'CLI admin permission required',
      code: 'CLI_ADMIN_REQUIRED',
    });
    return;
  }
  next();
}

/**
 * Helper to check if request has full CLI access
 */
export function hasFullCLIAccess(req: Request): boolean {
  const perms = req.cliPermissions;
  if (!perms) return false;
  return perms.read && perms.write && perms.execute && perms.admin;
}

/**
 * Get CLI user ID from request
 */
export function getCLIUserId(req: Request): string {
  return req.cliPermissions?.userId || (req.headers['x-user-id'] as string) || 'system';
}
