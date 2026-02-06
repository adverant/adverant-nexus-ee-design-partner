/**
 * Nexus EE Design API - Main Router
 *
 * Registers all API routes for the schematic generation service.
 */

import express, { Router } from 'express';
import symbolAssemblyRouter from './routes/symbol-assembly';
import complianceRouter from './routes/compliance';

const router: Router = express.Router();

// Health check
router.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'nexus-ee-design',
    version: '3.0.0',
    timestamp: new Date().toISOString(),
  });
});

// MAPO v3.0 Routes
router.use('/symbol-assembly', symbolAssemblyRouter);
router.use('/compliance', complianceRouter);

export default router;
