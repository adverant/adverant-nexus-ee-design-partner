/**
 * Build script using esbuild for fast, type-error-tolerant builds
 * This transpiles TypeScript without type checking for deployment
 */

import * as esbuild from 'esbuild';
import * as path from 'node:path';
import * as fs from 'node:fs';

const srcDir = './src';
const outDir = './dist';

async function findFilesRecursive(dir, matches = []) {
  const entries = await fs.promises.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      await findFilesRecursive(fullPath, matches);
    } else if (entry.isFile() && entry.name.endsWith('.ts') && !entry.name.endsWith('.test.ts') && !entry.name.endsWith('.spec.ts')) {
      matches.push(fullPath);
    }
  }
  return matches;
}

async function build() {
  console.log('Building nexus-ee-design with esbuild...');

  // Clean output directory
  await fs.promises.rm(outDir, { recursive: true, force: true });
  await fs.promises.mkdir(outDir, { recursive: true });

  // Find all TypeScript files
  const entryPoints = await findFilesRecursive(srcDir);
  console.log(`Found ${entryPoints.length} TypeScript files`);

  try {
    await esbuild.build({
      entryPoints,
      outdir: outDir,
      bundle: false,
      platform: 'node',
      target: 'node20',
      format: 'esm',
      sourcemap: true,
      outExtension: { '.js': '.js' },
      // Preserve directory structure
      outbase: srcDir,
      // Handle .js extensions in imports
      resolveExtensions: ['.ts', '.js'],
    });

    console.log('Build completed successfully!');
  } catch (error) {
    console.error('Build failed:', error);
    process.exit(1);
  }
}

build();
