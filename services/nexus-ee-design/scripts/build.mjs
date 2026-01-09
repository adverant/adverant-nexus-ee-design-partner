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

// Plugin to add .js extension to relative imports
const addJsExtensionPlugin = {
  name: 'add-js-extension',
  setup(build) {
    build.onEnd(async (result) => {
      if (result.errors.length > 0) return;

      // Process all output files
      const processFile = async (dir) => {
        const entries = await fs.promises.readdir(dir, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          if (entry.isDirectory()) {
            await processFile(fullPath);
          } else if (entry.isFile() && fullPath.endsWith('.js')) {
            let content = await fs.promises.readFile(fullPath, 'utf8');
            // Add .js extension to relative imports that don't have it
            content = content.replace(
              /from\s+["'](\.[^"']+)["']/g,
              (match, importPath) => {
                if (!importPath.endsWith('.js') && !importPath.endsWith('.json')) {
                  return `from "${importPath}.js"`;
                }
                return match;
              }
            );
            // Also handle import statements
            content = content.replace(
              /import\s+["'](\.[^"']+)["']/g,
              (match, importPath) => {
                if (!importPath.endsWith('.js') && !importPath.endsWith('.json')) {
                  return `import "${importPath}.js"`;
                }
                return match;
              }
            );
            await fs.promises.writeFile(fullPath, content);
          }
        }
      };

      await processFile(outDir);
    });
  }
};

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
      plugins: [addJsExtensionPlugin],
    });

    console.log('Build completed successfully!');
  } catch (error) {
    console.error('Build failed:', error);
    process.exit(1);
  }
}

build();
