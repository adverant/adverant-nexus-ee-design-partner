import { chromium, Browser, Page } from 'playwright';

async function validateIdeationPage() {
  console.log('🔍 Starting validation of https://dashboard.adverant.ai/dashboard/ee-design/ideation\n');

  let browser: Browser | null = null;

  try {
    // Launch browser
    browser = await chromium.launch({
      headless: true,
    });

    const context = await browser.newContext({
      viewport: { width: 1920, height: 1080 },
    });

    const page: Page = await context.newPage();

    // Inject console monitoring code BEFORE navigation
    await page.addInitScript(() => {
      // @ts-ignore
      window.__errors = [];
      // @ts-ignore
      window.__apiCalls = [];
      // @ts-ignore
      window.__consoleErrors = [];
      // @ts-ignore
      window.__consoleWarnings = [];

      // Capture runtime errors
      window.onerror = (msg, src, line, col, err) => {
        // @ts-ignore
        window.__errors.push({
          type: 'runtime',
          msg: String(msg),
          src,
          line,
          col,
          stack: err?.stack,
          timestamp: new Date().toISOString()
        });
        return false;
      };

      // Capture unhandled promise rejections
      window.addEventListener('unhandledrejection', (event) => {
        // @ts-ignore
        window.__errors.push({
          type: 'unhandledRejection',
          msg: event.reason?.message || String(event.reason),
          stack: event.reason?.stack,
          timestamp: new Date().toISOString()
        });
      });

      // Intercept fetch calls
      const origFetch = window.fetch;
      // @ts-ignore
      window.fetch = async (...args: Parameters<typeof fetch>) => {
        const start = Date.now();
        const url = typeof args[0] === 'string' ? args[0] : args[0]?.url || 'unknown';
        try {
          const res = await origFetch(...args);
          // @ts-ignore
          window.__apiCalls.push({
            url,
            method: args[1]?.method || 'GET',
            status: res.status,
            statusText: res.statusText,
            time: Date.now() - start,
            timestamp: new Date().toISOString()
          });
          return res;
        } catch (e: any) {
          // @ts-ignore
          window.__apiCalls.push({
            url,
            method: args[1]?.method || 'GET',
            error: e.message,
            time: Date.now() - start,
            timestamp: new Date().toISOString()
          });
          throw e;
        }
      };

      // Capture console.error
      const origError = console.error;
      console.error = (...args) => {
        // @ts-ignore
        window.__consoleErrors.push({
          args: args.map(a => {
            try {
              return typeof a === 'object' ? JSON.stringify(a) : String(a);
            } catch {
              return String(a);
            }
          }),
          timestamp: new Date().toISOString()
        });
        origError.apply(console, args);
      };

      // Capture console.warn
      const origWarn = console.warn;
      console.warn = (...args) => {
        // @ts-ignore
        window.__consoleWarnings.push({
          args: args.map(a => {
            try {
              return typeof a === 'object' ? JSON.stringify(a) : String(a);
            } catch {
              return String(a);
            }
          }),
          timestamp: new Date().toISOString()
        });
        origWarn.apply(console, args);
      };
    });

    // Also capture console messages from Playwright's perspective
    const playwrightErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        playwrightErrors.push(msg.text());
      }
    });

    page.on('pageerror', error => {
      playwrightErrors.push(`Page Error: ${error.message}\nStack: ${error.stack}`);
    });

    console.log('📡 Navigating to page...');
    const startTime = Date.now();

    // Navigate to the page
    const response = await page.goto('https://dashboard.adverant.ai/dashboard/ee-design/ideation', {
      waitUntil: 'networkidle',
      timeout: 60000,
    });

    const loadTime = Date.now() - startTime;
    console.log(`✅ Page loaded in ${loadTime}ms`);
    console.log(`📊 Response status: ${response?.status()}\n`);

    // Wait a bit for any async operations
    await page.waitForTimeout(3000);

    // Check for "Application error" text
    const applicationError = await page.locator('text=Application error').count();
    const hasApplicationError = applicationError > 0;

    // Get page title
    const title = await page.title();
    console.log(`📄 Page title: "${title}"`);

    // Get the full page text to check for error messages
    const bodyText = await page.locator('body').innerText();

    // Retrieve injected error data
    const errors = await page.evaluate(() => (window as any).__errors || []);
    const apiCalls = await page.evaluate(() => (window as any).__apiCalls || []);
    const consoleErrors = await page.evaluate(() => (window as any).__consoleErrors || []);
    const consoleWarnings = await page.evaluate(() => (window as any).__consoleWarnings || []);

    // Check for TypeErrors
    const typeErrors = errors.filter((e: any) =>
      e.msg?.includes('TypeError') ||
      e.msg?.includes('Cannot read properties of undefined') ||
      e.msg?.includes('Cannot read properties of null')
    );

    // Take a screenshot
    await page.screenshot({
      path: '/Users/don/Adverant/adverant-nexus-ee-design-partner/tests/ideation-page-screenshot.png',
      fullPage: true
    });
    console.log('📸 Screenshot saved to tests/ideation-page-screenshot.png\n');

    // Report findings
    console.log('=' .repeat(80));
    console.log('                         VALIDATION RESULTS');
    console.log('=' .repeat(80));

    // Application Error Check
    console.log('\n🔍 APPLICATION ERROR CHECK:');
    if (hasApplicationError) {
      console.log('   ❌ "Application error" text FOUND on page!');
    } else {
      console.log('   ✅ No "Application error" text found');
    }

    // Runtime Errors
    console.log('\n🔍 RUNTIME ERRORS:');
    if (errors.length === 0) {
      console.log('   ✅ No runtime errors captured');
    } else {
      console.log(`   ❌ ${errors.length} runtime error(s) found:`);
      errors.forEach((e: any, i: number) => {
        console.log(`\n   Error ${i + 1}:`);
        console.log(`     Type: ${e.type}`);
        console.log(`     Message: ${e.msg}`);
        if (e.src) console.log(`     Source: ${e.src}`);
        if (e.line) console.log(`     Line: ${e.line}, Col: ${e.col}`);
        if (e.stack) console.log(`     Stack: ${e.stack.substring(0, 500)}...`);
      });
    }

    // TypeErrors specifically
    console.log('\n🔍 TypeError CHECK (Cannot read properties of undefined/null):');
    if (typeErrors.length === 0) {
      console.log('   ✅ No TypeErrors found');
    } else {
      console.log(`   ❌ ${typeErrors.length} TypeError(s) found:`);
      typeErrors.forEach((e: any, i: number) => {
        console.log(`     ${i + 1}. ${e.msg}`);
      });
    }

    // Console Errors
    console.log('\n🔍 CONSOLE ERRORS:');
    if (consoleErrors.length === 0 && playwrightErrors.length === 0) {
      console.log('   ✅ No console errors');
    } else {
      const allConsoleErrors = [...consoleErrors.map((e: any) => e.args.join(' ')), ...playwrightErrors];
      console.log(`   ⚠️  ${allConsoleErrors.length} console error(s):`);
      allConsoleErrors.slice(0, 10).forEach((e: string, i: number) => {
        console.log(`     ${i + 1}. ${e.substring(0, 200)}${e.length > 200 ? '...' : ''}`);
      });
      if (allConsoleErrors.length > 10) {
        console.log(`     ... and ${allConsoleErrors.length - 10} more`);
      }
    }

    // API Calls
    console.log('\n🔍 API CALLS:');
    if (apiCalls.length === 0) {
      console.log('   ℹ️  No API calls captured');
    } else {
      console.log(`   📡 ${apiCalls.length} API call(s):`);
      apiCalls.forEach((c: any, i: number) => {
        const status = c.error ? `❌ Error: ${c.error}` :
                       c.status >= 400 ? `⚠️  ${c.status} ${c.statusText}` :
                       `✅ ${c.status}`;
        console.log(`     ${i + 1}. [${c.method}] ${c.url.substring(0, 60)}${c.url.length > 60 ? '...' : ''}`);
        console.log(`        ${status} (${c.time}ms)`);
      });
    }

    // Console Warnings (brief)
    console.log('\n🔍 CONSOLE WARNINGS:');
    if (consoleWarnings.length === 0) {
      console.log('   ✅ No console warnings');
    } else {
      console.log(`   ⚠️  ${consoleWarnings.length} warning(s) (showing first 5):`);
      consoleWarnings.slice(0, 5).forEach((w: any, i: number) => {
        console.log(`     ${i + 1}. ${w.args.join(' ').substring(0, 150)}...`);
      });
    }

    // Check visible content
    console.log('\n🔍 PAGE CONTENT CHECK:');
    const hasIdeasSection = bodyText.includes('Idea') || bodyText.includes('idea');
    const hasResearchSection = bodyText.includes('Research') || bodyText.includes('research');
    const hasAIContent = bodyText.includes('AI') || bodyText.includes('Generate');

    console.log(`   ${hasIdeasSection ? '✅' : '❓'} Ideas/Ideation content: ${hasIdeasSection ? 'Found' : 'Not found'}`);
    console.log(`   ${hasResearchSection ? '✅' : '❓'} Research content: ${hasResearchSection ? 'Found' : 'Not found'}`);
    console.log(`   ${hasAIContent ? '✅' : '❓'} AI/Generate content: ${hasAIContent ? 'Found' : 'Not found'}`);

    // Final Summary
    console.log('\n' + '=' .repeat(80));
    console.log('                            SUMMARY');
    console.log('=' .repeat(80));

    const hasIssues = hasApplicationError || errors.length > 0 || typeErrors.length > 0;

    if (hasIssues) {
      console.log('\n❌ VALIDATION FAILED - Issues found:');
      if (hasApplicationError) console.log('   - Application error displayed on page');
      if (errors.length > 0) console.log(`   - ${errors.length} runtime error(s)`);
      if (typeErrors.length > 0) console.log(`   - ${typeErrors.length} TypeError(s)`);
    } else {
      console.log('\n✅ VALIDATION PASSED - No critical issues found');
    }

    console.log(`\n📊 Stats:`);
    console.log(`   - Load time: ${loadTime}ms`);
    console.log(`   - API calls: ${apiCalls.length}`);
    console.log(`   - Console errors: ${consoleErrors.length + playwrightErrors.length}`);
    console.log(`   - Console warnings: ${consoleWarnings.length}`);
    console.log('');

    await browser.close();

    // Return exit code based on validation
    process.exit(hasIssues ? 1 : 0);

  } catch (error: any) {
    console.error('\n❌ VALIDATION FAILED WITH EXCEPTION:');
    console.error(`   ${error.message}`);
    if (error.stack) {
      console.error('\nStack trace:');
      console.error(error.stack);
    }

    if (browser) {
      await browser.close();
    }

    process.exit(1);
  }
}

validateIdeationPage();
