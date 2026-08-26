const { chromium } = require('playwright-core');

async function main() {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || '/home/kaiser/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome',
    headless: true,
    args: [
      '--no-sandbox', '--headless=new', '--use-angle=vulkan',
      '--enable-features=Vulkan', '--disable-vulkan-surface',
      '--enable-unsafe-webgpu', '--enable-webgpu-developer-features',
    ],
  });
  const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
  page.on('console', (msg) => console.log('[console]', msg.text()));
  page.on('pageerror', (err) => console.log('[pageerror]', err.message));

  await page.goto('http://127.0.0.1:8974/index_b16.html', { waitUntil: 'load', timeout: 30000 });

  await page.waitForFunction(
    () => window.__demoDone === true,
    undefined,
    { timeout: 300000 },
  );

  const result = await page.evaluate(() => window.__demoResult || null);
  const error = await page.evaluate(() => window.__demoError || null);
  console.log('RESULT:', JSON.stringify(result));
  if (error) console.log('DEMO_ERROR:', error);

  await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
